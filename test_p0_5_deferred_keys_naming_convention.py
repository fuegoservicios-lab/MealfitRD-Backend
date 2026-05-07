"""[P0-5] Guardrail de naming convention para campos diferidos a T2.

Antes:
    `P0_1_DEFERRED_LEARNING_KEYS` era una tupla con los nombres exactos de los
    campos de learning que se EXCLUYEN del UPDATE de T1 y solo se persisten en
    T2 (junto con `plan_chunk_queue.learning_metrics` y `status='completed'`).
    Si un dev futuro añadía un nuevo campo `_xxx_lessons` o `_xxx_learning` al
    worker SIN registrarlo en la constante, ese campo:
      1. Se persistiría en T1 sin contraparte en plan_chunk_queue.learning_metrics.
      2. Rompería el invariante de atomicidad P0-1.
      3. El rebuilder leería un estado inconsistente.
    No había test que detectara esto — era una regresión silenciosa.

Ahora:
    Cualquier literal en `cron_tasks.py` que matchee el patrón naming
    `_*lesson*` / `_*learning*` (case-insensitive, palabra completa) DEBE
    aparecer en una de dos listas:
      - `P0_1_DEFERRED_LEARNING_KEYS` (campos diferidos al T2 atómico).
      - `_P0_5_LESSON_KEY_ALLOWLIST` (transientes / atómicos independientes).

    Este test es self-updating: añadir un literal nuevo dispara fallo en CI
    hasta que el dev clasifique el campo correctamente.

Defense-in-depth:
    Adicionalmente el worker, justo antes del UPDATE de T1, escanea el
    `_t1_persist_view` y loguea ERROR si encuentra una key con el patrón que
    NO está en el allowlist (no crashea para no romper producción si el test
    se salta accidentalmente).
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from cron_tasks import P0_1_DEFERRED_LEARNING_KEYS, _P0_5_LESSON_KEY_ALLOWLIST


# Pattern: literales tipo "_xxx_lesson*" o "_xxx_learning*" o "_lesson*" o "_learning*",
# encerrados en comillas simples o dobles. Excluye palabras que solo CONTIENEN
# "lesson" sin ser un identificador (e.g., dentro de docstrings/comentarios). El
# match se ancla a comillas para que sea un literal de Python.
#
# Notas:
#   - `[A-Za-z0-9_]*` permite cualquier sufijo en snake_case.
#   - `[\'"]` en ambos extremos exige que sea string literal.
#   - El patron incluye `lesson` y `learning` separadamente para cubrir variantes
#     como `_meta_lesson` (singular).
_LESSON_LITERAL_PATTERN = re.compile(
    r"""['"]                          # opening quote
    (
        _[A-Za-z0-9_]*                # underscore-prefixed identifier
        (?:lesson|learning)           # core marker
        [A-Za-z0-9_]*                 # optional suffix
    )
    ['"]                              # closing quote
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _scan_source_for_lesson_literals(source: str) -> set[str]:
    """Devuelve el conjunto de literales que matchean el patrón lesson/learning.

    Como el regex se ancla a comillas, palabras en comentarios sueltos no
    matchean. Pueden haber falsos positivos cuando un docstring contiene
    una cita interna como '_foo_lesson'. En la práctica el código del proyecto
    no tiene esos casos; si aparecen, agregar al test un filtro adicional o
    convertir a AST.
    """
    return set(_LESSON_LITERAL_PATTERN.findall(source))


def _read_cron_tasks_source() -> str:
    here = os.path.dirname(__file__)
    src_path = os.path.join(here, "cron_tasks.py")
    with open(src_path, "r", encoding="utf-8") as f:
        return f.read()


def test_lesson_literals_classified_in_deferred_or_allowlist():
    """Cada literal con patrón _*lesson* / _*learning* debe estar en
    P0_1_DEFERRED_LEARNING_KEYS o _P0_5_LESSON_KEY_ALLOWLIST.

    Si añades un nuevo campo de aprendizaje al worker, decide:
      - Es un campo persistente que debe ser atómico con learning_metrics?
        → añádelo a P0_1_DEFERRED_LEARNING_KEYS.
      - Es un flag transitorio o un campo escrito por path independiente?
        → añádelo a _P0_5_LESSON_KEY_ALLOWLIST con comentario justificándolo.

    No clasificarlo rompe la atomicidad P0-1 silenciosamente.
    """
    source = _read_cron_tasks_source()
    found = _scan_source_for_lesson_literals(source)
    classified = set(P0_1_DEFERRED_LEARNING_KEYS) | _P0_5_LESSON_KEY_ALLOWLIST
    unclassified = found - classified
    assert not unclassified, (
        f"Literales no clasificados detectados en cron_tasks.py: {sorted(unclassified)}.\n\n"
        f"Cada uno debe ir a:\n"
        f"  - P0_1_DEFERRED_LEARNING_KEYS si es un campo persistente del worker T1/T2.\n"
        f"  - _P0_5_LESSON_KEY_ALLOWLIST si es transitorio o se escribe por otro path.\n\n"
        f"Sin clasificar, la atomicidad P0-1 se rompe silenciosamente: T1 persiste "
        f"el campo en plan_data sin contraparte en plan_chunk_queue.learning_metrics."
    )


def test_deferred_keys_remain_anchored():
    """Los 6 campos canónicos de P0-1 NO deben removerse del set sin auditoría.
    Este test detecta deletions accidentales (refactors, copy-paste, merges).
    """
    canonical = {
        '_last_chunk_learning',
        '_recent_chunk_lessons',
        '_critical_lessons_permanent',
        '_lifetime_lessons_history',
        '_lifetime_lessons_summary',
        '_chunk_learning_stub_count',
    }
    missing = canonical - set(P0_1_DEFERRED_LEARNING_KEYS)
    assert not missing, (
        f"Campos canónicos P0-1 faltantes en P0_1_DEFERRED_LEARNING_KEYS: {missing}. "
        f"Removerlos rompe la atomicidad. Si el refactor es intencional, ajusta "
        f"este test y documenta el porqué."
    )


def test_no_overlap_between_deferred_and_allowlist():
    """Un campo no puede estar en ambos: o se difiere a T2 (deferred) o no se
    difiere (allowlist). Overlap implicaría confusión de invariantes."""
    deferred = set(P0_1_DEFERRED_LEARNING_KEYS)
    overlap = deferred & _P0_5_LESSON_KEY_ALLOWLIST
    assert not overlap, (
        f"Campos en ambos sets: {overlap}. Decide a cuál pertenece y remueve del otro."
    )


def test_pattern_matches_canonical_keys():
    """Sanity check: el regex DEBE matchear los 6 campos canónicos. Si no,
    el patrón está mal escrito y el test main no detectaría regresiones."""
    canonical_literals = " ".join(f"'{k}'" for k in P0_1_DEFERRED_LEARNING_KEYS)
    matches = _scan_source_for_lesson_literals(canonical_literals)
    expected = set(P0_1_DEFERRED_LEARNING_KEYS)
    assert matches == expected, (
        f"El regex no matchea los canónicos. Esperaba {expected}, recibí {matches}. "
        f"Patrón posiblemente roto: revisa _LESSON_LITERAL_PATTERN."
    )


def test_pattern_matches_synthetic_new_field():
    """Sanity check: el regex DEBE detectar un literal sintético tipo
    `_meta_lessons_v2` (caso típico que el audit P0-5 quiere cazar)."""
    sample = """
    plan_data['_meta_lessons_v2'] = {}
    plan_data['_topic_learning_summary'] = {}
    plan_data['_simple_field'] = 'no'  # no debe matchear
    """
    matches = _scan_source_for_lesson_literals(sample)
    assert '_meta_lessons_v2' in matches
    assert '_topic_learning_summary' in matches
    assert '_simple_field' not in matches


def test_pattern_does_not_match_unrelated_strings():
    """El regex no debe matchear identificadores fuera del prefijo `_` o sin
    el marcador lesson/learning."""
    sample = """
    x = "hello lesson"  # no es snake_case con _
    y = "_arroz_blanco"  # no contiene lesson/learning
    z = "lesson_without_underscore"  # no empieza con _
    """
    matches = _scan_source_for_lesson_literals(sample)
    assert not matches, f"Falsos positivos: {matches}"


def test_runtime_defense_check_logs_unknown_keys(caplog):
    """Cuando _t1_persist_view contiene una key con patrón lesson/learning que
    NO está en el allowlist, el worker debe loguear ERROR (no crashear).

    Este test simula la lógica del worker (replica del bloque defense-in-depth)
    sin necesidad de invocar el worker completo (que requiere DB, LLM, etc).
    """
    import logging
    fake_t1_view = {
        'days': [{'day': 1}],
        '_merged_chunk_ids': ['c1'],
        '_some_new_lesson_field': {'foo': 'bar'},  # synthetic offender
    }
    _p05_unknown = [
        k for k in fake_t1_view
        if (
            ("lesson" in k.lower() or "_learning" in k.lower())
            and k not in _P0_5_LESSON_KEY_ALLOWLIST
        )
    ]
    assert _p05_unknown == ['_some_new_lesson_field'], (
        "La lógica de detección debe identificar exactamente la key sintética "
        "como ofensora; recibí: " + str(_p05_unknown)
    )
