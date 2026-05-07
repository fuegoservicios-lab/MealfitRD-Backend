"""[P1-CHUNKS-4] Invariante del state-machine de `plan_chunk_queue.status`.

Cierra el drift entre `_P0_2_CHUNK_TERMINAL_STATES`, `_CHUNK_STATUS_CANONICAL_STATES`
y la doc-string del state-machine en `process_plan_chunk_queue`. Si un futuro
PR añade un estado nuevo (e.g., `paused_billing`) y olvida actualizar uno de los
sets o la doc, este test falla en CI con un mensaje accionable.

Tres niveles de invariante:

  1. ⊆ canonical:  `_P0_2_CHUNK_TERMINAL_STATES` debe ser subset de
                    `_CHUNK_STATUS_CANONICAL_STATES`.
  2. terminales documentados: `_P0_2_CHUNK_TERMINAL_STATES` cubre exactamente
                    los 3 estados terminales del doc (`completed`, `failed`, `cancelled`).
  3. drift en literales: cada literal `status='xxx'` o `status IN (...)`
                    encontrado en bloques SQL que mencionan `plan_chunk_queue`
                    DEBE estar en `_CHUNK_STATUS_CANONICAL_STATES`. Cubre los
                    archivos `cron_tasks.py`, `routers/plans.py`, `db_plans.py`
                    y `db_profiles.py`.

El scoping por mención de `plan_chunk_queue` en el mismo string literal evita
false positives con `meal_plans.status = 'active'` y similares.
"""
import ast
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_SOURCE_FILES = (
    "cron_tasks.py",
    "routers/plans.py",
    "db_plans.py",
    "db_profiles.py",
)

# Captura `status = 'xxx'` (write o read literal) tras un word-boundary que
# excluye prefijos como `subscription_status`, `reservation_status`,
# `generation_status`, `chunk_status`. El underscore es \w → no hay boundary
# antes de `status` cuando viene precedido por `_`.
_STATUS_EQ_PATTERN = re.compile(r"\bstatus\b\s*=\s*'([a-z_]+)'", re.IGNORECASE)
# Captura `status IN ('a', 'b', ...)` para listar todos los valores referenciados.
_STATUS_IN_PATTERN = re.compile(
    r"\bstatus\b\s+IN\s*\(\s*((?:'[a-z_]+'\s*,?\s*)+)\)",
    re.IGNORECASE,
)
_QUOTED_LITERAL = re.compile(r"'([a-z_]+)'")


def _extract_status_literals_from_sql_string(sql: str) -> set[str]:
    """Extrae todos los valores de status referenciados en un fragmento SQL.

    Captura tanto `status = 'xxx'` como `status IN ('a', 'b')`.
    """
    out: set[str] = set()
    for m in _STATUS_EQ_PATTERN.finditer(sql):
        out.add(m.group(1).lower())
    for m in _STATUS_IN_PATTERN.finditer(sql):
        for v in _QUOTED_LITERAL.findall(m.group(1)):
            out.add(v.lower())
    return out


def _scoped_status_literals(file_text: str) -> set[str]:
    """Walk `ast.Constant` strings; quédate solo con los que mencionan
    `plan_chunk_queue` y extrae sus literales de status.

    Con esto evitamos falsos positivos en queries sobre `meal_plans.status='active'`
    (que viven en strings que NO mencionan `plan_chunk_queue`).
    """
    try:
        tree = ast.parse(file_text)
    except SyntaxError:
        # Si el archivo no parsea (no debería pasar en producción), no
        # bloqueamos el test — el invariante dependía de él pero CI ya falla
        # con SyntaxError aguas arriba.
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if "plan_chunk_queue" in s:
                out |= _extract_status_literals_from_sql_string(s)
    return out


def _read_file(rel: str) -> str:
    p = _BACKEND_DIR / rel
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Invariantes
# ---------------------------------------------------------------------------
def test_terminal_states_subset_of_canonical():
    """`_P0_2_CHUNK_TERMINAL_STATES ⊆ _CHUNK_STATUS_CANONICAL_STATES`.

    Si un terminal nuevo se añade sin que también esté en el universe canónico,
    `_validate_chunk_pre_llm` rechazaría chunks vivos por error o aceptaría
    chunks ya muertos.
    """
    from cron_tasks import _CHUNK_STATUS_CANONICAL_STATES, _P0_2_CHUNK_TERMINAL_STATES

    terminal_set = set(_P0_2_CHUNK_TERMINAL_STATES)
    canonical_set = set(_CHUNK_STATUS_CANONICAL_STATES)
    drift = terminal_set - canonical_set
    assert not drift, (
        f"Estados en _P0_2_CHUNK_TERMINAL_STATES no presentes en "
        f"_CHUNK_STATUS_CANONICAL_STATES: {sorted(drift)}. "
        f"Si añadiste un terminal nuevo, agrégalo al universe canónico."
    )


def test_terminal_states_match_documented_terminals():
    """`_P0_2_CHUNK_TERMINAL_STATES` cubre exactamente los 3 terminales del doc.

    El state-machine documentado lista `completed`, `failed`, `cancelled` como
    los únicos terminales. Cualquier divergencia indica drift entre la doc del
    state-machine y la implementación de `_validate_chunk_pre_llm`.
    """
    from cron_tasks import _P0_2_CHUNK_TERMINAL_STATES

    expected = {"completed", "failed", "cancelled"}
    actual = set(_P0_2_CHUNK_TERMINAL_STATES)
    assert actual == expected, (
        f"_P0_2_CHUNK_TERMINAL_STATES drifteó del doc: "
        f"esperaba {sorted(expected)}, encontró {sorted(actual)}. "
        f"Si la semántica de un estado cambió, actualiza la doc-string del "
        f"state-machine en process_plan_chunk_queue."
    )


def test_canonical_set_size_matches_documented_count():
    """El doc lista 7 estados (+ sub-estados de pending_user_action).

    El número 7 está hardcoded en la doc del state-machine; este test
    detecta si alguien añadió/quitó un estado del frozenset sin actualizar
    la sección "Estados (7 + sub-estados...)" en la doc.
    """
    from cron_tasks import _CHUNK_STATUS_CANONICAL_STATES

    assert len(_CHUNK_STATUS_CANONICAL_STATES) == 7, (
        f"_CHUNK_STATUS_CANONICAL_STATES tiene {len(_CHUNK_STATUS_CANONICAL_STATES)} "
        f"estados (esperaba 7). Si añadiste/quitaste un estado, actualiza:\n"
        f"  1. La sección 'Estados (N + sub-estados...)' del state-machine doc\n"
        f"  2. Este test con el nuevo número esperado\n"
        f"  3. La tabla de transiciones del doc"
    )


def test_canonical_set_contains_required_named_states():
    """Verifica explícitamente que los nombres de los 7 estados estén presentes.

    Cubre el caso patológico donde alguien renombra un estado (e.g.,
    `pending` → `queued`) sin cambiar el nombre en la doc o en otros sites:
    el test fuerza que los 7 nombres canónicos sigan existiendo.
    """
    from cron_tasks import _CHUNK_STATUS_CANONICAL_STATES

    expected = {
        "pending",
        "processing",
        "stale",
        "pending_user_action",
        "completed",
        "failed",
        "cancelled",
    }
    missing = expected - set(_CHUNK_STATUS_CANONICAL_STATES)
    extra = set(_CHUNK_STATUS_CANONICAL_STATES) - expected
    assert not missing and not extra, (
        f"_CHUNK_STATUS_CANONICAL_STATES drifteó. "
        f"Faltan: {sorted(missing)}. Extra: {sorted(extra)}. "
        f"Si renombraste un estado, actualiza este test, la doc, y todos los "
        f"call sites."
    )


def test_no_status_literal_drift_across_source_files():
    """Scan SQL en cron_tasks/routers/plans/db_plans/db_profiles para detectar
    literales `status='xxx'` no registrados en el SSOT.

    Scoping por mención de `plan_chunk_queue` en el mismo string literal evita
    falsos positivos sobre tablas distintas (e.g., `meal_plans.status='active'`).
    """
    from cron_tasks import _CHUNK_STATUS_CANONICAL_STATES

    seen: dict[str, set[str]] = {}
    union: set[str] = set()
    for rel in _SOURCE_FILES:
        text = _read_file(rel)
        if not text:
            continue
        values = _scoped_status_literals(text)
        if values:
            seen[rel] = values
            union |= values

    drift = union - set(_CHUNK_STATUS_CANONICAL_STATES)
    assert not drift, (
        f"Literales status='...' fuera del SSOT _CHUNK_STATUS_CANONICAL_STATES: "
        f"{sorted(drift)}.\n"
        f"Origen por archivo: { {f: sorted(v) for f, v in seen.items()} }\n"
        f"Si añadiste un estado nuevo, actualiza:\n"
        f"  1. _CHUNK_STATUS_CANONICAL_STATES en cron_tasks.py\n"
        f"  2. La doc-string de process_plan_chunk_queue\n"
        f"  3. _P0_2_CHUNK_TERMINAL_STATES si es terminal"
    )


def test_canonical_set_states_appear_in_state_machine_docstring():
    """Cada estado del SSOT debe aparecer textualmente en la doc-string del
    state-machine. Cierra el drift "constante actualizado pero doc no".

    Solo verificamos que cada nombre aparezca; el orden/formato del cuadro
    box-drawing puede cambiar sin romper este invariante.
    """
    from cron_tasks import _CHUNK_STATUS_CANONICAL_STATES, process_plan_chunk_queue

    doc = (process_plan_chunk_queue.__doc__ or "")
    missing = [s for s in _CHUNK_STATUS_CANONICAL_STATES if s not in doc]
    assert not missing, (
        f"Estados ausentes en la doc-string del state-machine: {sorted(missing)}. "
        f"Si añadiste un estado al SSOT, agrégalo a la sección 'Estados (N + ...)' "
        f"y a la tabla de transiciones."
    )


def test_canonical_set_is_immutable_frozenset():
    """`_CHUNK_STATUS_CANONICAL_STATES` debe ser frozenset para que el test
    de drift no se pueda evadir con `_CHUNK_STATUS_CANONICAL_STATES.add(...)`.
    """
    from cron_tasks import _CHUNK_STATUS_CANONICAL_STATES

    assert isinstance(_CHUNK_STATUS_CANONICAL_STATES, frozenset), (
        f"_CHUNK_STATUS_CANONICAL_STATES debe ser frozenset (encontrado: "
        f"{type(_CHUNK_STATUS_CANONICAL_STATES).__name__}) para evitar mutación "
        f"runtime que evada el invariante."
    )


# ---------------------------------------------------------------------------
# Sanity check: el helper debe rechazar drift simulado
# ---------------------------------------------------------------------------
def test_helper_extracts_drift_correctly():
    """Sanity: si inyectamos un literal `status='unknown_state'` en un string
    que menciona `plan_chunk_queue`, el helper debe capturarlo. Garantiza
    que el drift detector NO devuelve set vacío silenciosamente.
    """
    fake_source = (
        '"""docstring"""\n'
        'def f():\n'
        '    sql = "UPDATE plan_chunk_queue SET status = \'unknown_state\' WHERE id = 1"\n'
        '    return sql\n'
    )
    found = _scoped_status_literals(fake_source)
    assert "unknown_state" in found, (
        f"Helper falló al detectar drift sintético; vio: {sorted(found)}"
    )


def test_helper_skips_non_plan_chunk_queue_strings():
    """Sanity inverso: literales en strings que NO mencionan `plan_chunk_queue`
    deben ignorarse. Evita falsos positivos en queries de otras tablas.
    """
    fake_source = (
        '"""docstring"""\n'
        'def f():\n'
        '    sql = "SELECT * FROM meal_plans WHERE status = \'active\'"\n'
        '    return sql\n'
    )
    found = _scoped_status_literals(fake_source)
    assert found == set(), (
        f"Helper capturó status literal de tabla no-chunk_queue: {sorted(found)}"
    )
