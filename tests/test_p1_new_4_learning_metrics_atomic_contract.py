"""[P1-NEW-4 · 2026-05-10] Regression guard: `plan_chunk_queue.learning_metrics`
se persiste ATÓMICAMENTE en T1 junto con `days` + `_merged_chunk_ids` +
campos de aprendizaje deferidos (los 6 de `P0_1_DEFERRED_LEARNING_KEYS`).

Bug temido (audit 2026-05-10 — descartado tras verificación):
    Race entre el compute de `learning_metrics` y la marcación
    `status='completed'` permitiría perder telemetría si el worker
    crashea entre ambos.

Verificación post-audit (cron_tasks.py snapshot 2026-05-10):
    El doc block en `cron_tasks.py:177-205` documenta explícitamente
    el cierre P0-1: T1 commitea `days` + `_merged_chunk_ids` +
    campos de learning + `plan_chunk_queue.learning_metrics` dentro
    del MISMO `FOR UPDATE`. Si T2 (shopping/quality/status=completed)
    falla, el retry detecta `_persisted_chunk_id == week_number` y
    salta el backfill — la lección ya está persistida.

    Pre-fix: estos campos se diferían a T2, y un crash entre T1 y T2
    dejaba `plan_chunk_queue.learning_metrics` en NULL. El retry path
    `chunk_already_merged` saltaba el merge y backfilleaba un STUB,
    perdiendo permanentemente las métricas reales.

Este test bloquea regresión del contrato:
    1. La constante `P0_1_DEFERRED_LEARNING_KEYS` existe con los 6
       campos canónicos.
    2. El doc block sobre atomicidad P0-1 sigue presente.
    3. El allowlist `_P0_5_LESSON_KEY_ALLOWLIST` existe (disciplina
       para que un nuevo dev no añada `_meta_lessons_v2` sin
       actualizarlo, rompiendo la atomicidad).
    4. El comentario referencia explícitamente `FOR UPDATE` y
       `plan_chunk_queue.learning_metrics` en el bloque P0-1.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


def _read_source() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. La constante de keys deferidos sigue intacta
# ---------------------------------------------------------------------------
def test_p0_1_deferred_learning_keys_exists_with_six_canonical():
    """`P0_1_DEFERRED_LEARNING_KEYS` debe contener los 6 campos canónicos
    de aprendizaje del worker. Si alguno falta, el merge de T1 puede no
    persistirlo atómicamente."""
    import cron_tasks as ct
    keys = getattr(ct, "P0_1_DEFERRED_LEARNING_KEYS", None)
    assert keys is not None, (
        "Constante `P0_1_DEFERRED_LEARNING_KEYS` desapareció. Sin ella, "
        "no hay una lista canónica de qué campos de learning debe el "
        "merge atómico de T1 persistir."
    )
    expected = {
        "_last_chunk_learning",
        "_recent_chunk_lessons",
        "_critical_lessons_permanent",
        "_lifetime_lessons_history",
        "_lifetime_lessons_summary",
        "_chunk_learning_stub_count",
    }
    actual = set(keys)
    missing = expected - actual
    assert not missing, (
        f"`P0_1_DEFERRED_LEARNING_KEYS` perdió los siguientes campos "
        f"canónicos: {sorted(missing)}. Si se removió intencionalmente, "
        f"verificar que el campo se migró a un path atómico independiente "
        f"(via update_plan_data_atomic) y se documentó en "
        f"`_P0_5_LESSON_KEY_ALLOWLIST`."
    )


# ---------------------------------------------------------------------------
# 2. El doc block del contrato P0-1 sigue presente
# ---------------------------------------------------------------------------
def test_p0_1_atomicity_doc_block_intact():
    """El bloque de doc en `cron_tasks.py:177-205` (P0-1) debe referenciar:
      - "FOR UPDATE" (la transacción T1 que persiste atómicamente).
      - "plan_chunk_queue.learning_metrics" (la columna afectada).
      - "P0-1" o "T1" (anchor a la lección histórica).
    Sin estos términos, un refactor futuro puede romper la atomicidad sin
    señales de revisión."""
    src = _read_source()
    # Localizar el doc block: empieza tras `class ReservationReconciliationExhausted`
    # exception (línea ~165) y termina antes de `P0_1_DEFERRED_LEARNING_KEYS = (`.
    end_match = re.search(r"^P0_1_DEFERRED_LEARNING_KEYS\s*=\s*\(", src, re.MULTILINE)
    assert end_match is not None, (
        "No encuentro la asignación `P0_1_DEFERRED_LEARNING_KEYS = (` que "
        "marca el final del doc block."
    )
    # Tomar los 80 líneas previos para captar el doc block completo.
    lines_before = src[:end_match.start()].splitlines()[-80:]
    doc_block = "\n".join(lines_before)

    required_anchors = [
        "FOR UPDATE",
        "plan_chunk_queue.learning_metrics",
        "P0-1",
    ]
    for anchor in required_anchors:
        assert anchor in doc_block, (
            f"El doc block sobre atomicidad de learning_metrics ya NO "
            f"menciona {anchor!r}. Sin ese anchor, un refactor futuro puede "
            f"desatomizar el merge sin alertar al reviewer."
        )


# ---------------------------------------------------------------------------
# 3. El allowlist de keys "lesson-pattern pero no deferred" sigue presente
# ---------------------------------------------------------------------------
def test_p0_5_lesson_key_allowlist_exists():
    """`_P0_5_LESSON_KEY_ALLOWLIST` debe existir. Es la lista de keys que
    matchean el pattern `_*learning*` / `_*lesson*` pero NO son campos
    persistidos en T1 (son flags de transporte, telemetría, o atómicos
    independientes). Sin él, un test parser scan rejecta keys legítimas
    y el dev se ve forzado a añadirlas a `P0_1_DEFERRED_LEARNING_KEYS`,
    rompiendo la atomicidad."""
    import cron_tasks as ct
    allowlist = getattr(ct, "_P0_5_LESSON_KEY_ALLOWLIST", None)
    assert allowlist is not None, (
        "`_P0_5_LESSON_KEY_ALLOWLIST` desapareció. Sin él, la disciplina "
        "P0-5 (separar campos deferred del worker T1 de flags transitorios) "
        "se pierde y futuros devs añadirán flags a `P0_1_DEFERRED_LEARNING_KEYS` "
        "incorrectamente."
    )
    assert isinstance(allowlist, frozenset), (
        "`_P0_5_LESSON_KEY_ALLOWLIST` debe ser un frozenset (inmutable)."
    )
    # Sanity: debe tener al menos algunos miembros esperados.
    expected_subset = {
        "_learning_provenance",
        "_chunk_lessons",
    }
    assert expected_subset.issubset(allowlist), (
        f"`_P0_5_LESSON_KEY_ALLOWLIST` perdió miembros esperados: "
        f"{sorted(expected_subset - allowlist)}."
    )


# ---------------------------------------------------------------------------
# 4. La UPDATE preflight de learning_metrics sigue presente
# ---------------------------------------------------------------------------
def test_preflight_learning_metrics_update_exists():
    """El UPDATE preflight (cron_tasks.py:~19698) que persiste un stub
    de `learning_metrics` antes de que el chunk arranque su procesamiento
    real debe seguir presente. Es la red de seguridad cuando T1 nunca
    logra completarse (crash pre-merge): al menos los counts prior-only
    quedan persistidos en la columna.

    El SQL vive como Python adjacent-string concatenation (multi-línea)
    así que el regex tolera comillas + whitespace entre tokens."""
    src = _read_source()
    # Tokens separados por `[\s"]+` permite cruzar boundaries de string
    # concatenation tipo `"... "` + newline + indent + `"WHERE..."`.
    pattern = re.compile(
        r"UPDATE[\s\"]+plan_chunk_queue[\s\"]+SET[\s\"]+learning_metrics"
        r"[\s\"]*=[\s\"]*%s::jsonb"
        r"[\s\"]+WHERE[\s\"]+id[\s\"]*=[\s\"]*%s"
        r"[\s\"]+AND[\s\"]+learning_metrics[\s\"]+IS[\s\"]+NULL"
    )
    assert pattern.search(src) is not None, (
        "El UPDATE preflight de `learning_metrics` con guard "
        "`AND learning_metrics IS NULL` desapareció. Sin esa red de "
        "seguridad, un crash pre-merge deja la columna en NULL y el "
        "chunk siguiente backfilea un STUB perdiendo telemetría real."
    )
