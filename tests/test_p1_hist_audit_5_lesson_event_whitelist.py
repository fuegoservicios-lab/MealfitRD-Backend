"""[P1-HIST-AUDIT-5 · 2026-05-09] Tests: ``api_plans_lessons_counts``
filtra por whitelist de events semánticos + drift detection
cross-archivo contra `cron_tasks.py`.

Bug original (audit historial 2026-05-08):
    El endpoint contaba TODAS las filas de
    ``chunk_lesson_telemetry`` con `meal_plan_id IS NOT NULL`,
    incluyendo events mecánicos (`synth_schema_invalid`,
    `learning_rebuild_failed`, `failed_chunk_skipped_for_learning`,
    etc.). El chip "X lecciones" en `History.jsx` mentía: un plan
    con solo descartes de síntesis aparecía con un chip "8 lecciones"
    aunque el sistema NO había aprendido nada de él.

Fix:
    Whitelist explícita de 4 events semánticos
    (`lesson_synthesized_low_confidence`, `synth_propagated_to_prompt`,
    `recent_lessons_partial_synthesis`, `indefinite_pause_unblocked`).
    Filter `event = ANY(%s)` en la query.

Cobertura:
    - Anchor del marker en el endpoint.
    - Whitelist module-level definida con los 4 events esperados.
    - SQL contiene `event = ANY` y NO un COUNT(*) sin filter.
    - Drift detection cross-archivo: cada `event="..."` emitido por
      `cron_tasks.py` está clasificado (whitelist o blacklist
      explícita en este test). Si se añade un event nuevo sin
      clasificar, el test falla — el operador DEBE decidir si cuenta
      como lección.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"
_CRON_TASKS_PY = _BACKEND_ROOT / "cron_tasks.py"


# ---------------------------------------------------------------------------
# Catálogo manual de classification (espejo de la whitelist en
# routers/plans.py + blacklist explícita). Si cron_tasks.py emite un
# event NUEVO no presente en NINGUNO de estos sets, el drift test
# falla. Mantener AMBOS sites en sync.
# ---------------------------------------------------------------------------
_LESSON_EVENTS_EXPECTED = {
    "lesson_synthesized_low_confidence",
    "synth_propagated_to_prompt",
    "recent_lessons_partial_synthesis",
    "indefinite_pause_unblocked",
}

_MECHANICAL_EVENTS_EXPECTED = {
    "synth_schema_invalid",
    "synth_schema_partial_invalid",
    "learning_rebuild_failed",
    "failed_chunk_skipped_for_learning",
    "lifetime_proxy_ratio_exceeded",
    # [P2-CHUNK-9] Telemetría del override del gate temporal/aprendizaje
    # (flexible_mode bypassea el gate). Es MÉTRICA mecánica — cuantifica
    # cuántos chunks se generan SIN aprendizaje continuo, NO una lección
    # semántica. Por eso NO está en `LESSON_COUNT_EVENT_WHITELIST` (constants.py)
    # pero SÍ en `CHUNK_LESSON_TELEMETRY_VALID_EVENTS`.
    "temporal_gate_override",
}


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_endpoint():
    from routers.plans import api_plans_lessons_counts
    src = inspect.getsource(api_plans_lessons_counts)
    assert "P1-HIST-AUDIT-5" in src, (
        "api_plans_lessons_counts debe mencionar el marker P1-HIST-AUDIT-5 "
        "para que un grep desde memoria/CLAUDE.md encuentre el cierre."
    )


# ---------------------------------------------------------------------------
# 2. Whitelist module-level
# ---------------------------------------------------------------------------
def test_whitelist_constant_present_with_expected_events():
    """La constante `_LESSON_COUNT_EVENT_WHITELIST` está definida en
    `routers/plans.py` y contiene los 4 events semánticos.
    """
    from routers import plans as plans_module
    whitelist = getattr(plans_module, "_LESSON_COUNT_EVENT_WHITELIST", None)
    assert whitelist is not None, (
        "_LESSON_COUNT_EVENT_WHITELIST no está definido en routers/plans.py"
    )
    assert set(whitelist) == _LESSON_EVENTS_EXPECTED, (
        f"Whitelist diverge del catálogo esperado:\n"
        f"  esperado: {sorted(_LESSON_EVENTS_EXPECTED)}\n"
        f"  actual:   {sorted(whitelist)}"
    )


# ---------------------------------------------------------------------------
# 3. SQL contract
# ---------------------------------------------------------------------------
def test_sql_uses_event_any_filter():
    from routers.plans import api_plans_lessons_counts
    src = inspect.getsource(api_plans_lessons_counts)
    # Aislar el SQL del SELECT (entre execute_sql_query y la cláusula
    # GROUP BY).
    sql_match = re.search(
        r"SELECT[\s\S]*?GROUP BY\s+meal_plan_id",
        src,
        re.IGNORECASE,
    )
    assert sql_match is not None, (
        "No se encontró el SELECT con GROUP BY en api_plans_lessons_counts"
    )
    sql = sql_match.group(0)
    assert re.search(r"event\s*=\s*ANY\s*\(\s*%s\s*\)", sql, re.IGNORECASE), (
        "El SQL debe filtrar por `event = ANY(%s)` con la whitelist como "
        "param. P1-HIST-AUDIT-5 — sin esto, el chip cuenta events mecánicos."
    )


def test_whitelist_is_passed_as_query_param():
    """La whitelist se pasa como param Python (no como SQL string
    interpolado), evitando inyección y permitiendo que Postgres use
    índices."""
    from routers.plans import api_plans_lessons_counts
    src = inspect.getsource(api_plans_lessons_counts)
    # `(verified_user_id, list(_LESSON_COUNT_EVENT_WHITELIST))` o similar.
    assert re.search(
        r"verified_user_id\s*,\s*list\s*\(\s*_LESSON_COUNT_EVENT_WHITELIST",
        src,
    ), (
        "La whitelist debe pasarse como param Python (no interpolada). "
        "Patrón esperado: `(verified_user_id, list(_LESSON_COUNT_EVENT_WHITELIST))`."
    )


# ---------------------------------------------------------------------------
# 4. Drift detection cross-archivo: cron_tasks.py emite events que
#    están clasificados (whitelist o blacklist) — sin sorpresas.
# ---------------------------------------------------------------------------
def _events_emitted_by_cron_tasks() -> set[str]:
    """Extrae todos los eventos `event="..."` literales en
    `cron_tasks.py`. Los call sites son del helper
    `_record_chunk_lesson_telemetry`, que recibe `event=<string>`.
    """
    text = _CRON_TASKS_PY.read_text(encoding="utf-8")
    # Patrón: `event="<lower_underscore_string>",` con coma final.
    # Filtra falsos positivos como `event="metric"` de _emit_progress.
    raw_matches = re.findall(
        r'event=(["\'])([a-z][a-z0-9_]+)\1',
        text,
    )
    # raw_matches es lista de tuplas (quote_char, event_name).
    return {m[1] for m in raw_matches}


def test_no_unclassified_lesson_events_in_cron_tasks():
    """Cada event emitido en `cron_tasks.py` (call sites de
    `_record_chunk_lesson_telemetry`) debe estar clasificado en
    `_LESSON_EVENTS_EXPECTED` o `_MECHANICAL_EVENTS_EXPECTED`.

    Filtramos events conocidos de `_emit_progress` (telemetría de
    streaming, no de aprendizaje): `metric`, `token`, `done`, `error`,
    `start`, etc. — esos viven en `graph_orchestrator.py` y no son
    chunk_lesson_telemetry.

    Si un developer agrega un nuevo event a un call site de
    `_record_chunk_lesson_telemetry` SIN actualizar la whitelist en
    `routers/plans.py` ni la blacklist en este test, el test falla.
    """
    # Events de la telemetría chunk + filtros de eventos NO de
    # chunk_lesson_telemetry que también aparecen como `event="..."`
    # (sse, progress streaming).
    _NON_LESSON_EVENT_KEYWORDS = {
        "metric", "token", "done", "error", "start", "end",
        "progress", "cancel", "retry",
    }

    emitted = _events_emitted_by_cron_tasks()
    classified = _LESSON_EVENTS_EXPECTED | _MECHANICAL_EVENTS_EXPECTED
    unclassified = emitted - classified - _NON_LESSON_EVENT_KEYWORDS

    assert not unclassified, (
        f"Events emitidos en cron_tasks.py SIN clasificar: {sorted(unclassified)}\n"
        f"Decide si cada uno es:\n"
        f"  (a) LECCIÓN semántica → agregar a `_LESSON_COUNT_EVENT_WHITELIST` "
        f"en routers/plans.py Y `_LESSON_EVENTS_EXPECTED` en este test.\n"
        f"  (b) MÉTRICA mecánica → agregar a `_MECHANICAL_EVENTS_EXPECTED` "
        f"en este test (no afecta el endpoint).\n"
        f"  (c) NO ES de chunk_lesson_telemetry (e.g., evento SSE/progress) "
        f"→ agregar a `_NON_LESSON_EVENT_KEYWORDS` en este test."
    )


def test_whitelist_subset_of_emitted_events():
    """Cada event en `_LESSON_COUNT_EVENT_WHITELIST` DEBE ser emitido
    por algún call site del backend. Si no lo es, está muerto y debe
    removerse — sin esto, la whitelist puede acumular events que ya
    no genera el sistema.
    """
    emitted = _events_emitted_by_cron_tasks()
    whitelist = _LESSON_EVENTS_EXPECTED
    dead_events = whitelist - emitted
    assert not dead_events, (
        f"La whitelist contiene events que NADIE emite en cron_tasks.py: "
        f"{sorted(dead_events)}. Si el call site fue removido en un "
        f"refactor, también remover de `_LESSON_COUNT_EVENT_WHITELIST`."
    )


# ---------------------------------------------------------------------------
# 5. Sin regresión P1-HIST-3 (sigue excluyendo meal_plan_id IS NULL)
# ---------------------------------------------------------------------------
def test_sql_still_excludes_null_meal_plan_id():
    """Post-P0-HIST-3, los rows con `meal_plan_id IS NULL` (planes
    eliminados) NO deben contar — coherente con cron P2-HIST-5 que los
    GC."""
    from routers.plans import api_plans_lessons_counts
    src = inspect.getsource(api_plans_lessons_counts)
    assert "meal_plan_id IS NOT NULL" in src
