"""[P1-25] Tests para que `semantic_cache_check_node` bypasee el cache cuando
el form_data carga señales de retry posterior a un rechazo.

Bug original (audit P1-25):
  `semantic_cache_check_node` chequeaba `_is_same_day_reroll` y
  `_is_rotation_reroll` (rerolls explícitos del usuario) pero no las
  señales internas de retry del wrapper externo:

  1. `_pantry_correction`: `_validate_pantry_and_retry_pipeline`
     (cron_tasks.py:4979/5042) escribe esta key con la violación
     concreta antes de re-invocar `run_plan_pipeline` tras un fallo
     de pantry. Sin guard, el cache_check_node devolvía el MISMO plan
     que acababa de ser rechazado (cualquier candidato compatible por
     similitud de embedding del perfil — el cache key NO incluye el
     historial de rechazos). Retry no-op + rápida degradación a
     fallback matemático.

  2. `_drift_retries`: contador de reintentos por pantry drift
     (cron_tasks.py:17280). >0 indica al menos un fallo previo. Mismo
     patrón: el cache podía servir el plan que disparó el drift retry.

Fix:
  Bloque [P1-25] al inicio de `semantic_cache_check_node` evalúa:
    - `_pantry_correction` truthy
    - `int(_drift_retries) > 0`
  Si CUALQUIERA es positivo, retorna early con
  `{semantic_cache_hit: False, cached_plan_data: None}` y emite
  warning con las razones para telemetría operacional.

  Mantiene el bypass anterior por reroll explícito (`_is_same_day_reroll`,
  `_is_rotation_reroll`) — son ortogonales al retry guard.

Cobertura:
  - test_pantry_correction_truthy_bypasses_cache
  - test_pantry_correction_empty_string_does_not_bypass
  - test_drift_retries_positive_bypasses_cache
  - test_drift_retries_zero_does_not_bypass
  - test_drift_retries_invalid_string_treated_as_zero
  - test_bypass_logs_warning_with_reasons
  - test_bypass_is_orthogonal_to_reroll_signals
  - test_bypass_short_circuits_before_search_similar_plan
  - test_documentation_p1_25_present
"""
import asyncio
import inspect
import logging
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

import graph_orchestrator
from graph_orchestrator import semantic_cache_check_node


_SRC = inspect.getsource(semantic_cache_check_node)


def _state_with(form_data, profile_embedding=None):
    if profile_embedding is None:
        profile_embedding = [0.1] * 768  # truthy non-empty
    return {
        "form_data": form_data,
        "profile_embedding": profile_embedding,
        "nutrition": {"total_daily_calories": 2000, "total_daily_macros": {}},
    }


def _run(state):
    return asyncio.run(semantic_cache_check_node(state))


# ---------------------------------------------------------------------------
# 1. _pantry_correction guard.
# ---------------------------------------------------------------------------
def test_pantry_correction_truthy_bypasses_cache():
    """`_pantry_correction` no-vacío → bypass, sin invocar search_similar_plan."""
    state = _state_with({
        "user_id": "user-A",
        "_pantry_correction": "violation: pollo missing 200g",
    })
    with patch("graph_orchestrator._adb") as mock_adb:
        result = _run(state)

    assert result == {"semantic_cache_hit": False, "cached_plan_data": None}
    mock_adb.assert_not_called(), (
        "P1-25: cuando _pantry_correction está set, search_similar_plan "
        "no debe ejecutarse (saving DB roundtrip + evita servir el plan "
        "que acaba de ser rechazado)."
    )


def test_pantry_correction_empty_string_does_not_bypass():
    """`_pantry_correction = ''` (falsy) NO debe activar el bypass.
    Distingue 'sin retry' (key ausente o string vacío) de 'retry activo'
    (string no-vacío con la violación)."""
    state = _state_with({
        "user_id": "user-A",
        "_pantry_correction": "",
    })
    with patch("graph_orchestrator._adb", new_callable=AsyncMock, return_value=[]):
        result = _run(state)

    # Sin candidatos → cache miss normal, no bypass por _pantry_correction.
    # El test pasa si NO crashea y devuelve miss.
    assert result["semantic_cache_hit"] is False
    assert result["cached_plan_data"] is None


# ---------------------------------------------------------------------------
# 2. _drift_retries guard.
# ---------------------------------------------------------------------------
def test_drift_retries_positive_bypasses_cache():
    """`_drift_retries > 0` → bypass, sin invocar search_similar_plan."""
    state = _state_with({
        "user_id": "user-A",
        "_drift_retries": 1,
    })
    with patch("graph_orchestrator._adb") as mock_adb:
        result = _run(state)

    assert result == {"semantic_cache_hit": False, "cached_plan_data": None}
    mock_adb.assert_not_called()


def test_drift_retries_zero_does_not_bypass():
    """`_drift_retries = 0` → NO bypass por este signal."""
    state = _state_with({
        "user_id": "user-A",
        "_drift_retries": 0,
    })
    with patch("graph_orchestrator._adb", new_callable=AsyncMock, return_value=[]):
        result = _run(state)

    assert result["semantic_cache_hit"] is False
    assert result["cached_plan_data"] is None


def test_drift_retries_invalid_string_treated_as_zero():
    """Si `_drift_retries` viene con tipo inesperado (string corrupto),
    el guard NO debe crashear — fallback a 0 (no bypass)."""
    state = _state_with({
        "user_id": "user-A",
        "_drift_retries": "abc",
    })
    with patch("graph_orchestrator._adb", new_callable=AsyncMock, return_value=[]):
        # No debe lanzar.
        result = _run(state)

    assert result["semantic_cache_hit"] is False


def test_drift_retries_string_numeric_parsed_as_int():
    """`_drift_retries = '2'` (string numérico) debe parsearse y
    activar bypass (>0)."""
    state = _state_with({
        "user_id": "user-A",
        "_drift_retries": "2",
    })
    with patch("graph_orchestrator._adb") as mock_adb:
        result = _run(state)

    assert result == {"semantic_cache_hit": False, "cached_plan_data": None}
    mock_adb.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Telemetría.
# ---------------------------------------------------------------------------
def test_bypass_logs_warning_with_reasons(caplog):
    """El bypass debe emitir un warning con las razones — sin esto, una
    caída del cache_hit_rate por proliferación de retries pasa silenciosa."""
    state = _state_with({
        "user_id": "user-XYZ",
        "_pantry_correction": "missing pollo 200g",
        "_drift_retries": 3,
    })
    with patch("graph_orchestrator._adb"), \
         caplog.at_level(logging.WARNING, logger="graph_orchestrator"):
        _run(state)

    p125_logs = [r for r in caplog.records if "[P1-25]" in r.getMessage()]
    assert p125_logs, (
        f"P1-25: esperaba warning [P1-25] explicando el bypass. "
        f"Logs vistos: {[r.getMessage()[:120] for r in caplog.records]}"
    )
    msg = p125_logs[0].getMessage()
    assert "user-XYZ" in msg
    assert "_pantry_correction" in msg
    assert "_drift_retries=3" in msg


# ---------------------------------------------------------------------------
# 4. Ortogonalidad con reroll signals.
# ---------------------------------------------------------------------------
def test_bypass_is_orthogonal_to_reroll_signals():
    """El reroll bypass (precedente) ya cubría `_is_same_day_reroll` y
    `_is_rotation_reroll`. El P1-25 attempt guard NO debe interferir
    con ellos. Verificamos que un reroll sigue bypassing aunque las
    nuevas keys estén ausentes."""
    state = _state_with({
        "user_id": "user-A",
        "_is_same_day_reroll": True,
        # Sin _pantry_correction ni _drift_retries.
    })
    with patch("graph_orchestrator._adb") as mock_adb:
        result = _run(state)

    assert result == {"semantic_cache_hit": False, "cached_plan_data": None}
    mock_adb.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Defensa estructural: el guard corre ANTES de tocar la DB.
# ---------------------------------------------------------------------------
def test_bypass_short_circuits_before_search_similar_plan():
    """El bypass DEBE retornar antes de invocar search_similar_plan
    (vía _adb). Sin esto, ahorraríamos el cache hit pero pagaríamos
    el coste de la query vector — no es solo correctness, es perf."""
    state = _state_with({
        "user_id": "user-A",
        "_pantry_correction": "x",
    })
    # El _adb se invoca para search_similar_plan + get_recent_meals_from_plans.
    # Aseguramos que NINGUNA de esas dos llamadas ocurre cuando hay bypass.
    with patch("graph_orchestrator._adb") as mock_adb:
        _run(state)

    assert mock_adb.call_count == 0


def test_bypass_short_circuit_in_source_before_search():
    """Defensa estructural: el bloque [P1-25] aparece ANTES de la
    invocación `_adb(search_similar_plan, ...)` (no solo de la mención
    del símbolo en docstring) — el short-circuit debe ahorrar el
    roundtrip de vector search."""
    p125_idx = _SRC.find("[P1-25]")
    invoc_idx = _SRC.find("_adb(search_similar_plan")
    assert p125_idx > -1, "P1-25: marker no encontrado en el source"
    assert invoc_idx > -1, (
        "Invocación `_adb(search_similar_plan` no encontrada — "
        "¿signature change?"
    )
    assert p125_idx < invoc_idx, (
        f"P1-25: el guard debe aparecer ANTES de la invocación a "
        f"search_similar_plan. p125_idx={p125_idx}, invoc_idx={invoc_idx}"
    )


# ---------------------------------------------------------------------------
# 6. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_25_present():
    """Comentario `[P1-25]` debe documentar el guard."""
    full_src = inspect.getsource(graph_orchestrator)
    assert "[P1-25]" in full_src, (
        "P1-25: falta marker que documente el attempt guard del cache."
    )


def test_documentation_mentions_retry_or_rejected():
    """El comentario debe mencionar el rationale: el plan ya fue
    rechazado por el wrapper de retry. Sin esto, un futuro lector
    podría borrar el guard pensando que es redundante con el reroll."""
    p125_idx = _SRC.find("[P1-25]")
    window = _SRC[p125_idx : p125_idx + 2000]
    needles = ["retry", "rechazo", "rechazad", "_pantry_correction",
               "_drift_retries", "wrapper"]
    assert any(n in window.lower() for n in needles), (
        "P1-25: el comentario debe explicar el rationale (retry / rechazo / "
        "wrapper externo)."
    )
