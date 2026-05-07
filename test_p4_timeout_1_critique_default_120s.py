"""[P4-TIMEOUT-1 / P4-TIMEOUT-3] Tests para el default de
`MEALFIT_CRITIQUE_FIX_TIMEOUT_S`.

Historia del knob:
  - Pre-P4-TIMEOUT-1: 90s — los 504 con SDK retry interno excedían el cap.
  - P4-TIMEOUT-1 (2026-05-05): bump a 120s. Absorbió ~95% de 504s.
  - P4-TIMEOUT-2 (2026-05-04 03:26): añadió circuit breaker en cascade.
  - P4-TIMEOUT-3 (este): bump a 150s. Cubre el último 5% de casos
    borderline donde 120s no fue suficiente para el SDK retry chain.

Bug observable (corridas 2026-05-05 múltiples):
  Gemini Flash sufrió 504 DEADLINE_EXCEEDED durante self-critique
  correction. Pattern:
    1. Llamada original al corrector se demora ~60-70s (con tools).
    2. Gemini SDK retry interno por 504 añade +2-5s.
    3. Wall-clock total ~85-95s, antes excedía el cap de 90s.
    4. asyncio.TimeoutError → día sin corregir → marker `_critique_unresolved`
       → P1-SURGICAL-1 fuerza regen en retry → costo adicional.

Cobertura:
  - Default es 150s (post P4-TIMEOUT-3)
  - Env var override sigue funcionando
  - El env var conserva tipo float (no int)
"""
import importlib
import os

import pytest


def _reload_module():
    """Recarga graph_orchestrator para releer env vars."""
    import graph_orchestrator
    importlib.reload(graph_orchestrator)
    return graph_orchestrator


# ---------------------------------------------------------------------------
# 1. Default value
# ---------------------------------------------------------------------------
def test_default_is_150s_post_p4_timeout_3_bump(monkeypatch):
    """[P4-TIMEOUT-3] Sin env var, el default es 150s (era 120s tras
    P4-TIMEOUT-1, era 90s pre-cualquier-fix)."""
    monkeypatch.delenv("MEALFIT_CRITIQUE_FIX_TIMEOUT_S", raising=False)
    go = _reload_module()
    assert go.CRITIQUE_FIX_TIMEOUT_S == 150.0, (
        f"Default debe ser 150.0s tras P4-TIMEOUT-3, recibido "
        f"{go.CRITIQUE_FIX_TIMEOUT_S}"
    )


def test_default_is_higher_than_120s_pre_p4_timeout_3():
    """Sanity guard contra regresión accidental al valor previo (120s o menos)."""
    import graph_orchestrator as go
    assert go.CRITIQUE_FIX_TIMEOUT_S > 120.0, (
        f"P4-TIMEOUT-3 bumpeó de 120s a 150s. Si alguien bajó a 120s o menos, "
        f"se introdujo regresión. Actual: {go.CRITIQUE_FIX_TIMEOUT_S}"
    )


def test_default_still_higher_than_90s_pre_any_fix():
    """Sanity guard de 2do orden: el valor original era 90s. Cualquier
    revert por debajo de eso es estrictamente peor que el estado inicial."""
    import graph_orchestrator as go
    assert go.CRITIQUE_FIX_TIMEOUT_S > 90.0, (
        f"P4-TIMEOUT-* bumpeó originalmente de 90s. Actual: {go.CRITIQUE_FIX_TIMEOUT_S}"
    )


# ---------------------------------------------------------------------------
# 2. Env var override
# ---------------------------------------------------------------------------
def test_env_var_override_works(monkeypatch):
    """Operadores pueden subir el cap si ven 504s persistentes."""
    monkeypatch.setenv("MEALFIT_CRITIQUE_FIX_TIMEOUT_S", "180")
    go = _reload_module()
    assert go.CRITIQUE_FIX_TIMEOUT_S == 180.0


def test_env_var_override_preserves_float_type(monkeypatch):
    """El env var debe parsearse como float (no int) — `_env_float` parser."""
    monkeypatch.setenv("MEALFIT_CRITIQUE_FIX_TIMEOUT_S", "150.5")
    go = _reload_module()
    assert go.CRITIQUE_FIX_TIMEOUT_S == 150.5
    assert isinstance(go.CRITIQUE_FIX_TIMEOUT_S, float)


def test_env_var_invalid_falls_to_default(monkeypatch):
    """Valor inválido en env var → fallback a default 150 con warning."""
    monkeypatch.setenv("MEALFIT_CRITIQUE_FIX_TIMEOUT_S", "garbage_value")
    go = _reload_module()
    # `_env_float` cae a default cuando no es parseable
    assert go.CRITIQUE_FIX_TIMEOUT_S == 150.0


def test_env_var_empty_falls_to_default(monkeypatch):
    monkeypatch.setenv("MEALFIT_CRITIQUE_FIX_TIMEOUT_S", "")
    go = _reload_module()
    assert go.CRITIQUE_FIX_TIMEOUT_S == 150.0


# ---------------------------------------------------------------------------
# 3. Sanity: el cap se usa efectivamente en self_critique_node
# ---------------------------------------------------------------------------
def test_critique_fix_timeout_referenced_in_self_critique_node(monkeypatch):
    """[P4-TIMEOUT-1] El módulo graph_orchestrator debe usar
    `CRITIQUE_FIX_TIMEOUT_S` como timeout del corrector LLM.
    Verificamos por substring en el código del módulo."""
    import inspect
    import graph_orchestrator as go
    src = inspect.getsource(go.self_critique_node)
    assert "CRITIQUE_FIX_TIMEOUT_S" in src, (
        "self_critique_node debe usar CRITIQUE_FIX_TIMEOUT_S como timeout. "
        "Si lo removiste o renombraste, el bump P4-TIMEOUT-1 no aplica."
    )


# ---------------------------------------------------------------------------
# 4. Repro de las corridas 2026-05-05
# ---------------------------------------------------------------------------
def test_repro_504_scenarios_now_fit_in_default():
    """Las 3 corridas observadas mostraron timeouts entre 85-95s.
    Verifica que el nuevo default (120s) absorbe ese rango."""
    import graph_orchestrator as go
    # Casos reales observados:
    observed_504_durations_s = [85, 90, 92, 95, 98]
    for duration in observed_504_durations_s:
        assert go.CRITIQUE_FIX_TIMEOUT_S > duration, (
            f"Corrida real con duration={duration}s NO cabe en cap actual "
            f"{go.CRITIQUE_FIX_TIMEOUT_S}s. P4-TIMEOUT-1 quedó corto."
        )

    # Headroom mínimo del 20% sobre el peor caso observado
    worst_observed = max(observed_504_durations_s)
    headroom = (go.CRITIQUE_FIX_TIMEOUT_S - worst_observed) / worst_observed
    assert headroom >= 0.20, (
        f"Headroom sobre worst case observed ({worst_observed}s) es solo "
        f"{headroom:.0%}. Se recomienda al menos 20% para absorber peaks."
    )


# ---------------------------------------------------------------------------
# 5. No regresión en knobs adyacentes
# ---------------------------------------------------------------------------
def test_other_critique_knobs_unaffected():
    """El bump no debe haber tocado otros knobs (hedging, retry budget,
    fact-check timeout, etc.). Sanity guard contra editing accidental."""
    import graph_orchestrator as go
    # Estos valores son los defaults conocidos al momento del fix
    expected_unchanged = {
        "HEDGE_AFTER_BASE_S": 45.0,
        "HARD_CEILING_S": 170.0,
        "FACT_CHECK_TOOL_TIMEOUT_S": 20.0,
        "GLOBAL_PIPELINE_TIMEOUT_S": 720,
        "MIN_RETRY_BUDGET_S": 180,
        "RETRY_SAFETY_MARGIN_S": 80,
    }
    for knob, expected in expected_unchanged.items():
        actual = getattr(go, knob, None)
        assert actual == expected, (
            f"Knob {knob} cambió inesperadamente. Esperado {expected}, "
            f"recibido {actual}. P4-TIMEOUT-1 solo debe afectar "
            f"CRITIQUE_FIX_TIMEOUT_S."
        )
