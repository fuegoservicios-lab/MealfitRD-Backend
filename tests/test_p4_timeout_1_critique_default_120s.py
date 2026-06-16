"""[P4-TIMEOUT-1 / P4-TIMEOUT-3 / P6-TIMEOUT-4] Tests para el default de
`MEALFIT_CRITIQUE_FIX_TIMEOUT_S`.

Historia del knob:
  - Pre-P4-TIMEOUT-1: 90s — los 504 con SDK retry interno excedían el cap.
  - P4-TIMEOUT-1 (2026-05-05): bump a 120s. Absorbió ~95% de 504s.
  - P4-TIMEOUT-2 (2026-05-04 03:26): añadió circuit breaker en cascade.
  - P4-TIMEOUT-3 (2026-05-04): bump a 150s. Cubrió el último 5% de casos
    borderline donde 120s no fue suficiente para el SDK retry chain.
  - P6-TIMEOUT-4 (2026-05-05 19:58 [c6eaf808]): bump 150→180s. Días borderline
    seguían timeouteando a 150s + re-timeout en P5-MARKER-REGEN. El CODE
    default vigente es 180.0 (graph_orchestrator.py:428).

[saneamiento drift 2026-06-16] El default subió 150→180 (P6-TIMEOUT-4) y el
knob adyacente `HEDGE_AFTER_BASE_S` subió 45→120 (P3-COST-CUT-V2 · 2026-05-21,
graph_orchestrator.py:258). Ambos bumps documentados + intencionales.

Bug observable (corridas 2026-05-05 múltiples):
  El corrector LLM sufrió 504 DEADLINE_EXCEEDED durante self-critique
  correction. Pattern:
    1. Llamada original al corrector se demora ~60-70s (con tools).
    2. El SDK retry interno por 504 añade +2-5s.
    3. Wall-clock total ~85-95s, antes excedía el cap de 90s.
    4. asyncio.TimeoutError → día sin corregir → marker `_critique_unresolved`
       → P1-SURGICAL-1 fuerza regen en retry → costo adicional.

Cobertura:
  - CODE default es 180s (post P6-TIMEOUT-4)
  - Env var override sigue funcionando
  - El env var conserva tipo float (no int)

NOTA sobre aislamiento del CODE default: el `.env` de dev puede setear
`MEALFIT_CRITIQUE_FIX_TIMEOUT_S` (override de operador, p.ej. 200). El truco
`delenv` + `importlib.reload` NO aísla el CODE default en ese entorno porque
la cadena de import (`db_core.load_dotenv()`) re-puebla `os.environ` al
reloadear. Para asertar el CODE default de forma robusta usamos la ruta de
fallback de `_env_float`: un valor inválido (`"garbage"`) o vacío fuerza el
`default` hardcodeado del callsite, independiente del `.env`.
"""
import importlib
import os

import pytest

# CODE default vigente del knob (graph_orchestrator.py:428, P6-TIMEOUT-4).
_CRITIQUE_CODE_DEFAULT = 180.0


def _reload_module():
    """Recarga graph_orchestrator para releer env vars."""
    import graph_orchestrator
    importlib.reload(graph_orchestrator)
    return graph_orchestrator


# ---------------------------------------------------------------------------
# 1. Default value
# ---------------------------------------------------------------------------
def test_default_is_180s_post_p6_timeout_4_bump(monkeypatch):
    """[P6-TIMEOUT-4] El CODE default es 180s (era 150s tras P4-TIMEOUT-3,
    120s tras P4-TIMEOUT-1, 90s pre-cualquier-fix).

    Forzamos la ruta de fallback de `_env_float` con un valor inválido para
    aislar el CODE default del override del `.env` de dev (ver NOTA del
    docstring del módulo)."""
    monkeypatch.setenv("MEALFIT_CRITIQUE_FIX_TIMEOUT_S", "__invalid__force_default__")
    go = _reload_module()
    assert go.CRITIQUE_FIX_TIMEOUT_S == _CRITIQUE_CODE_DEFAULT, (
        f"CODE default debe ser {_CRITIQUE_CODE_DEFAULT}s tras P6-TIMEOUT-4, "
        f"recibido {go.CRITIQUE_FIX_TIMEOUT_S}"
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
    """Valor inválido en env var → fallback al CODE default (180s) con warning."""
    monkeypatch.setenv("MEALFIT_CRITIQUE_FIX_TIMEOUT_S", "garbage_value")
    go = _reload_module()
    # `_env_float` cae a default cuando no es parseable
    assert go.CRITIQUE_FIX_TIMEOUT_S == _CRITIQUE_CODE_DEFAULT


def test_env_var_empty_falls_to_default(monkeypatch):
    monkeypatch.setenv("MEALFIT_CRITIQUE_FIX_TIMEOUT_S", "")
    go = _reload_module()
    assert go.CRITIQUE_FIX_TIMEOUT_S == _CRITIQUE_CODE_DEFAULT


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
    """El bump del CRITIQUE timeout no debe haber tocado los CODE DEFAULTS de
    otros knobs (hedging, retry budget, fact-check timeout, etc.). Sanity guard
    contra editing accidental.

    [saneamiento drift 2026-06-16] Dos cambios desde que se escribió este test:
      - `HEDGE_AFTER_BASE_S` default 45 → 120 (P3-COST-CUT-V2 · 2026-05-21,
        ahorro de hedges-perdidos; documentado en graph_orchestrator.py:246).
    Además, este test asertaba `getattr(go, knob)` (valor RUNTIME), que es
    frágil: el `.env` de dev puede setear overrides (p.ej.
    `MEALFIT_HEDGE_AFTER_BASE_S=150`). El propósito real del guard es detectar
    edits accidentales a los DEFAULTS del SOURCE, así que ahora parseamos el
    2º argumento de cada `_env_float/_env_int(...)` callsite directamente del
    código — robusto frente a cualquier override de entorno."""
    import re
    import inspect
    import graph_orchestrator as go

    src = inspect.getsource(go)

    # CODE defaults vigentes (2º arg posicional del callsite `_env_*`).
    expected_code_defaults = {
        "HEDGE_AFTER_BASE_S": "120.0",
        "HARD_CEILING_S": "170.0",
        "FACT_CHECK_TOOL_TIMEOUT_S": "20.0",
        "GLOBAL_PIPELINE_TIMEOUT_S": "720",
        "MIN_RETRY_BUDGET_S": "180",
        "RETRY_SAFETY_MARGIN_S": "80",
    }
    for knob, expected in expected_code_defaults.items():
        # Captura: NOMBRE = _env_float|_env_int ( "MEALFIT_..." , <default>
        m = re.search(
            rf"{knob}\s*=\s*_env_(?:float|int)\s*\(\s*\"MEALFIT_[A-Z_]+\"\s*,\s*([\d.]+)",
            src,
        )
        assert m, f"Callsite `{knob} = _env_*(...)` no encontrado en graph_orchestrator."
        actual = m.group(1).rstrip(".")
        assert actual == expected.rstrip("."), (
            f"CODE default de {knob} cambió. Esperado {expected}, código tiene "
            f"{m.group(1)}. El bump del CRITIQUE timeout solo debe afectar "
            f"CRITIQUE_FIX_TIMEOUT_S — si {knob} cambió a propósito, actualiza "
            f"este guard."
        )
