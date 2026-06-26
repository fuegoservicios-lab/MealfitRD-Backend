"""[P1-PLANNER-PRO-FALLBACK · 2026-06-26] El planificador (esqueleto) cae a pro si flash falla.

INCIDENTE RAÍZ (user d4bc3af5, corr=0dcc4bf8, 2026-06-26 07:25): una renovación falló con "IA
saturada" pese a que DeepSeek estaba SANO (probe directo HTTP 200). Causa: el circuit breaker
COMPARTIDO de `deepseek-v4-flash` (global a todos los usuarios) se abrió bajo carga concurrente (otro
usuario con dieta vegana generando), y el nodo planificador —que corre SOLO en flash y NO tenía
fallback a pro como el self-critique— murió → EXTREME GRACEFUL DEGRADATION → plan de emergencia
band-0.0.

FIX: `plan_skeleton_node` reintenta UNA vez con `_PRO_MODEL_NAME` (breaker independiente, menos
cargado) antes de degradar. Si el planner ya es pro, o pro también falla, degrada genuino. Knob
MEALFIT_PLANNER_PRO_FALLBACK_ENABLED (default True). Anchor: P1-PLANNER-PRO-FALLBACK.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = open(os.path.join(_ROOT, "graph_orchestrator.py"), encoding="utf-8").read()


def _go_or_skip():
    try:
        import graph_orchestrator as go
        return go
    except Exception as e:  # pragma: no cover - venv sin langchain_openai
        pytest.skip(f"graph_orchestrator no importable en este venv: {type(e).__name__}: {e}")


def test_knob_default_on():
    go = _go_or_skip()
    assert go.PLANNER_PRO_FALLBACK_ENABLED is True


def test_knob_registrado():
    from knobs import _env_bool, get_knobs_registry_snapshot
    _env_bool("MEALFIT_PLANNER_PRO_FALLBACK_ENABLED", True)
    assert "MEALFIT_PLANNER_PRO_FALLBACK_ENABLED" in get_knobs_registry_snapshot()


def test_marker_y_helper_presentes():
    assert "P1-PLANNER-PRO-FALLBACK" in _SRC
    assert "_do_planner_invoke" in _SRC


def _planner_node_body() -> str:
    # Anclar a la definición top-level (columna 0); hay un stub indentado `async def
    # plan_skeleton_node(state): ...` más arriba que NO es el nodo real.
    start = _SRC.index("\nasync def plan_skeleton_node")
    end = _SRC.index("\nasync def ", start + 1)
    return _SRC[start:end]


def test_fallback_estructura_en_el_planner():
    """Parser-based: el cuerpo del planner reintenta con _PRO_MODEL_NAME en el except del invoke,
    guardado por el knob y por planner_model != pro (no reintenta si ya es pro)."""
    body = _planner_node_body()
    # El invoke principal está envuelto en try/except.
    i_invoke = body.index("response = await invoke_planner()")
    i_except = body.index("except Exception as _planner_flash_err", i_invoke)
    # La rama de fallback usa el knob + la guarda de modelo distinto + reintenta con pro.
    i_guard = body.index("PLANNER_PRO_FALLBACK_ENABLED and _PRO_MODEL_NAME and _PRO_MODEL_NAME != planner_model", i_except)
    i_pro_invoke = body.index("_do_planner_invoke(_pro_planner_llm, _pro_planner_cb, _PRO_MODEL_NAME)", i_guard)
    assert i_invoke < i_except < i_guard < i_pro_invoke


def test_degrada_si_pro_tambien_falla():
    """Si el reintento con pro falla, re-lanza el error original (degradación genuina, no loop)."""
    body = _planner_node_body()
    assert "raise _planner_flash_err" in body


def test_invoke_principal_sigue_en_flash_con_breaker_check():
    """El path principal sigue corriendo con el modelo ruteado (flash para free) y su breaker —
    el fallback es ADITIVO, no reemplaza el comportamiento normal."""
    body = _planner_node_body()
    assert "_do_planner_invoke(planner_llm, _planner_cb, planner_model)" in body
    assert "acan_proceed()" in body  # el breaker check sigue dentro del invoke core
