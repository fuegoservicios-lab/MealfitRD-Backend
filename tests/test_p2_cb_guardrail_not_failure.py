"""[P2-CB-GUARDRAIL-NOT-FAILURE · 2026-06-24] Un rechazo de guardrail/validador (ValueError) en
swap_meal NO debe contar como circuit-breaker failure.

Incidente que lo motivó (corr=ca92cac0, plan 4ba66a06, 2026-06-24): un regenerate-day donde el LLM
escribía 'dorado' en la receta sin listarlo → el guard de coherencia receta↔lista rechazó los 3
intentos → swap_meal levantó SWAP_LLM_RETRIES_EXHAUSTED → el except contaba `record_failure()` → el
breaker per-modelo (COMPARTIDO entre usuarios) se abrió → merienda/cena del día ni se intentaron, y el
usuario vio "IA no disponible" aunque DeepSeek estaba sano. El breaker debe tripear por caídas del
PROVEEDOR (timeout/5xx/conexión), no porque NUESTRO validador rechazó un output bien-formado.

Fix: en el except de swap_meal, si `isinstance(e, ValueError)` (todos los validadores levantan
ValueError) y NO está el knob de override → log + NO record_failure. Genuinos errores de proveedor
(no-ValueError) siguen contando. Knob MEALFIT_SWAP_CB_COUNT_GUARDRAIL=true revierte.

Parser-based (la lógica del except no se puede aislar sin importar agent.py completo / langgraph).
"""
import ast
import os

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def _func_src(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"función {name!r} no encontrada")


AGENT = _read("agent.py")
APP = _read("app.py")
SWAP = _func_src(AGENT, "swap_meal")


def test_marker_and_knob_present():
    assert "P2-CB-GUARDRAIL-NOT-FAILURE" in SWAP, "tooltip-anchor para que un renombre falle el test"
    assert "MEALFIT_SWAP_CB_COUNT_GUARDRAIL" in SWAP, "knob de rollback"


def test_guardrail_valueerror_skips_record_failure():
    # el guard isinstance(e, ValueError) debe preceder a record_failure, y record_failure
    # debe quedar en la rama else (no incondicional).
    assert "isinstance(e, ValueError)" in SWAP
    guard_idx = SWAP.index("isinstance(e, ValueError)")
    rf_idx = SWAP.index("_swap_cb.record_failure()")
    assert guard_idx < rf_idx, "el guard de guardrail debe preceder a record_failure"
    # record_failure ya NO es la sentencia incondicional tras el rate-limit raise:
    # debe estar dentro de un bloque else.
    assert "else:\n            _swap_cb.record_failure()" in SWAP, (
        "record_failure debe estar gateado en un else (solo errores reales del proveedor)"
    )


def test_rate_limit_check_still_precedes():
    # el descarte de rate-limit (P1-CHAT-LLM-429) sigue ANTES del nuevo guard de guardrail.
    assert "_is_rate_limit_error(e)" in SWAP
    assert SWAP.index("_is_rate_limit_error(e)") < SWAP.index("isinstance(e, ValueError)")


def test_last_known_pfix_bumped():
    assert "P2-CB-GUARDRAIL-NOT-FAILURE" in APP, "_LAST_KNOWN_PFIX debe reflejar este cierre"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
