"""[P2-PANTRY-SUFFICIENCY · 2026-06-23] Contrato del gate de suficiencia en el swap.

El requisito del owner: si la Nevera no cubre las macros del objetivo para el plato,
NO generar nada y avisar — SIN descontar regeneración. Eso exige que el gate corra
ANTES de `log_api_usage` (cuota) y de `swap_meal` (LLM), y que retorne el soft-fail
`pantry_insufficient_for_goal`. Test PARSER-BASED sobre el cuerpo de `api_swap_meal`
(el handler tiene Depends/side-effects difíciles de invocar en unit); ancla el ORDEN
para que un refactor que mueva el gate después del cobro falle aquí antes que en prod.
"""
import os
import re
from pathlib import Path

import pytest

_PLANS = (Path(__file__).resolve().parent.parent / "routers" / "plans.py").read_text(encoding="utf-8")


def _extract(fn: str) -> str:
    m = re.search(rf"^def {re.escape(fn)}\(", _PLANS, re.MULTILINE)
    assert m, f"{fn} no encontrada"
    start = m.start()
    nxt = re.search(r"^(def |@router\.)", _PLANS[start + 1:], re.MULTILINE)
    return _PLANS[start: start + 1 + nxt.start()] if nxt else _PLANS[start:]


@pytest.fixture(scope="module")
def swap_body() -> str:
    return _extract("api_swap_meal")


def test_gate_helper_reads_master_knob():
    assert re.search(r"def _pantry_sufficiency_gate_on\(", _PLANS), "falta el helper del knob"
    assert "MEALFIT_PANTRY_SUFFICIENCY_GATE" in _PLANS, "el gate debe leer MEALFIT_PANTRY_SUFFICIENCY_GATE"


def test_gate_invokes_evaluator_knob_gated(swap_body: str):
    assert "evaluate_pantry_sufficiency" in swap_body, "el swap no invoca el evaluador de suficiencia"
    assert "_pantry_sufficiency_gate_on()" in swap_body, "el gate no está detrás del knob master"


def test_gate_runs_before_quota_and_llm(swap_body: str):
    """El gate (return soft-fail) DEBE preceder a `log_api_usage` y `swap_meal(`."""
    eval_idx = swap_body.find("evaluate_pantry_sufficiency")
    block_return_idx = swap_body.find("pantry_insufficient_for_goal")
    quota_idx = swap_body.find('log_api_usage(user_id, "llm_swap_meal")')
    swap_idx = swap_body.find("swap_meal(data)")
    assert eval_idx > 0 and block_return_idx > 0, "gate ausente"
    assert quota_idx > 0 and swap_idx > 0, "anclas log_api_usage/swap_meal ausentes"
    assert eval_idx < quota_idx, "el gate corre DESPUÉS de log_api_usage → descontaría cuota indebidamente"
    assert eval_idx < swap_idx, "el gate corre DESPUÉS de swap_meal → llamaría al LLM indebidamente"
    assert block_return_idx < quota_idx, "el return de insuficiencia NO precede al cobro de cuota"


def test_softfail_shape_is_http200_body_flag(swap_body: str):
    """Soft-fail 200 con swap_failed+error_code (NO 4xx) — mismo patrón que los demás swap fails."""
    seg = swap_body[swap_body.find("pantry_insufficient_for_goal") - 200: swap_body.find("pantry_insufficient_for_goal") + 200]
    assert '"swap_failed": True' in seg, "el soft-fail debe llevar swap_failed=True"
    assert "return" in seg, "debe ser un return (no un raise HTTPException)"
