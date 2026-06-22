"""[P1-SURGICAL-PROMOTE-BYPASS · 2026-06-22] (audit fresco P1-2) `surgical_marker_regen_node`
promovía el plan re-corregido a `_best_attempt_review_passed=True` + severity='approved' de forma
INCONDICIONAL cuando fixed_count>0. Si la corrección LLM introducía un déficit de proteína (observado
en vivo) o un producto animal en un plan veg*, ese snapshot 'approved' sobrevivía a la re-review (la
democión queda bloqueada por current_rank<prior_rank) y `_swap_to_best_attempt_if_better` restauraba
review_passed=True → los backstops fail-hard (piso de proteína / diet guard), gateados por
`not review_passed`, NO corrían → plan degradado entregado SIN banner.

Fix: antes de promover re-validamos el plan post-surgical con los MISMOS escáneres deterministas del
review (`_protein_floor_shortfall` + `_scan_diet_violations`) vía el helper SSOT testeable
`_surgical_promote_blocked_reason`. Si regresa motivo, NO promovemos.

Parser-based (cableado del nodo) + funcional (helper puro, sin DB ni LLM).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _func_body(src: str, signature_prefix: str) -> str:
    start = src.find(signature_prefix)
    assert start >= 0, f"No se encontró la función: {signature_prefix}"
    nxt = src.find("\ndef ", start + 1)
    return src[start: nxt if nxt > 0 else len(src)]


# ─────────────────────────── A. Cableado (parser-anchors) ───────────────────────────

def test_knob_default_true():
    assert (
        'SURGICAL_PROMOTE_REVALIDATE = _env_bool("MEALFIT_SURGICAL_PROMOTE_REVALIDATE", True)'
        in _GRAPH
    ), "El gate de re-validación debe existir con default True (safety)."


def test_helper_defined():
    assert "def _surgical_promote_blocked_reason(" in _GRAPH
    helper = _func_body(_GRAPH, "def _surgical_promote_blocked_reason(")
    # Re-usa los MISMOS escáneres del review.
    assert "_protein_floor_shortfall(" in helper
    assert "_scan_diet_violations(" in helper
    assert "DIET_HARD_GUARD" in helper


def test_node_revalidates_before_promotion():
    node = _func_body(_GRAPH, "async def surgical_marker_regen_node(")
    # El nodo llama al helper bajo el knob, ANTES de la promoción.
    assert "SURGICAL_PROMOTE_REVALIDATE and fixed_count > 0" in node
    assert "_sr_promote_blocked_reason = _surgical_promote_blocked_reason(new_plan_result, form_data)" in node

    # La promoción a review_passed=True está GATEADA por que no haya motivo de bloqueo.
    i_gate = node.find("if fixed_count > 0 and _sr_promote_blocked_reason is None:")
    i_promote = node.find('state_update["_best_attempt_review_passed"] = True')
    assert i_gate >= 0, "La promoción debe estar gateada por `_sr_promote_blocked_reason is None`."
    assert i_promote > i_gate, "review_passed=True debe estar DENTRO de la rama gateada."


def test_no_unconditional_promotion():
    # Anti-regresión: la promoción NO debe ocurrir bajo un simple `if fixed_count > 0:` sin el gate.
    # Usamos límite de línea (newline + 4 espacios) para NO matchear el `elif fixed_count > 0:`
    # legítimo (rama de bloqueo), que contiene "if fixed_count > 0:" como substring.
    node = _func_body(_GRAPH, "async def surgical_marker_regen_node(")
    assert "\n    if fixed_count > 0:\n" not in node, (
        "La promoción incondicional `if fixed_count > 0:` reabre el bypass P1-2."
    )


def test_tooltip_anchor_present():
    assert _GRAPH.count("P1-SURGICAL-PROMOTE-BYPASS") >= 3


# ─────────────────────────── B. Funcional del helper (sin DB/LLM) ───────────────────────────

@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def _plan_protein_ok():
    # target 120g; cada día suma 120g ≥ piso (0.90×120=108).
    return {
        "macros": {"protein": "120g"},
        "days": [
            {"day": 1, "meals": [
                {"name": "Almuerzo", "protein": "70g", "ingredients": ["150g de pollo", "100g de arroz"]},
                {"name": "Cena", "protein": "50g", "ingredients": ["150g de pescado"]},
            ]},
        ],
    }


def test_safe_plan_returns_none(go):
    assert go._surgical_promote_blocked_reason(_plan_protein_ok(), {"dietType": "balanced"}) is None


def test_protein_deficit_blocks_promotion(go):
    plan = _plan_protein_ok()
    # Bajar la proteína declarada muy por debajo del piso.
    plan["days"][0]["meals"] = [{"name": "Almuerzo", "protein": "20g", "ingredients": ["100g de arroz"]}]
    reason = go._surgical_promote_blocked_reason(plan, {"dietType": "balanced"})
    assert reason and "piso de proteína" in reason


def test_diet_violation_blocks_promotion(go):
    # Plan vegano con pollo (proteína OK para que el check de dieta sea el que dispara).
    plan = {
        "macros": {"protein": "120g"},
        "days": [
            {"day": 1, "meals": [
                {"name": "Almuerzo", "protein": "120g", "ingredients": ["150g de pollo", "100g de arroz"]},
            ]},
        ],
    }
    reason = go._surgical_promote_blocked_reason(plan, {"dietType": "vegano"})
    assert reason and "dieta veg*" in reason


def test_renal_capped_protein_exempt(go):
    # Un plan renal-capeado con proteína baja NO debe bloquearse por el piso (su techo KDIGO manda).
    plan = _plan_protein_ok()
    plan["renal_protein_cap"] = {"applied": True}
    plan["days"][0]["meals"] = [{"name": "Almuerzo", "protein": "20g", "ingredients": ["100g de arroz"]}]
    # Sin violación de dieta → None (exento renal del piso de proteína).
    assert go._surgical_promote_blocked_reason(plan, {"dietType": "balanced"}) is None
