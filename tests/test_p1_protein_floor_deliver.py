"""[P1-PROTEIN-FLOOR-DELIVER · 2026-06-26] (auditoría gap #3) El backstop fail-hard del piso de proteína
NO debe producir CERO plan.

Bug: cuando el plan del LLM quedaba bajo el piso de proteína tras agotar retries, P2-PROTEIN-FLOOR-FAILHARD
lo sustituía por el fallback matemático. Pero ese fallback nace con `aggregated_shopping_list = []` (la lista
se construye en assemble_plan_node, que ya corrió) → el guard SSE/sync lo trata como `_is_fallback` y ERRORea
(422 'critical_restriction' / 503 'IA saturada') → el usuario se queda SIN plan tras una "garantía", con el
toast engañoso "Revisa tus restricciones".

Fix: si el ÚNICO motivo de fallo es el piso de proteína (no schema inválido ni rechazo médico crítico, que
SIEMPRE van al fallback por seguridad), entregar el plan ORIGINAL del LLM (que SÍ tiene lista de compras)
como DEGRADADO (MODE 1: _review_failed_but_delivered + disclaimer honesto), en vez de cero-plan. Gated por
MEALFIT_PROTEIN_FLOOR_DELIVER_DEGRADED (default True).
"""
from __future__ import annotations

import graph_orchestrator as go

_NUTR = {"target_calories": 2000, "macros": {"protein_g": 120, "carbs_g": 200, "fats_g": 60}}


def _breach(plan, *, renal_capped):  # firma de _protein_floor_shortfall
    return [("Día 1", 50.0, 120.0)]


def _no_breach(plan, *, renal_capped):
    return []


def test_protein_floor_only_delivers_original_degraded(monkeypatch):
    """Piso de proteína como ÚNICO motivo → entrega el plan ORIGINAL degradado, NO cero-plan."""
    monkeypatch.setattr(go, "PROTEIN_FLOOR_FAILHARD_GATE", True)
    monkeypatch.setattr(go, "PROTEIN_FLOOR_DELIVER_DEGRADED", True)
    monkeypatch.setattr(go, "_protein_floor_shortfall", _breach)
    plan = {"days": [{"meals": []}], "main_goal": "Salud General"}
    state = {"review_passed": False, "_rejection_severity": "high",
             "rejection_reasons": ["proteína bajo el piso diario"], "plan_result": plan}

    go._apply_critical_review_guardrails(state, nutrition=_NUTR, actual_form_data={}, requested_days=3)
    pr = state["plan_result"]

    assert pr is plan, "debe entregar el plan ORIGINAL (no swappear al fallback de lista vacía = cero-plan)"
    assert not pr.get("_is_fallback"), "el plan entregado NO debe marcarse como fallback"
    assert not pr.get("_critical_rejection"), "NO es rechazo crítico (no hay restricción que revisar)"
    assert pr.get("_review_failed_but_delivered") is True, "debe marcarse degradado-pero-entregado (MODE 1)"
    assert pr.get("_fallback_reason") == "protein_floor"
    disc = (pr.get("_review_disclaimer") or "").lower()
    assert "proteína" in disc and "lista de compras" in disc, f"disclaimer no es el de proteína: {disc!r}"


def test_knob_off_reverts_to_fallback_swap(monkeypatch):
    """Rollback: knob OFF → comportamiento previo (swap al fallback con _is_fallback + _critical_rejection)."""
    monkeypatch.setattr(go, "PROTEIN_FLOOR_FAILHARD_GATE", True)
    monkeypatch.setattr(go, "PROTEIN_FLOOR_DELIVER_DEGRADED", False)
    monkeypatch.setattr(go, "_protein_floor_shortfall", _breach)
    plan = {"days": [{"meals": []}], "main_goal": "Salud General"}
    state = {"review_passed": False, "_rejection_severity": "high",
             "rejection_reasons": ["proteína bajo el piso"], "plan_result": plan}

    go._apply_critical_review_guardrails(state, nutrition=_NUTR, actual_form_data={}, requested_days=2)
    pr = state["plan_result"]

    assert pr.get("_is_fallback") is True, "knob OFF → swap al fallback matemático (comportamiento previo)"
    assert pr.get("_critical_rejection") is True


def test_medical_critical_always_goes_to_fallback(monkeypatch):
    """Un rechazo médico CRÍTICO (alérgeno/condición) jamás se entrega degradado — siempre fallback."""
    monkeypatch.setattr(go, "PROTEIN_FLOOR_DELIVER_DEGRADED", True)
    monkeypatch.setattr(go, "_protein_floor_shortfall", _no_breach)
    plan = {"days": [{"meals": []}], "main_goal": "Salud General"}
    state = {"review_passed": False, "_rejection_severity": "critical",
             "rejection_reasons": ["alérgeno declarado presente"], "plan_result": plan}

    go._apply_critical_review_guardrails(state, nutrition=_NUTR, actual_form_data={}, requested_days=2)
    pr = state["plan_result"]

    assert pr.get("_is_fallback") is True, "rechazo crítico debe ir SIEMPRE al fallback (seguridad)"
    assert pr.get("_critical_rejection") is True


def test_knob_default_is_true():
    """El default de código debe ser True (deliver-degraded), no el cero-plan previo."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    m = re.search(
        r'PROTEIN_FLOOR_DELIVER_DEGRADED\s*=\s*_env_bool\(\s*["\']MEALFIT_PROTEIN_FLOOR_DELIVER_DEGRADED["\']\s*,\s*(True|False)\s*\)',
        src,
    )
    assert m is not None, "no se encontró la declaración del knob PROTEIN_FLOOR_DELIVER_DEGRADED"
    assert m.group(1) == "True", "el default debe ser True (entregar degradado en vez de cero-plan)"
    assert "P1-PROTEIN-FLOOR-DELIVER" in src, "falta el tooltip-anchor P1-PROTEIN-FLOOR-DELIVER"
