"""[P1-SLOT-INCOHERENCE-GATE / P1-STAPLE-REPEAT-GATE · 2026-07-09] Porta al gate DETERMINISTA del
reviewer los detectores que hoy SOLO viven en self_critique (sin equivalente en review_plan_node):
`_detect_slot_incoherence` (almuerzo↔cena comparten carbohidrato / merienda-plato-fuerte /
heavy-protein multi-slot) y `_count_staple_repetitions` (staple cross-día ≥2). Prerequisito para poder
apagar self_critique (Fase 2) sin perder cobertura — para usuarios sanos/guest el reviewer LLM se
bypasea, así que estos detectores serían el ÚNICO floor.

Nacen OFF (convención repo: gates nuevos default OFF) → el commit es INERTE hasta canario. Los issues
de slot-incoherence son day-attributable (`Día N:`) → ruta quirúrgica; staple es cross-día → retry
completo (conservador).

Parser-based + un funcional de inertness (default OFF no cambia `_surgical_reject_targets`).
"""
import os

import graph_orchestrator as go

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def _src():
    return _read(_BACKEND, "graph_orchestrator.py")


# ---------- knobs nacen OFF ----------

def test_knobs_defined_and_default_off():
    assert go.SLOT_INCOHERENCE_GATE_ENABLED is False, "SLOT_INCOHERENCE_GATE debe nacer OFF"
    assert go.STAPLE_REPEAT_GATE_ENABLED is False, "STAPLE_REPEAT_GATE debe nacer OFF"
    src = _src()
    assert 'MEALFIT_SLOT_INCOHERENCE_GATE' in src
    assert 'MEALFIT_STAPLE_REPEAT_GATE' in src


def test_markers_present():
    src = _src()
    assert 'P1-SLOT-INCOHERENCE-GATE' in src
    assert 'P1-STAPLE-REPEAT-GATE' in src


# ---------- las gates viven en review_plan_node y consumen los detectores ----------

def test_review_gate_consumes_detectors():
    src = _src()
    review = src[src.find("async def review_plan_node"):]
    review = review[: review.find("\nasync def ", 1) if review.find("\nasync def ", 1) > 0 else 120000]
    # bajo el knob de slot-incoherence, el reviewer debe invocar el detector y rechazar
    assert "SLOT_INCOHERENCE_GATE_ENABLED" in review, "el gate de slot-incoherence debe estar en review_plan_node"
    assert "_detect_slot_incoherence(" in review, "el reviewer debe invocar _detect_slot_incoherence"
    assert "STAPLE_REPEAT_GATE_ENABLED" in review, "el gate de staple-repeat debe estar en review_plan_node"
    assert "_count_staple_repetitions(" in review, "el reviewer debe invocar _count_staple_repetitions"


# ---------- ruta quirúrgica: slot-incoherence es day-attributable ----------

def test_slot_incoherence_reasons_in_surgical_whitelist():
    src = _src()
    wl = src[src.find("_SURGICAL_REJECT_SAFE_PREFIXES"):]
    wl = wl[: wl.find(")\n")]
    # los prefijos de slot-incoherence (day-attributable) deben permitir ruta quirúrgica
    assert "comparten carbohidrato" in wl, "shared-carb debe estar en la whitelist quirúrgica"
    assert "merienda parece un plato fuerte" in wl, "merienda-heavy debe estar en la whitelist quirúrgica"


def test_staple_repeat_NOT_in_surgical_whitelist():
    """staple es CROSS-día (no atribuible a un día) → debe ir a retry completo, NO quirúrgico.
    Chequea las ENTRADAS reales del tuple (no el comentario que menciona 'staple' para explicarlo)."""
    assert not any("staple" in p.lower() for p in go._SURGICAL_REJECT_SAFE_PREFIXES), (
        "ningún prefijo quirúrgico debe referirse a staple (es cross-día → retry completo)"
    )


# ---------- inertness: con las gates OFF (default), _surgical_reject_targets no cambia ----------

def test_surgical_targets_inert_when_gate_off():
    """Con SLOT_INCOHERENCE_GATE OFF (default), un rechazo NO-whitelisted sigue devolviendo None
    (el detector nuevo no debe alterar la atribución cuando el gate está apagado)."""
    state = {
        "rejection_reasons": ["razón no atribuible cualquiera (skeleton fidelity)"],
        "plan_result": {"days": [{"day": 1, "meals": []}, {"day": 2, "meals": []}]},
    }
    assert go._surgical_reject_targets(state) is None
