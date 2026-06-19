"""[P1-WARFARIN-VITAMIN-K · 2026-06-18] (audit fresco P1-B) Monitor de consistencia de vitamina K
para usuarios de anticoagulante.

El riesgo de la warfarina NO es el valor absoluto de vit K sino su CONSISTENCIA día a día (cambios
bruscos desestabilizan el INR). El catálogo no tiene columna `vit_k_mcg` (follow-up de DATOS), así que
el monitor usa una heurística por NOMBRE (vegetales de hoja verde por día) y mide la VARIABILIDAD, que
es el riesgo clínico real. Este test ancla la clasificación de variabilidad + el gating por anticoagulante.
"""
from __future__ import annotations

from pathlib import Path

import medication_rules as mr


def _plan(per_day_greens):
    """Construye un plan con `n` ingredientes de hoja verde por día (n = cada entrada de la lista)."""
    days = []
    for n in per_day_greens:
        ings = ["100g de arroz", "150g de pollo"] + ["espinaca salteada"] * n
        days.append({"meals": [{"name": "Almuerzo", "ingredients": ings}]})
    return {"days": days}


def test_consistent_greens_low_variability():
    vk = mr.vitamin_k_consistency(_plan([1, 1, 1, 1]))
    assert vk["applicable"] is True
    assert vk["per_day"] == [1, 1, 1, 1]
    assert vk["spread"] == 0
    assert vk["variability"] == "low"
    assert vk["method"] == "name_presence_heuristic"


def test_swingy_greens_high_variability():
    # Día 1 con mucha hoja verde, resto sin → desestabiliza el INR.
    vk = mr.vitamin_k_consistency(_plan([5, 0, 0, 1]))
    assert vk["per_day"] == [5, 0, 0, 1]
    assert vk["spread"] == 5
    assert vk["variability"] == "high"


def test_moderate_variability_band():
    vk = mr.vitamin_k_consistency(_plan([3, 1, 2, 0]))  # spread = 3 → moderate
    assert vk["spread"] == 3
    assert vk["variability"] == "moderate"


def test_empty_plan_unknown():
    vk = mr.vitamin_k_consistency({"days": []})
    assert vk["applicable"] is True
    assert vk["per_day"] == []
    assert vk["variability"] == "unknown"


def test_detects_diverse_green_names():
    # Distintos nombres es-DO de hoja verde cuentan (brócoli, repollo, lechuga, acelga).
    plan = {"days": [{"meals": [{"name": "C", "ingredients":
            ["brócoli al vapor", "repollo", "lechuga romana", "acelga"]}]}]}
    vk = mr.vitamin_k_consistency(plan)
    assert vk["per_day"] == [4]


def test_note_is_consistency_not_avoidance():
    vk = mr.vitamin_k_consistency(_plan([1, 1]))
    note = vk["note"].lower()
    assert "consistente" in note
    assert "no los elimines" in note  # NO se recomienda eliminar (vit K es saludable)


def test_gating_anchor_in_orchestrator():
    here = Path(__file__).resolve().parent.parent  # backend/
    src = (here / "graph_orchestrator.py").read_text(encoding="utf-8")
    # El monitor solo corre para anticoagulantes, gateado por el knob.
    assert "WARFARIN_VITAMIN_K_GATING and detect_anticoagulant(form_data)" in src
    assert 'WARFARIN_VITAMIN_K_GATING = _env_bool("MEALFIT_WARFARIN_VITAMIN_K_GATING", True)' in src
