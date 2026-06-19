"""[P1-MACRO-AWARE-RECONCILE · 2026-06-15] Reconcile de día consciente del split C:F.

El reconcile legacy (`_protein_preserving_day_reconcile`) es single-factor: escala carbos+grasas
JUNTOS para clavar kcal → distorsiona el split que el solver logró (benchmark: carbos 19% / grasas
22% MAPE vs proteína 6%). Este reconcile escala carbo-dominantes hacia target_carbs y grasa-dominantes
hacia target_fats con factores SEPARADOS, preservando la proteína. Como el target es consistente
(kcal=4P+4C+9F), clavar C y F clava kcal.

Validación DETERMINISTA (sin LLM, sin créditos): con un catálogo stub, verifica que proteína queda
fija, carbos→target, grasas→target, y los ingredientes se escalan por su grupo dominante.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


class _StubDB:
    """macros_from_ingredient_string por keyword. arroz=carbo, aceite=grasa, pollo=proteína."""
    def macros_from_ingredient_string(self, s):
        t = str(s).lower()
        if "arroz" in t:
            return {"kcal": 120.0, "protein": 2.0, "carbs": 28.0, "fats": 0.0}
        if "aceite" in t:
            return {"kcal": 126.0, "protein": 0.0, "carbs": 0.0, "fats": 14.0}
        if "pollo" in t:
            return {"kcal": 150.0, "protein": 30.0, "carbs": 0.0, "fats": 3.0}
        return None  # no resuelto


def _meal():
    return {
        "name": "Almuerzo",
        "ingredients": ["100g de arroz", "20g de aceite", "150g de pollo"],
        "ingredients_raw": ["100g de arroz", "20g de aceite", "150g de pollo"],
        "protein": 30, "carbs": 28, "fats": 20, "cals": 412,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════
# A. Clasificador de grupo dominante
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_ingredient_macro_group(go):
    db = _StubDB()
    assert go._ingredient_macro_group("100g de arroz", db) == "carbs"
    assert go._ingredient_macro_group("20g de aceite", db) == "fats"
    assert go._ingredient_macro_group("150g de pollo", db) == "protein"
    assert go._ingredient_macro_group("1 sazón al gusto", db) is None


# ════════════════════════════════════════════════════════════════════════════════════════════════
# B. El reconcile clava C y F por separado, preservando proteína
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_reconcile_hits_carbs_and_fats_independently(go):
    db = _StubDB()
    m = _meal()
    # curC=28 → target 42 (fC=1.5); curF=20 → target 12 (fF=0.6); proteína fija
    ok = go._macro_aware_day_reconcile([m], target_carbs=42, target_fats=12, db=db)
    assert ok is True
    assert m["protein"] == 30, "la proteína debe quedar FIJA"
    assert m["carbs"] == 42, ("carbos al target", m["carbs"])
    assert m["fats"] == 12, ("grasas al target", m["fats"])
    # kcal recomputada coherente con 4P+4C+9F
    assert m["cals"] == 4 * 30 + 4 * 42 + 9 * 12


def test_reconcile_scales_ingredients_by_their_group(go):
    db = _StubDB()
    m = _meal()
    go._macro_aware_day_reconcile([m], target_carbs=42, target_fats=12, db=db)
    ings = " | ".join(m["ingredients"])
    assert "150g de arroz" in ings, ("arroz carbo-dominante escalado ×1.5", ings)
    assert "12g de aceite" in ings, ("aceite grasa-dominante escalado ×0.6", ings)
    assert "150g de pollo" in ings, ("pollo proteína-dominante INTACTO", ings)


def test_reconcile_improves_both_macros_vs_legacy_distortion(go):
    """C y F parten lejos del target en direcciones OPUESTAS — el single-factor no podría clavar ambos."""
    db = _StubDB()
    m = _meal()  # C=28 (bajo), F=20 (alto)
    tC, tF = 42, 12
    err_before = abs(m["carbs"] - tC) / tC + abs(m["fats"] - tF) / tF
    go._macro_aware_day_reconcile([m], target_carbs=tC, target_fats=tF, db=db)
    err_after = abs(m["carbs"] - tC) / tC + abs(m["fats"] - tF) / tF
    assert err_after < err_before
    assert err_after < 0.05, ("ambos macros esencialmente clavados", err_after)


def test_reconcile_noop_when_already_on_target(go):
    db = _StubDB()
    m = _meal()  # C=28, F=20
    ok = go._macro_aware_day_reconcile([m], target_carbs=28, target_fats=20, db=db)
    assert ok is False, "factores ≈1 → no debe tocar"
    assert m["carbs"] == 28 and m["fats"] == 20 and m["protein"] == 30


def test_reconcile_clamps_extreme_factors(go):
    """Factores acotados a [0.4,1.8] → no produce porciones absurdas aunque el gap sea enorme."""
    db = _StubDB()
    m = _meal()  # C=28
    go._macro_aware_day_reconcile([m], target_carbs=280, target_fats=20, db=db)  # pediría ×10
    assert m["carbs"] == round(28 * 1.8), ("carbo clampeado a ×1.8", m["carbs"])


def test_reconcile_failsafe_on_bad_db(go):
    class _Boom:
        def macros_from_ingredient_string(self, s):
            raise RuntimeError("db caída")
    m = _meal()
    ok = go._macro_aware_day_reconcile([m], target_carbs=42, target_fats=12, db=_Boom())
    assert ok is False  # fail-safe: no revienta, retorna False


# ════════════════════════════════════════════════════════════════════════════════════════════════
# C. Parser anchor + knob default ON [P1-MACRO-RECONCILE-DEFAULT · 2026-06-18, audit fresco P1-C]
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_knob_default_on_and_marker(go):
    # [P1-MACRO-RECONCILE-DEFAULT · 2026-06-18] Default flipeado OFF→ON: el A/B OFF-vs-ON ya se validó
    # (grasas 22%→10.7% MAPE, all-4-en-banda 18.5%→28.1%) y el VPS lo lleva ON vía .env desde 2026-06-15.
    # El default de código en False dejaba el reconcile multi-macro Y el Guard 4c (post-quant, gateado por
    # este AND) MUERTOS en un redeploy con .env limpio o en dev local.
    assert go.MACRO_AWARE_RECONCILE is True, "el knob DEBE estar ON por default (A/B validado, prod ya ON)"
    from pathlib import Path
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert "P1-MACRO-AWARE-RECONCILE" in src
    assert "P1-MACRO-RECONCILE-DEFAULT" in src
    assert "def _macro_aware_day_reconcile(" in src
