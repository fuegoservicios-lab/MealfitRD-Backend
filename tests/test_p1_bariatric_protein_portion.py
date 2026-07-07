"""[P1-BARIATRIC-PROTEIN-PORTION · 2026-06-27] Refinamiento de FASE A (P1-PROTEIN-FLOOR-POST-CAPS) tras re-test en
vivo (corr=ee272f5b): el re-cierre post-cap funcionaba pero (a) llenaba cada comida a SU target → sobre-disparaba
el día (91g/114% de banda) y (b) sin cap por-comida metía 213g de camarones → el revisor rechazaba por volumen.

Fix: el re-cierre apunta al PISO DEL DÍA (para cuando lo alcanza, no sobre-dispara) y en bariátrica cada inyección
por comida se limita a ≤BARIATRIC_PROTEIN_PORTION_CAP_G (90g) de alimento (~22g proteína). 6 comidas × ~22g = 132g
posibles ≥ piso 80g → compatible con cumplir el piso sin porciones absurdas.
"""
from __future__ import annotations

from pathlib import Path

import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent
_NUT = {"macros": {"protein_g": 80, "carbs_g": 180, "fats_g": 53}}


def _wire(monkeypatch, cap=None):
    import graph_orchestrator as g

    def fake_closer(m, target, db, cands, allergies=None, fill_pct=0.92, max_add_g=300, slot_cal_target=0.0, enforce_min_threshold=True):
        if cap is not None:
            cap["max_add_g"] = max_add_g
        cur = g._meal_macro_num(m.get("protein"))
        if cur >= target * 0.9:
            return 0
        m["protein"] = round(target * 0.92)
        return round(target * 0.92 - cur)

    monkeypatch.setattr(g, "_close_protein_gap_for_meal", fake_closer)
    monkeypatch.setattr(g, "_macro_aware_day_reconcile", lambda ms, c, f, db: True)
    monkeypatch.setattr(g, "_safe_high_density_proteins",
                        lambda al, db, min_protein=18.0: [(0.25, "Pechuga de pollo", None), (0.22, "Camarones", None)])
    return g


def test_day_floor_no_overshoot(monkeypatch):
    """Re-cierre para al alcanzar el PISO del día (~90% del target), no sube a 114%."""
    g = _wire(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "protein": 10, "cals": 300},
                                 {"meal": "Almuerzo", "protein": 10, "cals": 400},
                                 {"meal": "Cena", "protein": 10, "cals": 350}]}]
    g._repair_protein_floor_post_caps(days, _NUT, {"medicalConditions": ["Cirugía Bariátrica"]}, db=object())
    tot = sum(m["protein"] for m in days[0]["meals"])
    assert 72 <= tot <= 88, f"el día debe quedar cerca del piso (72-88g), no sobre-disparado: {tot}"


def test_day_already_at_floor_skipped(monkeypatch):
    g = _wire(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "protein": 30, "cals": 400},
                                 {"meal": "Cena", "protein": 30, "cals": 400},
                                 {"meal": "Desayuno", "protein": 30, "cals": 300}]}]
    assert g._repair_protein_floor_post_caps(days, _NUT, {"medicalConditions": ["Cirugía Bariátrica"]}, db=object()) == 0


def test_bariatric_per_meal_portion_cap_passed(monkeypatch):
    cap = {}
    g = _wire(monkeypatch, cap=cap)
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "protein": 5, "cals": 400}]}]
    g._repair_protein_floor_post_caps(days, _NUT, {"medicalConditions": ["Cirugía Bariátrica"]}, db=object())
    assert cap.get("max_add_g") == g.BARIATRIC_PROTEIN_PORTION_CAP_G, \
        f"bariátrica debe pasar el cap de porción ({g.BARIATRIC_PROTEIN_PORTION_CAP_G}g), no 300: {cap}"


def test_non_bariatric_uses_wide_add(monkeypatch):
    cap = {}
    g = _wire(monkeypatch, cap=cap)
    days = [{"day": 1, "meals": [{"meal": "Almuerzo", "protein": 5, "cals": 400}]}]
    g._repair_protein_floor_post_caps(days, _NUT, {"medicalConditions": ["Ninguna"]}, db=object())
    assert cap.get("max_add_g") == 300  # no-bariátrica conserva el max_add_g amplio del closer


def test_anchor():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-BARIATRIC-PROTEIN-PORTION" in go and "BARIATRIC_PROTEIN_PORTION_CAP_G" in go
    assert "_day_floor" in go  # el re-cierre apunta al piso del día
