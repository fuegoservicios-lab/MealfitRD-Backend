"""[P1-DM2-GLYCEMIC-PORTION-CAP · 2026-06-27] Cap DURO de porción de víver/almidón de alto índice glucémico
(batata/yuca/papa/plátano maduro/mangú/casabe) a ≤150g por comida para DM2 — el revisor médico rechazaba
'376g batata en una comida' por carga glucémica. Determinista (no depende del LLM). Recupera las kcal
removidas escalando los OTROS ingredientes del plato → kcal del plato ~iguales (band-safe), carbos abajo
(objetivo DM2), índice glucémico abajo.

NB: usa un DB falso para no depender de Neon (el cap llama macros_from_ingredient_string).
"""
from __future__ import annotations

import re

import graph_orchestrator as g


class _FakeDB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g\s+(?:de\s+)?(.*)", str(s), re.I)
        if not m:
            return None
        grams = float(m.group(1)); name = m.group(2).strip().title(); low = name.lower()
        if any(k in low for k in ("batata", "papa", "yuca", "mangu", "casabe")):
            macro = {"carbs": grams * 0.20, "protein": grams * 0.02, "fats": 0.0}
        elif "pollo" in low:
            macro = {"protein": grams * 0.30, "carbs": 0.0, "fats": grams * 0.03}
        elif "aceite" in low:
            macro = {"protein": 0.0, "carbs": 0.0, "fats": grams * 1.0}
        else:
            macro = {"protein": grams * 0.02, "carbs": grams * 0.05, "fats": grams * 0.01}
        macro.update(name=name, grams=grams)
        return macro


def _meal(ings):
    return {"meal": "Desayuno", "name": "x", "ingredients": list(ings), "protein": 0, "carbs": 0, "fats": 0, "cals": 0}


def _meal_kcal(m, db):
    tot = 0.0
    for ing in m["ingredients"]:
        tot += g._ing_kcal_estimate(db.macros_from_ingredient_string(ing) or {})
    return tot


def test_caps_high_gi_viver_and_preserves_calories():
    db = _FakeDB()
    days = [{"meals": [_meal(["376g de Batata asada", "150g de Pollo", "10g de Aceite de oliva"])]}]
    kcal_before = _meal_kcal(days[0]["meals"][0], db)
    n = g.cap_dm2_high_gi_portions(days, {"medicalConditions": ["Diabetes T2"]}, db=db)
    assert n == 1
    ings = days[0]["meals"][0]["ingredients"]
    # batata recortada a 150g
    batata = next(i for i in ings if "batata" in i.lower())
    assert "150" in batata
    # kcal del plato preservadas (±5%) → band-safe
    kcal_after = _meal_kcal(days[0]["meals"][0], db)
    assert abs(kcal_after - kcal_before) / kcal_before < 0.05, (kcal_before, kcal_after)
    # carbos bajaron (objetivo DM2): la batata aportaba el grueso
    assert (db.macros_from_ingredient_string(batata) or {})["carbs"] <= 150 * 0.20 + 0.1


def test_excludes_papaya_and_low_gi():
    db = _FakeDB()
    days = [{"meals": [
        _meal(["300g de Papaya", "120g de Pollo"]),          # papaya (fruta) NO es papa
        _meal(["250g de Platano verde", "100g de Pollo"]),   # plátano VERDE bajo IG → fuera de la lista
        _meal(["200g de Arroz integral", "100g de Pollo"]),  # integral (no es 'blanco') → no se capea
    ]}]
    n = g.cap_dm2_high_gi_portions(days, {"medicalConditions": ["Diabetes T2"]}, db=db)
    assert n == 0
    assert days[0]["meals"][0]["ingredients"][0] == "300g de Papaya"
    assert days[0]["meals"][1]["ingredients"][0] == "250g de Platano verde"


def test_noop_when_not_dm2():
    db = _FakeDB()
    days = [{"meals": [_meal(["376g de Batata asada", "150g de Pollo"])]}]
    assert g.cap_dm2_high_gi_portions(days, {"medicalConditions": ["Hipertensión"]}, db=db) == 0
    assert "376" in days[0]["meals"][0]["ingredients"][0]


def test_under_cap_not_touched():
    db = _FakeDB()
    days = [{"meals": [_meal(["120g de Batata", "150g de Pollo"])]}]
    assert g.cap_dm2_high_gi_portions(days, {"medicalConditions": ["Diabetes T2"]}, db=db) == 0


def test_freetext_dm2_triggers_cap():
    db = _FakeDB()
    days = [{"meals": [_meal(["400g de Yuca hervida", "150g de Pollo", "10g de Aceite"])]}]
    n = g.cap_dm2_high_gi_portions(days, {"otherConditions": "diabetes t2 controlada"}, db=db)
    assert n == 1


def test_knob_default_on():
    assert g.DM2_GLYCEMIC_PORTION_CAP_ENABLED is True
    assert g.DM2_HIGH_GI_CAP_G == 150
