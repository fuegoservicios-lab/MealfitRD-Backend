"""[P4-UNIFIED-RESOLVER · 2026-06-14] Las columnas nuevas (magnesium/saturated_fat/phosphorus/
cholesterol) pobladas desde USDA vuelven ENFORCED con dato real (no solo prompt) dos condiciones del
set Pareto: HTA (patrón DASH: pisos de potasio 4700 + magnesio 500) y dislipidemia (techo de grasa
saturada <7% de las kcal). Verifica que `build_micronutrient_report` añade esos condition_targets +
evalúa el panel con los nuevos micros.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import micronutrients as mn


class _StubDB:
    """Devuelve micros fijos por ingrediente (independiente del nombre) → control total del test."""
    def __init__(self, **micros):
        self._m = micros

    def micros_from_ingredient_string(self, s):
        return dict(self._m)


def _plan(n_ings=3):
    meal = {"ingredients": [f"ing{i}" for i in range(n_ings)]}
    return {"days": [{"meals": [meal]}]}


# ── DRI general: el magnesio ahora está SIEMPRE en el panel ──
def test_magnesium_always_in_panel():
    db = _StubDB(grams=100, magnesium_mg=50)
    rep = mn.build_micronutrient_report(_plan(), db, sex="M", conditions=["Ninguna"])
    keys = {p["key"] for p in rep["panel"]}
    assert "magnesium_mg" in keys
    # satfat NO está en el panel sin dislipidemia
    assert "saturated_fat_g" not in keys


# ── HTA → DASH: pisos elevados de potasio + magnesio ──
def test_hta_adds_dash_potassium_magnesium_targets():
    db = _StubDB(grams=100, potassium_mg=200, magnesium_mg=30, sodium_mg=100)
    rep = mn.build_micronutrient_report(_plan(), db, sex="M", conditions=["Hipertensión"])
    cts = {c["condicion"]: c for c in rep["condition_targets"]}
    assert any("DASH" in k for k in cts)
    panel = {p["key"]: p for p in rep["panel"]}
    assert panel["potassium_mg"]["piso"] == 4700.0    # DASH eleva sobre el DRI (3400)
    assert panel["magnesium_mg"]["piso"] == 500.0      # DASH eleva sobre el DRI (420)


def test_no_hta_keeps_dri_potassium_floor():
    db = _StubDB(grams=100, potassium_mg=200, magnesium_mg=30)
    rep = mn.build_micronutrient_report(_plan(), db, sex="M", conditions=["Ninguna"])
    panel = {p["key"]: p for p in rep["panel"]}
    assert panel["potassium_mg"]["piso"] == 3400.0     # DRI male, sin DASH
    assert panel["magnesium_mg"]["piso"] == 420.0      # DRI male


# ── Dislipidemia → techo de grasa saturada <7% kcal ──
def test_dyslipidemia_adds_satfat_ceiling():
    db = _StubDB(grams=100, saturated_fat_g=10)
    rep = mn.build_micronutrient_report(
        _plan(), db, sex="F", conditions=["Colesterol alto"], daily_kcal=2000)
    panel = {p["key"]: p for p in rep["panel"]}
    assert "saturated_fat_g" in panel
    # 7% de 2000 kcal / 9 = ~15.6 g
    assert panel["saturated_fat_g"]["techo"] == round(0.07 * 2000 / 9.0, 1)
    cts = [c["condicion"] for c in rep["condition_targets"]]
    assert any("islipid" in c for c in cts)


def test_dyslipidemia_satfat_over_ceiling_flags_alto():
    # 3 ingredientes × 30g satfat = 90g/día >> techo ~15.6g → status alto
    db = _StubDB(grams=100, saturated_fat_g=30)
    rep = mn.build_micronutrient_report(
        _plan(3), db, sex="M", conditions=["dislipidemia"], daily_kcal=2000)
    panel = {p["key"]: p for p in rep["panel"]}
    assert panel["saturated_fat_g"]["status"] == "alto"
    assert any(g["key"] == "saturated_fat_g" for g in rep["gaps"])


def test_no_dyslipidemia_no_satfat_target():
    db = _StubDB(grams=100, saturated_fat_g=30)
    rep = mn.build_micronutrient_report(_plan(), db, sex="M", conditions=["Ninguna"], daily_kcal=2000)
    assert "saturated_fat_g" not in {p["key"] for p in rep["panel"]}


# ── nutrition_db expone las columnas nuevas ──
def test_nutrition_db_exposes_new_columns():
    import nutrition_db as ndb
    row = {"name": "Tofu", "kcal_per_100g": 144, "protein_g_per_100g": 17,
           "carbs_g_per_100g": 3, "fats_g_per_100g": 9,
           "magnesium_mg_per_100g": 58, "phosphorus_mg_per_100g": 190,
           "saturated_fat_g_per_100g": 1.26, "cholesterol_mg_per_100g": 0}
    db = ndb.IngredientNutritionDB(rows=[row])
    info = db.lookup("tofu")
    assert info.magnesium_mg == 58 and info.phosphorus_mg == 190
    assert info.saturated_fat_g == 1.26 and info.cholesterol_mg == 0
    micros = db.micros_from_ingredient_string("150g de tofu")
    assert micros["magnesium_mg"] == round(58 * 1.5, 3)
    assert micros["saturated_fat_g"] == round(1.26 * 1.5, 3)
