"""[P1-GAINMUSCLE-FLOOR-FINAL-REFILL · 2026-07-08] Re-relleno final del floor de bulk.

Forense del plan vivo c2aef769 (renovación gain_muscle del owner): Día 3 quedó en 1648/2100 kcal (78%)
y el banner "por debajo de banda" (band_score 0.556). El `_repair_gainmuscle_day_kcal` corre en el
macro-engine ANTES de cheese-dump-final + postquantize-recheck + micro-postengine re-trim +
sodium-postmotor — pasadas que BAJAN kcal de un día DESPUÉS de que el floor lo subió, sin re-relleno.
Un día de bulk podía quedar bajo banda (y su comida principal con carbos vestigiales — "15g de arroz").

Fix: re-correr el floor como ÚLTIMA pasada aditiva (`final_pass=True`: ignora el marker per-meal +
consolida la línea de arroz existente en vez de duplicar), tras todos los recortes tardíos.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ─────────────────────────── Parser anchors ───────────────────────────

def test_marker_and_knob_present():
    assert "P1-GAINMUSCLE-FLOOR-FINAL-REFILL" in _GO
    assert 'GAINMUSCLE_FLOOR_FINAL_REFILL = _env_bool("MEALFIT_GAINMUSCLE_FLOOR_FINAL_REFILL", True)' in _GO


def test_final_call_runs_after_late_trims():
    """La pasada final debe correr DESPUÉS de cheese-dump-final y sodium-postmotor (última aditiva)."""
    assert "_repair_gainmuscle_day_kcal(days, nutrition, form_data, final_pass=True)" in _GO
    i_final = _GO.rindex("GAINMUSCLE-FLOOR-FINAL-REFILL")
    assert _GO.index("P1-CHEESE-DUMP-FINAL") < i_final, "el re-relleno debe correr tras cheese-dump-final"
    assert _GO.index("P1-SODIUM-POSTMOTOR") < i_final, "el re-relleno debe correr tras sodium-postmotor"


# ─────────────────────────── Funcional ───────────────────────────

def _nut():
    return {"macros": {"protein_g": 123.0, "carbs_g": 271.0, "fats_g": 58.0}}  # target_kcal≈2098, floor≈1993


def _bulk_fd():
    return {"mainGoal": "gain_muscle"}


def _rice_lines(meal):
    return [i for i in meal.get("ingredients", []) if "arroz blanco cocido" in str(i).lower()]


def _rice_grams(meal):
    ls = _rice_lines(meal)
    if not ls:
        return 0
    m = re.match(r"\s*(\d+)", ls[0])
    return int(m.group(1)) if m else 0


def _low_bulk_day(marked=True):
    # 1650 kcal < floor 1993 — como el Día 3 vivo; la principal YA marcada por el floor temprano.
    return {"day": 3, "meals": [
        {"meal": "Almuerzo", "name": "Pescado con Arroz", "cals": 400, "protein": 25, "carbs": 60, "fats": 8,
         "ingredients": ["100 g de pescado", "20g de arroz blanco cocido"],
         "ingredients_raw": ["100 g de pescado", "20g de arroz blanco cocido"],
         **({"_gainmuscle_kcal_floor": True} if marked else {})},
        {"meal": "Cena", "name": "Platano con Queso", "cals": 450, "protein": 17, "carbs": 74, "fats": 9,
         "ingredients": ["150 g de platano verde", "60 g de queso blanco"],
         "ingredients_raw": ["150 g de platano verde", "60 g de queso blanco"]},
        {"meal": "Desayuno", "name": "Tostadas con Huevo", "cals": 300, "protein": 13, "carbs": 39, "fats": 11,
         "ingredients": ["2 huevos", "2 tostadas integrales"],
         "ingredients_raw": ["2 huevos", "2 tostadas integrales"]},
        {"meal": "Merienda", "name": "Uvas dulces con Cottage", "cals": 500, "protein": 26, "carbs": 58, "fats": 21,
         "ingredients": ["1 taza de uvas", "120 g de queso cottage"],
         "ingredients_raw": ["1 taza de uvas", "120 g de queso cottage"]},
    ]}


def test_final_refill_ignores_marker_and_consolidates():
    import graph_orchestrator as g
    day = _low_bulk_day(marked=True)
    before = _rice_grams(day["meals"][0])
    added = g._repair_gainmuscle_day_kcal([day], _nut(), _bulk_fd(), final_pass=True)
    assert added > 0, "debe re-rellenar el día de bulk bajo banda PESE al marker per-meal"
    # consolidó en la línea de arroz existente (NO duplicó)
    assert len(_rice_lines(day["meals"][0])) == 1, f"debe consolidar, no duplicar: {_rice_lines(day['meals'][0])}"
    assert _rice_grams(day["meals"][0]) > before, "la línea de arroz existente debe crecer"


def test_early_pass_respects_marker():
    """Sin final_pass, el marker per-meal sigue bloqueando (comportamiento previo intacto)."""
    import graph_orchestrator as g
    day = _low_bulk_day(marked=True)
    # solo la principal (Almuerzo) está marcada; el floor temprano puede usar OTRAS principales,
    # pero NO la marcada → su arroz NO crece.
    before = _rice_grams(day["meals"][0])
    g._repair_gainmuscle_day_kcal([day], _nut(), _bulk_fd(), final_pass=False)
    assert _rice_grams(day["meals"][0]) == before, "sin final_pass, la comida marcada no se re-rellena"


def test_no_refill_when_day_in_band():
    import graph_orchestrator as g
    day = {"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "X", "cals": 2050, "protein": 123, "carbs": 271, "fats": 58,
         "ingredients": ["comida"], "ingredients_raw": ["comida"]}]}
    assert g._repair_gainmuscle_day_kcal([day], _nut(), _bulk_fd(), final_pass=True) == 0


def test_no_refill_non_bulk_goal():
    import graph_orchestrator as g
    day = _low_bulk_day(marked=False)
    assert g._repair_gainmuscle_day_kcal([day], _nut(), {"mainGoal": "lose_weight"}, final_pass=True) == 0


def test_refill_respects_carb_ceiling():
    """El re-relleno no debe empujar los carbos del día sobre 1.10× del target (otra celda de banda)."""
    import graph_orchestrator as g
    day = _low_bulk_day(marked=False)
    g._repair_gainmuscle_day_kcal([day], _nut(), _bulk_fd(), final_pass=True)
    day_carbs = sum(g._meal_macro_num(m.get("carbs")) for m in day["meals"])
    assert day_carbs <= 1.10 * 271 + 1, f"carbos del día {day_carbs} sobre el techo 1.10×271"
