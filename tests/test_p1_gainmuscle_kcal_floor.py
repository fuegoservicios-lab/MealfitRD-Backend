"""[P1-GAINMUSCLE-KCAL-FLOOR · 2026-07-06] Review de logs (plan del owner 4339544f, ganancia
muscular, DEGRADADO por kcal): el cerrador de piso de calorías existía SOLO para bariátrico. Los
perfiles de ganancia muscular (superávit) también se quedan bajo banda de kcal — el solver
clampea (no escala proteína animal infinitamente) y no había cerrador → degradaban. Este añade
arroz blanco cocido (carbo denso, macro-apropiado para músculo — el aceite sesgaría las grasas)
a comidas principales no-dulces, respetando los techos de kcal y carbs.
"""
import pytest

import graph_orchestrator as go

# target: 4×123 + 4×310 + 9×65 = 2317 kcal; floor 0.95× = ~2201.
NUT = {"macros": {"protein_g": 123, "carbs_g": 310, "fats_g": 65}}
FD = {"mainGoal": "gain_muscle"}


def _meal(slot, name, cals, carbs, prot, fats):
    return {"meal": slot, "name": name, "cals": cals, "carbs": carbs, "protein": prot,
            "fats": fats, "ingredients": [f"{prot}g de proteina"],
            "ingredients_raw": [f"{prot}g de proteina"]}


def _under_day():
    # total 2000 kcal (bajo el piso 2201), carbs 205 (bajo target 310 → hay headroom).
    return [{"day": 1, "meals": [
        _meal("Almuerzo", "Bistec de Res con Mapuey", 520, 50, 44, 15),
        _meal("Cena", "Pollo con Vegetales", 490, 55, 30, 12),
        _meal("Desayuno", "Revoltillo con Plátano", 499, 60, 26, 15),
        _meal("Merienda", "Yogurt con Fresa", 491, 40, 16, 18),
    ]}]


def test_adds_rice_to_reach_floor():
    days = _under_day()
    n = go._repair_gainmuscle_day_kcal(days, NUT, FD)
    assert n > 0, "día bajo banda de kcal → añade calorías"
    total = sum(float(m["cals"]) for m in days[0]["meals"][0].get("_x", []) or days[0]["meals"])
    total = sum(float(m["cals"]) for m in days[0]["meals"])
    assert total >= 2201 * 0.99, f"kcal subida hacia el piso (~2201): {total}"
    # arroz añadido a comida PRINCIPAL (no a la merienda dulce)
    merienda = days[0]["meals"][3]
    assert not any("arroz" in str(i).lower() for i in merienda["ingredients"]), (
        "no mete arroz en una merienda dulce de yogurt"
    )
    assert any(any("arroz blanco cocido" in str(i).lower() for i in m["ingredients"])
               for m in days[0]["meals"][:2]), "arroz en almuerzo/cena"


def test_respects_carb_ceiling():
    # día con carbs ya altos (cerca del techo 310×1.10=341) → NO sobre-empujar carbos.
    days = [{"day": 1, "meals": [
        _meal("Almuerzo", "Arroz con Pollo", 600, 160, 40, 15),
        _meal("Cena", "Pasta con Res", 590, 165, 35, 15),
        _meal("Desayuno", "Avena", 400, 10, 15, 10),  # total kcal 1590 (bajo), carbs 335 (casi techo)
    ]}]
    go._repair_gainmuscle_day_kcal(days, NUT, FD)
    day_carbs = sum(float(m["carbs"]) for m in days[0]["meals"])
    assert day_carbs <= 341 + 1e-6, f"carbos NO superan el techo de banda (1.10×310=341): {day_carbs}"


def test_day_in_band_untouched():
    days = [{"day": 1, "meals": [
        _meal("Almuerzo", "Bistec con Arroz", 620, 70, 45, 18),
        _meal("Cena", "Pollo con Arroz", 600, 75, 40, 16),
        _meal("Desayuno", "Avena", 600, 80, 28, 16),
        _meal("Merienda", "Yogurt", 500, 45, 20, 14),  # total 2320 ≥ floor
    ]}]
    assert go._repair_gainmuscle_day_kcal(days, NUT, FD) == 0, "día en banda → intacto"


def test_non_gainmuscle_skipped():
    assert go._repair_gainmuscle_day_kcal(_under_day(), NUT, {"mainGoal": "lose_weight"}) == 0


def test_marker_anchored():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-GAINMUSCLE-KCAL-FLOOR" in src
