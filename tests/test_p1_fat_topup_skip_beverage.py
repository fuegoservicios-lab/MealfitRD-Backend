"""[P1-FAT-TOPUP-SKIP-BEVERAGE · 2026-06-28] El day-kcal-floor / fat-topup añadían aceite de oliva a comidas SALADAS bajo
banda. `_is_sweet_meal` NO marca dulce a una BEBIDA salada-neutra ("leche tibia con canela", "infusión de manzanilla") →
el fix le metía aceite a un té/leche (visto en vivo: el revisor flageó "1 cdta de aceite de oliva en una bebida de leche
tibia" como esteatorrea/inusual). Fix: saltar bebidas (markers de bebida) además del sweet-guard.
"""
from __future__ import annotations

import graph_orchestrator as g

_NUT = {"macros": {"protein_g": 90, "carbs_g": 180, "fats_g": 67}}
_BARIA = {"medicalConditions": ["Cirugía Bariátrica"]}


def _meals():
    return [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Revuelto de huevo con espinaca", "ingredients": ["2 huevos"],
         "protein": 18, "carbs": 10, "fats": 8, "cals": 200},
        {"meal": "Almuerzo", "name": "Pollo guisado", "ingredients": ["120g pollo"],
         "protein": 38, "carbs": 40, "fats": 8, "cals": 400},
        {"meal": "Merienda Nocturna", "name": "Leche tibia con canela", "ingredients": ["leche descremada", "canela"],
         "protein": 8, "carbs": 12, "fats": 3, "cals": 110},
        {"meal": "Cena", "name": "Infusión de manzanilla con limón", "ingredients": ["manzanilla", "limon"],
         "protein": 1, "carbs": 3, "fats": 0, "cals": 20},
    ]}]


def test_beverages_skipped_savory_topped(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    d = _meals()
    g._repair_day_kcal_floor_post_caps(d, _NUT, _BARIA)
    by = {m["name"]: m for m in d[0]["meals"]}
    # bebidas NO reciben aceite
    assert by["Leche tibia con canela"].get("_day_kcal_floor") is None
    assert by["Infusión de manzanilla con limón"].get("_day_kcal_floor") is None
    # saladas reales SÍ
    assert by["Revuelto de huevo con espinaca"].get("_day_kcal_floor") is True
    assert by["Pollo guisado"].get("_day_kcal_floor") is True
    # ninguna bebida terminó con aceite en sus ingredientes
    for nm in ("Leche tibia con canela", "Infusión de manzanilla con limón"):
        assert not any("aceite" in str(_i).lower() for _i in by[nm]["ingredients"])


def test_anchor_present():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-FAT-TOPUP-SKIP-BEVERAGE" in src
    assert "_BEVERAGE_MEAL_MARKERS" in src
    # aplicado en ambas pasadas (day-kcal-floor + fat-topup)
    assert src.count("for b in _BEVERAGE_MEAL_MARKERS") >= 2
