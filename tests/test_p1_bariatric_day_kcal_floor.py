"""[P1-BARIATRIC-DAY-KCAL-FLOOR · 2026-06-28] El plan bariátrico nace ~1500 kcal/día (el volume cap descarta kcal sin
recuperación) y FASE A solo sube kcal en días donde añade PROTEÍNA → un día con proteína OK pero kcal < banda (0.95×target)
se quedaba bajo (band kcal 2/3). NO es límite físico: cerrar ~400 kcal se hace con grasa MUFA (aceite de oliva, ~0 volumen,
dumping-safe, sin conflicto DM2) en comidas SALADAS. Esta pasada lo cierra; RENAL skip, idempotente, graceful, no toca días
ya en banda.

Incluye el fix P1-SWEET-MARKER-WORDBOUNDARY: "pina" (piña) matcheaba dentro de "es-PINA-ca" (espinaca) → platos salados con
espinaca se marcaban dulces (rompía el sweet-guard del closer Y este day-kcal-floor). Word-boundary lo corrige.
"""
from __future__ import annotations

import graph_orchestrator as g
from constants import strip_accents as _sa


_NUT = {"macros": {"protein_g": 90, "carbs_g": 180, "fats_g": 67}}  # target_kcal=1683, floor 0.96≈1616
_BARIA = {"medicalConditions": ["Cirugía Bariátrica"]}


def _day(cals_each=None):
    cals = cals_each or [300, 450, 250, 350]
    return {"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Revuelto de huevo con espinaca", "ingredients": ["2 huevos", "espinaca"],
         "protein": 18, "carbs": 10, "fats": 8, "cals": cals[0]},
        {"meal": "Almuerzo", "name": "Pollo guisado con arroz", "ingredients": ["120g pollo", "arroz"],
         "protein": 38, "carbs": 45, "fats": 8, "cals": cals[1]},
        {"meal": "Merienda", "name": "Batido de yogurt con fresas", "ingredients": ["yogurt griego", "fresas"],
         "protein": 15, "carbs": 20, "fats": 3, "cals": cals[2]},
        {"meal": "Cena", "name": "Pescado al horno con vegetales", "ingredients": ["120g pescado", "vegetales"],
         "protein": 32, "carbs": 18, "fats": 6, "cals": cals[3]},
    ]}


# ---- sweet-marker word-boundary fix ----
def test_espinaca_not_sweet():
    assert g._is_sweet_meal({"name": "Revuelto de huevo con espinaca", "ingredients": []}, _sa) is False
    assert g._is_sweet_meal({"name": "Pollo con espinaca", "ingredients": []}, _sa) is False


def test_real_sweet_still_detected():
    assert g._is_sweet_meal({"name": "Batido de yogurt con fresas", "ingredients": []}, _sa) is True
    assert g._is_sweet_meal({"name": "Avena con piña", "ingredients": []}, _sa) is True  # piña como palabra


# ---- day-kcal-floor ----
def test_raises_low_day_with_oil_on_savory():
    d = [_day([200, 400, 167, 254])]  # día ~1021, muy bajo
    n = g._repair_day_kcal_floor_post_caps(d, _NUT, _BARIA)
    assert n > 0
    # la merienda DULCE no recibe aceite
    assert d[0]["meals"][2].get("_day_kcal_floor") is None
    # las comidas saladas (incluido el revuelto de huevo) SÍ
    topped = [m["meal"] for m in d[0]["meals"] if m.get("_day_kcal_floor")]
    assert "Desayuno" in topped and "Almuerzo" in topped and "Cena" in topped


def test_day_in_band_untouched():
    d = [_day([500, 500, 450, 500])]  # ~1950 > floor
    assert g._repair_day_kcal_floor_post_caps(d, _NUT, _BARIA) == 0


def test_non_bariatric_noop():
    d = [_day([200, 400, 167, 254])]
    assert g._repair_day_kcal_floor_post_caps(d, _NUT, {"medicalConditions": ["Ninguna"]}) == 0


def test_renal_skip():
    d = [_day([200, 400, 167, 254])]
    assert g._repair_day_kcal_floor_post_caps(d, _NUT, {"medicalConditions": ["Cirugía Bariátrica", "Enfermedad Renal Crónica"]}) == 0


def test_idempotent():
    d = [_day([200, 400, 167, 254])]
    g._repair_day_kcal_floor_post_caps(d, _NUT, _BARIA)
    assert g._repair_day_kcal_floor_post_caps(d, _NUT, _BARIA) == 0


def test_anchor_and_wiring():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-BARIATRIC-DAY-KCAL-FLOOR" in src
    assert "def _repair_day_kcal_floor_post_caps" in src
    assert "_repair_day_kcal_floor_post_caps(days, nutrition, form_data)" in src  # callsite
    assert "P1-SWEET-MARKER-WORDBOUNDARY" in src and "_SWEET_MARKER_RE" in src
