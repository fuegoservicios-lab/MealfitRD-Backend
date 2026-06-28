"""[P1-BARIATRIC-CROSSDAY-CONDIMENT-CAP · 2026-06-28] Cap ACUMULADO POR DÍA de canela (≤0.5 cdta/día — cumarina cassia
~3× TDI EFSA en 1 cdta) y linaza/chía molida (≤1 cda ≈10g/día — tolerancia de fibra + no desplazar proteína). El cap
per-comida no veía la dosis sumada cross-día (canela en 3 slots = ~3 cdta). Verificado por el revisor en vivo (rechazaba
linaza 20g + canela acumulada). Mantiene ítems enteros desde el primero hasta el cap, recorta la frontera, elimina el
sobrante. Determinista, idempotente, no-op no-bariátrico.
"""
from __future__ import annotations

import re

import graph_orchestrator as g

_FORM = {"medicalConditions": ["Cirugía Bariátrica"]}
_NON = {"medicalConditions": ["Diabetes tipo 2"]}


def _total(days, kw):
    t = 0.0
    for d in days:
        for m in d["meals"]:
            for ing in m["ingredients"]:
                if kw in ing.lower():
                    mm = re.match(r"\s*(\d+(?:\.\d+)?)", ing)
                    if mm:
                        t += float(mm.group(1))
    return t


def test_canela_crossday_capped():
    days = [{"day": 0, "meals": [
        {"meal": "Desayuno", "name": "Avena", "ingredients": ["1 cdta de canela", "40 g de avena"]},
        {"meal": "Merienda PM", "name": "Yogurt", "ingredients": ["1 cdta de canela", "120 g de yogurt griego"]},
        {"meal": "Cena", "name": "Pollo", "ingredients": ["1 cdta de canela", "120g de pollo"]}]}]
    g.cap_bariatric_portions(days, _FORM)
    assert _total(days, "canela") <= 0.5 + 1e-6


def test_linaza_crossday_capped():
    days = [{"day": 0, "meals": [
        {"meal": "Desayuno", "name": "Batido", "ingredients": ["2 cda de linaza molida", "1 guineo"]},
        {"meal": "Merienda", "name": "Yogurt", "ingredients": ["1 cda de linaza molida", "120g de yogurt griego"]}]}]
    g.cap_bariatric_portions(days, _FORM)
    assert _total(days, "linaza") <= 1.0 + 1e-6


def test_under_cap_noop():
    days = [{"day": 0, "meals": [{"meal": "Desayuno", "name": "Avena",
        "ingredients": ["0.5 cdta de canela", "1 cda de linaza molida"]}]}]
    before = [list(m["ingredients"]) for m in days[0]["meals"]]
    g.cap_bariatric_portions(days, _FORM)
    assert [m["ingredients"] for m in days[0]["meals"]] == before


def test_idempotent():
    days = [{"day": 0, "meals": [
        {"meal": "Desayuno", "name": "x", "ingredients": ["1 cdta de canela", "100g pollo"]},
        {"meal": "Cena", "name": "y", "ingredients": ["1 cdta de canela", "100g pescado"]}]}]
    g.cap_bariatric_portions(days, _FORM)
    snap = [list(m["ingredients"]) for m in days[0]["meals"]]
    g.cap_bariatric_portions(days, _FORM)
    assert [m["ingredients"] for m in days[0]["meals"]] == snap


def test_non_bariatric_untouched():
    days = [{"day": 0, "meals": [
        {"meal": "Desayuno", "name": "x", "ingredients": ["2 cdta de canela", "3 cda de linaza"]},
        {"meal": "Cena", "name": "y", "ingredients": ["2 cdta de canela"]}]}]
    before = [list(m["ingredients"]) for m in days[0]["meals"]]
    g.cap_bariatric_portions(days, _NON)
    assert [m["ingredients"] for m in days[0]["meals"]] == before


def test_knob_and_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-BARIATRIC-CROSSDAY-CONDIMENT-CAP" in src
    assert "BARIATRIC_CROSSDAY_CONDIMENT_CAP_ENABLED" in src
    assert "BARIATRIC_CANELA_DAILY_CAP_CDTA" in src and "BARIATRIC_LINAZA_DAILY_CAP_CDA" in src
    assert g.BARIATRIC_CANELA_DAILY_CAP_CDTA == 0.5  # default clínico (no 1.0)


def test_bariatric_prompt_hard_rules():
    import condition_rules as cr
    b = cr.build_condition_prompt(_FORM)
    assert "½ cucharadita en TODO el día" in b           # canela
    assert "1 cucharada (≈10 g) en TODO el día" in b      # linaza
    assert "INGREDIENTES FANTASMA" in b                   # coherencia nombre↔ingr
    assert "crudos" in b.lower() and "cafe" in b.lower().replace("é", "e")  # crudos + cafeína
    # no-bariátrico no ve estas reglas
    nb = cr.build_condition_prompt(_NON)
    assert "INGREDIENTES FANTASMA" not in nb
