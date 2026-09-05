# -*- coding: utf-8 -*-
"""[P2-NAME-POLISH-EGG-DUP · 2026-09-05] Plan vivo 606e9017: «Revoltillo criollo ligero con casabe, Aguacate, aguacate y
Huevo». (a) el reflejo del cerrador añadía «y Huevo» a un revoltillo (solo comparaba el token «huevo»); (b) el autofix de
fruta-dulce dejó «Aguacate, aguacate»."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
from constants import strip_accents as _sa  # noqa: E402


def test_egg_dishes_already_contain_the_egg(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    for nm in ("Revoltillo criollo ligero con casabe", "Tortilla de espinaca y cebolla", "Omelette de vegetales"):
        m = {"name": nm}
        assert go._reflect_added_protein_in_name(m, "huevo", _sa) is False, nm
        assert m["name"] == nm
    # otras proteínas siguen reflejándose
    m = {"name": "Batido de Frutas"}
    assert go._reflect_added_protein_in_name(m, "yogur griego", _sa) is True and m["name"] == "Batido de Frutas con Yogur Griego"
    m2 = {"name": "Ensalada Verde"}
    assert go._reflect_added_protein_in_name(m2, "pechuga de pollo", _sa) is True


def test_adjacent_duplicate_words_are_collapsed():
    f = go._dedupe_adjacent_name_words
    assert f("Revoltillo criollo ligero con casabe, Aguacate, aguacate y Huevo") == "Revoltillo criollo ligero con casabe, Aguacate y Huevo"
    assert f("Pollo guisado con papa y papa") == "Pollo guisado con papa"
    assert f("Arroz con habichuelas rojas") == "Arroz con habichuelas rojas"
    days = [{"day": 1, "meals": [{"name": "Batido tropical de mango, mango y avena"}, {"name": "Guiso de lentejas"}]}]
    assert go._polish_meal_names(days) == 1 and days[0]["meals"][0]["name"] == "Batido tropical de mango y avena"


def test_polish_is_wired_in_finalize():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def finalize_plan_data_coherence(")
    assert "_npn = _polish_meal_names(days)" in src[i:i + 40000]
