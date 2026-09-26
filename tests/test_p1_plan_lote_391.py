# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-391 · 2026-09-26] Lo que se desgrana es la granada.

Plan de la batería real (adulto mayor con HTA): «desgrana 45 g de guineo», «desgrana 65 g de piña»; en el corpus
«¼ taza de guineo desgranada» y «desgrana guineo y desecha la cáscara blanca» — la sustitución de la granada deja su
verbo, su adjetivo y su cáscara."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_fruta_que_no_se_desgrana_se_pela_y_se_corta():
    m = {"recipe": ["Mise en place: desgrana 45 g de guineo, corta 10 g en cubos y desmenuza 20 g de queso fresco.",
                    "Mise en place: corta 90 g de lechosa en cubos, desgrana 65 g de piña, mide 5 g de semillas de girasol.",
                    "Mise en place: desgrana suficiente guineo para obtener ½ taza y corta ¼ lechosa en cubitos.",
                    "Mise en place: mide ¼ taza de canela en polvo, ¼ taza de guineo desgranada y ¼ cdta de semillas.",
                    "Mise en place: desgrana guineo y desecha la cáscara blanca, mide los ¾ taza de yogurt (80 g).",
                    "Desgrana ½ guineo y corta ¼ lechosa en cubos."]}
    assert pc.lo_que_se_desgrana(m) == 6
    assert m["recipe"] == [
        "Mise en place: pela y corta 45 g de guineo, corta 10 g en cubos y desmenuza 20 g de queso fresco.",
        "Mise en place: corta 90 g de lechosa en cubos, pela y corta 65 g de piña, mide 5 g de semillas de girasol.",
        "Mise en place: pela y corta suficiente guineo para obtener ½ taza y corta ¼ lechosa en cubitos.",
        "Mise en place: mide ¼ taza de canela en polvo, ¼ taza de guineo en ruedas y ¼ cdta de semillas.",
        "Mise en place: pela y corta guineo, mide los ¾ taza de yogurt (80 g).",
        "Pela y corta ½ guineo y corta ¼ lechosa en cubos."], m["recipe"]


def test_lo_que_si_se_desgrana_no_se_toca():
    pasos = ["Mise en place: desgrana 45 g de granada.", "Mise en place: lava y desgrana la uva.",
             "Mise en place: desgrana 1 mazorca de maíz y 30 g de guandules."]
    m = {"recipe": list(pasos)}
    assert pc.lo_que_se_desgrana(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").lo_que_se_desgrana(meal)  # [P1-PLAN-LOTE-391]' in src
