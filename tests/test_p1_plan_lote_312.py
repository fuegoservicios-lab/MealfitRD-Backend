# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-312 · 2026-09-25] Los decimales de máquina se escriben como en la cocina.

Baterías del 25-sep (7 de 25 planes, aun después del sincronizador exacto): «pica 1 tomate y 2.22 cdas de cebolla», «mide
1.25 tazas de lentejas cocidas», «15.12g de vainitas», «Sal mínima (0.25 cdta) al gusto»."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc
import recipe_contract as rc
from culinary_coherence import build_culinary_index

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_cucharadas_y_tazas_a_fraccion_de_cocina_y_gramos_a_entero():
    m = {"ingredients": ["15.12g de vainitas", "Sal mínima (0.25 cdta) al gusto", "1.25 tazas de lentejas cocidas"],
         "ingredients_raw": ["15.12g de vainitas", "Sal mínima (0.25 cdta) al gusto", "1.25 tazas de lentejas cocidas"],
         "recipe": ["Mise en place: pica finamente 1 tomate y 2.22 cdas de cebolla; corta 15.12 g de vainitas; mide 1.25 "
                    "tazas de lentejas cocidas y 0.67 taza de habichuelas.",
                    "⚠️ Seguridad alimentaria: 0.25 cdta de sal como máximo."]}
    assert pc.decimales_de_cocina(m) == 4
    assert m["recipe"][0] == ("Mise en place: pica finamente 1 tomate y 2¼ cdas de cebolla; corta 15 g de vainitas; mide 1¼ "
                              "tazas de lentejas cocidas y ⅔ taza de habichuelas."), m["recipe"][0]
    assert m["ingredients"] == ["15g de vainitas", "Sal mínima (¼ cdta) al gusto", "1¼ tazas de lentejas cocidas"]
    assert m["ingredients_raw"][0] == "15.12g de vainitas", "lo que mide el motor no se toca"
    assert "0.25 cdta" in m["recipe"][1], "las notas no se tocan"


def test_el_numero_concuerda_con_la_unidad():
    # plan real del perfil del dueño (25-sep): «escurre 2 taza de habichuelas», «pica 3 taza de kale», «1 tazas de cebolla»
    m = {"ingredients": ["2 tazas de habichuelas rojas en lata, escurridas", "1 tazas de cebolla picada"],
         "recipe": ["Mise en place: escurre 2 taza de habichuelas rojas en lata; pica 3 taza de kale, 1 tazas de cebolla, "
                    "2½ taza de espinacas, ½ tazas de ají y 2-3 cdas de agua."]}
    pc.decimales_de_cocina(m)
    assert m["recipe"][0] == ("Mise en place: escurre 2 tazas de habichuelas rojas en lata; pica 3 tazas de kale, 1 taza de "
                              "cebolla, 2½ tazas de espinacas, ½ taza de ají y 2-3 cdas de agua."), m["recipe"][0]
    assert m["ingredients"][1] == "1 taza de cebolla picada"


def test_lo_que_no_se_toca():
    m = {"ingredients": ["0.07 ml de leche", "1.000 ml de agua"],
         "recipe": ["Bate la clara con 0.07 ml de leche; añade 0.4 taza de avena y 2.5 cm de jengibre; hierve 1.000 ml de agua."]}
    antes = [list(m["ingredients"]), list(m["recipe"])]
    assert pc.decimales_de_cocina(m) == 0 and [m["ingredients"], m["recipe"]] == antes
    assert pc.decimales_de_cocina(None) == 0


def test_corre_en_el_contrato_final():
    m = {"name": "Guiso", "meal": "Almuerzo", "ingredients": ["2¼ cdas de cebolla"], "ingredients_raw": ["2¼ cdas de cebolla"],
         "recipe": ["Mise en place: pica 1 tomate y 2.22 cdas de cebolla."]}
    rc._aplicar_meal(m, build_culinary_index([{"name": "Cebolla", "aliases": ["cebollas"], "category": "Vegetales"}]),
                     "repair")
    assert "2¼ cdas de cebolla" in m["recipe"][0], m["recipe"][0]


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").decimales_de_cocina(meal)  # [P1-PLAN-LOTE-312]' in src
