# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-930 · 2026-09-29] La regla de la «doble fracción» no le quita la cifra a una línea que se MIDE.

Batería real de embarazo rdv864 (día 1, cena): la lista visible decía «g de nabo pelado y cortado en rodajas de 1 cm» y la
del motor «264.35 g de nabo pelado y cortado en rodajas de 1 cm». El lote 917 le devolvía la cifra sin saber quién la
quitaba. Quién: `humanize_ingredients._collapse_double_fraction` (P1-INGREDIENT-DOUBLE-FRACTION, 29-jun), que nació para
«0.5 jugo de 0.5 limón» y trata como cantidad espuria la de CUALQUIER línea que más adelante diga «de 1 …»: «de 1 cm».

Corpus del VPS (11.719 líneas distintas del motor): la regla cambia 4; tres son las suyas («½ jugo de ½ limón», 27
apariciones) y la cuarta es el nabo. Localizado por fuerza bruta: la línea por todas las funciones del backend que
devuelven texto (144.820 llamadas).
"""
from __future__ import annotations

import pathlib

import humanize_ingredients as h

_BACKEND = pathlib.Path(__file__).resolve().parents[1]

_MEDIDAS = ("250 g de nabo pelado y cortado en rodajas de 1 cm",
            "264.35 g de nabo pelado y cortado en rodajas de 1 cm",
            "200 g de batata en cubos de 2 cm",
            "120 g de pechuga de pollo en tiras de 1 cm",
            "1 cda de jugo de 1 limón",
            "2 tazas de caldo de 1 cubito",
            "½ taza de jugo de 2 naranjas",
            "150 ml de leche de 2 % de grasa",
            "1 lata de atún de 5 onzas")


def test_la_linea_que_se_mide_conserva_su_cifra():
    for linea in _MEDIDAS:
        assert h._collapse_double_fraction(linea) == linea, linea


def test_la_doble_fraccion_de_verdad_sigue_saliendo():
    assert h._collapse_double_fraction("0.5 jugo de 0.5 limón") == "jugo de 0.5 limón"
    assert h._collapse_double_fraction("½ jugo de ½ limón") == "jugo de ½ limón"
    assert h._collapse_double_fraction("1 jugo de ½ limón") == "jugo de ½ limón"
    assert h._collapse_double_fraction("3 jugo de ½ limón") == "jugo de ½ limón"
    assert h._collapse_double_fraction("1 ralladura de 1 limón") == "ralladura de 1 limón"
    for intacta in ("150 g de arroz", "2 huevos", "jugo de ½ limón", "1 taza de leche con 1 cdta de canela"):
        assert h._collapse_double_fraction(intacta) == intacta


def test_lo_que_no_es_un_alimento_contable_no_es_una_fraccion():
    """«de 1 cm» es una medida de corte: aunque la línea no lleve unidad, su cantidad no es espuria."""
    assert h._collapse_double_fraction("2 nabos en rodajas de 1 cm") == "2 nabos en rodajas de 1 cm"
    assert h._collapse_double_fraction("1 batata en cubos de 2 cm") == "1 batata en cubos de 2 cm"
    assert h._collapse_double_fraction("3 filetes de 150 g") == "3 filetes de 150 g"


def test_el_plato_entero_sale_con_su_cifra():
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Nabo asado a la parrilla con queso blanco fresco",
        "ingredients": ["264.35 g de nabo pelado y cortado en rodajas de 1 cm", "20 g de queso blanco fresco"],
        "recipe": ["Mise en place: pela y corta el nabo.", "Montaje: sirve."]}]}]}
    h.humanize_plan_ingredients(plan)
    linea = plan["days"][0]["meals"][0]["ingredients"][0]
    assert linea[:1].isdigit() and "nabo" in linea and not linea.lower().startswith("g de"), linea
    assert plan["days"][0]["meals"][0]["ingredients_raw"][0].startswith("264.35 g de nabo")


def test_con_el_knob_apagado_la_regla_de_antes(monkeypatch):
    monkeypatch.setenv("MEALFIT_DOUBLE_FRACTION_KEEPS_MEASURED", "false")
    assert h._collapse_double_fraction(_MEDIDAS[0]) == "g de nabo pelado y cortado en rodajas de 1 cm"


def test_ancla():
    src = (_BACKEND / "humanize_ingredients.py").read_text(encoding="utf-8")
    cuerpo = src[src.index("def _collapse_double_fraction("):src.index("# Diccionario de equivalencias")]
    assert "tooltip-anchor: P1-PLAN-LOTE-930" in cuerpo and "MEALFIT_DOUBLE_FRACTION_KEEPS_MEASURED" in src
    b = (_BACKEND / "humanize_ingredients.py").read_bytes()
    assert b"\x08" not in b
