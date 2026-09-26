# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-356 · 2026-09-26] Los gramos del paso son los de la PIEZA que cuenta la lista (sin «≈»).

Plan renal real: «¼ filete de pescado» en la lista y en el motor (37,5 g) y «mide 90 g de tilapia» en el paso; en los
planes recientes, «1½ pechugas de pollo» (255 g) con «corta 300 g de pechuga de pollo»."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


class _DB:
    _T = {"¼ filete de pescado": (37.5, "Filete de pescado blanco"), "1½ pechugas de pollo": (255.0, "Pechuga de pollo"),
          "1 pechuga de pollo": (170.0, "Pechuga de pollo"), "1 filete de pescado": (150.0, "Filete de pescado blanco"),
          "145 g de filete de pescado blanco": (145.0, "Filete de pescado blanco"),
          "271.78 g de pechuga de pollo cocida y enfriada": (271.78, "Pechuga de pollo")}

    def macros_from_ingredient_string(self, s):
        g, n = self._T.get(s.strip(), (0.0, ""))
        return {"grams": g, "name": n}


def test_el_paso_pesa_la_pieza_del_catalogo():
    m = {"ingredients": ["¼ filete de pescado", "2 tortillas de trigo", "1 tomate mediano"],
         "ingredients_raw": ["2 tortillas de trigo", "1 tomate mediano", "¼ filete de pescado"],
         "recipe": ["Mise en place: pica el tomate; mide 90 g de tilapia y 2 tortillas de trigo.",
                    "El Toque de Fuego: cocina la tilapia 3-4 min por lado."]}
    assert pc.pieza_del_catalogo(m, _DB()) == 1
    assert "mide 40 g de tilapia" in m["recipe"][0], m["recipe"][0]
    p = {"ingredients": ["1½ pechugas de pollo", "150 g de arroz blanco"], "ingredients_raw": ["1½ pechugas de pollo"],
         "recipe": ["Mise en place: corta 300 g de pechuga de pollo en cubos."]}
    assert pc.pieza_del_catalogo(p, _DB()) == 1 and "corta 255 g de pechuga de pollo" in p["recipe"][0], p["recipe"]


def test_manda_la_linea_del_motor_en_gramos():
    # la visible cuenta «1 filete» (150 g de catálogo) pero el motor mide 145 g: manda el motor
    m = {"ingredients": ["1 filete de pescado", "⅓ taza de arroz blanco"],
         "ingredients_raw": ["145 g de filete de pescado blanco", "85 g de arroz blanco"],
         "recipe": ["Mise en place: corta 205 g de pescado blanco en tiras."]}
    assert pc.pieza_del_catalogo(m, _DB()) == 1 and "corta 145 g de pescado blanco" in m["recipe"][0], m["recipe"]
    # el motor mide la pechuga COCIDA: un paso en crudo no se toca; uno cocido, sí
    c = {"ingredients": ["1½ pechugas de pollo"], "ingredients_raw": ["271.78 g de pechuga de pollo cocida y enfriada"],
         "recipe": ["Mise en place: corta 300 g de pechuga de pollo en tiras."]}
    assert pc.pieza_del_catalogo(c, _DB()) == 0
    c["recipe"] = ["Mise en place: desmenuza 300 g de pechuga de pollo cocida."]
    assert pc.pieza_del_catalogo(c, _DB()) == 1 and "desmenuza 270 g de pechuga" in c["recipe"][0], c["recipe"]


def test_lo_que_no_se_toca():
    casos = [
        (["1 pechuga de pollo"], ["Mise en place: desmenuza 75 g de pechuga de pollo ya cocida."]),        # cocido
        (["1 pechuga de pollo"], ["Mise en place: reserva la mitad, 100 g de pechuga, para la cena."]),     # reparto
        (["1 pechuga de pollo", "50 g de pollo desmenuzado"], ["Mise en place: corta 250 g de pollo."]),    # ambiguo
        (["1 pechuga de pollo"], ["Mise en place: corta 250 g de pechuga.", "Montaje: sirve 200 g de pechuga."]),  # dos cifras
    ]
    for lista, pasos in casos:
        m = {"ingredients": list(lista), "recipe": list(pasos)}
        assert pc.pieza_del_catalogo(m, _DB()) == 0 and m["recipe"] == pasos, (lista, m["recipe"])
    m = {"ingredients": ["1 pechuga de pollo"], "recipe": ["Corta 250 g de pechuga."]}
    assert pc.pieza_del_catalogo(m, None) == 0                                                            # sin catálogo


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").pieza_del_catalogo(meal, db)  # [P1-PLAN-LOTE-356]' in src
