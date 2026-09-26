# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-333 · 2026-09-25] La plantilla del cerrador de proteína, puesta a lo que no es proteína.

Perfil del dueño, batería de cierre: «Incorpora arroz blanco crudo a la plancha o hervido y sírvelo como proteína del
plato»; otros perfiles: «Cocina cebolla a la plancha o hervida y sírvela como proteína del plato», «Añade tayota…». 22
frases en 308 planes, y en las 22 el alimento aparece en otro paso del plato: la frase es un eco que sobra."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_eco_del_arroz_sale_y_la_proteina_real_se_queda():
    m = {"ingredients": ["⅓ taza de arroz blanco crudo", "3 huevos", "2 claras de huevo", "2 nabos medianos"],
         "recipe": ["Mise en place: corta 2 nabos medianos; mide 150 g de arroz blanco cocido.",
                    "El Toque de Fuego: saltea el nabo 3-4 minutos. Incorpora arroz blanco crudo a la plancha o hervido y "
                    "sírvelo como proteína del plato. Cocina 3 huevos y 2 claras de huevo a la plancha o hervido y sírvelos "
                    "como proteína del plato.",
                    "💪 Cocina brócoli a la plancha o hervido y sírvelo como proteína del plato.",
                    "Montaje: sirve el brócoli y los vegetales sobre el arroz."]}
    assert pc.plantilla_de_proteina(m) == 2
    assert m["recipe"][1] == ("El Toque de Fuego: saltea el nabo 3-4 minutos. Cocina 3 huevos y 2 claras de huevo a la "
                              "plancha o hervido y sírvelos como proteína del plato.")
    assert len(m["recipe"]) == 3 and m["recipe"][2].startswith("Montaje:")      # el paso que solo era el eco, fuera


def test_la_proteina_o_lo_unico_que_nombra_al_alimento_se_queda():
    pasos = ["Mise en place: corta la tayota.",
             "💪 Cocina pechuga de pollo a la plancha o hervido y sírvelo como proteína del plato.",
             "El Toque de Fuego: Incorpora quinoa a la plancha o hervida y sírvela como proteína del plato.",
             "Montaje: sirve."]
    m = {"ingredients": ["½ pechuga de pollo (≈115 g)", "⅓ taza de quinoa", "½ tayota"], "recipe": list(pasos)}
    assert pc.plantilla_de_proteina(m) == 0 and m["recipe"] == pasos      # la quinoa no sale en otro paso: se queda


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").plantilla_de_proteina(meal, index)' in src
