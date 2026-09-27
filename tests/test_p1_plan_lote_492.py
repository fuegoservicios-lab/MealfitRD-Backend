# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-492 · 2026-09-27] El puré o la salsa que la receta HACE con el fresco también se reescribe.

El 466 dejó de reescribir cualquier mención tras «salsa/puré/pasta/jugo… de» —para no escribir «salsa de salsa de
tomate»— y con eso quedaron «Atún con puré de plátano verde al ajo» con batata en la lista, «integra la pasta con la
salsa de pechuga de pavo» con atún, «Sardinas en salsa de espinacas» con repollo (replay forzado de los días 21+). Un
caldo sigue siendo despensa; con la salsa de tomate de sustituto, una preparación «de tomate» lo sigue siendo; lo demás
sólo es producto si es otra línea de la lista del plato.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import sustitucion_fresca as sf  # noqa: E402


def test_el_pure_de_la_receta_cambia_con_el_fresco():
    m = {"name": "Atún con puré de plátano verde al ajo y zanahoria salteada",
         "ingredients": ["½ plátano verde mediano", "269 g de atún en agua"],
         "ingredients_raw": ["½ plátano verde mediano", "269 g de atún en agua"],
         "recipe": ["Montaje: sirve el puré de plátano verde al ajo como cama y coloca encima el atún."]}
    sf.sustituir_en_plato(m, 0, "½ plátano verde mediano", "140 g de batata", "batata")
    assert m["name"] == "Atún con puré de batata al ajo y zanahoria salteada", m["name"]
    assert m["recipe"][0] == "Montaje: sirve el puré de batata al ajo como cama y coloca encima el atún.", m["recipe"]


def test_la_salsa_de_la_receta_cambia_con_la_proteina():
    m = {"name": "Pasta integral con pavo", "ingredients": ["230 g de pechuga de pavo", "60 g de salsa de tomate"],
         "ingredients_raw": ["230 g de pechuga de pavo", "60 g de salsa de tomate"],
         "recipe": ["Montaje: integra la pasta escurrida con la salsa de pechuga de pavo y vegetales."]}
    sf.sustituir_en_plato(m, 0, "230 g de pechuga de pavo", "230 g de atún en agua", "atun en agua")
    assert "salsa de atún y vegetales" in m["recipe"][0], m["recipe"][0]
    assert "pavo" not in m["recipe"][0]


def test_el_caldo_y_la_salsa_de_tomate_no():
    m = {"name": "Pollo guisado", "ingredients": ["1 pechuga de pollo", "½ taza de caldo de pollo"],
         "ingredients_raw": ["1 pechuga de pollo", "½ taza de caldo de pollo"],
         "recipe": ["El Toque de Fuego: añade el caldo de pollo y cocina 5 minutos."]}
    sf.sustituir_en_plato(m, 0, "1 pechuga de pollo", "150 g de atún en agua", "atun en agua")
    assert "el caldo de pollo" in m["recipe"][0], m["recipe"][0]
    m = {"name": "Pasta integral", "desc": "Pasta integral en una salsa ligera de tomate, ajo y cebolla.",
         "ingredients": ["1 tomate mediano"], "ingredients_raw": ["1 tomate mediano"],
         "recipe": ["El Toque de Fuego: sofríe la cebolla y guisa el tomate 5 minutos."]}
    sf.sustituir_en_plato(m, 0, "1 tomate mediano", "60 g de salsa de tomate", "salsa de tomate")
    assert m["desc"] == "Pasta integral en una salsa ligera de tomate, ajo y cebolla.", m["desc"]
