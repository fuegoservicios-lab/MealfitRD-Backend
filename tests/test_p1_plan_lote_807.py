# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-807 · 2026-09-29] En el embarazo, el queso blanco que se puede dorar recibe su paso, no sólo la nota.

2.ª batería real de embarazo del 29-sep: «Guiso ligero de lentejas con remolacha y queso blanco fresco» servía el queso
en cubos, frío, con la nota del lote 193 pidiendo calentarlo hasta que humee.
"""
from __future__ import annotations

import copy

import embarazo_seguro as es

_EMBARAZO = {"medicalConditions": ["Embarazo"]}
_GUISO = {
    "name": "Guiso ligero de lentejas con remolacha y queso blanco fresco",
    "ingredients": ["⅔ taza de lentejas secas", "50 g de remolacha", "40 g de queso blanco pasteurizado", "20 g de tomate"],
    "recipe": ["Mise en place: corta la remolacha en cubos pequeños y el tomate en dados.",
               "El Toque de Fuego: hierve la remolacha 15-20 minutos; cocina el tomate y añade las lentejas 5-7 minutos.",
               "Montaje: sirve las lentejas con la remolacha y el queso blanco pasteurizado en cubos."]}


def _etiquetar(meal, form=_EMBARAZO):
    plan = {"days": [{"day": 1, "meals": [copy.deepcopy(meal)]}]}
    es.etiquetar(plan, form)
    return plan["days"][0]["meals"][0]["recipe"]


def test_el_queso_blanco_se_dora_en_un_paso():
    rec = _etiquetar(_GUISO)
    i = next(i for i, p in enumerate(rec) if p.startswith("El Toque de Fuego"))
    assert rec[i].endswith("5-7 minutos. Dora el queso blanco pasteurizado en la sartén caliente, 1-2 minutos por lado, hasta "
                           "que humee y esté bien caliente por dentro (74 °C)."), rec[i]   # [P1-PLAN-LOTE-863]
    assert rec[i + 1].startswith("Montaje"), rec
    assert any("Seguridad alimentaria (embarazo)" in p for p in rec), "la nota sigue"
    assert _etiquetar({**_GUISO, "recipe": rec}) == rec, "idempotente"


def test_plato_frio_ya_calentado_o_lactancia_sin_paso():
    vasito = {**_GUISO, "name": "Vasito fresco de yogur natural con mango y queso blanco fresco"}
    assert not any("Dora el queso" in p for p in _etiquetar(vasito))
    dorado = copy.deepcopy(_GUISO)
    dorado["recipe"][1] += " Dora el queso blanco en la sartén 2 minutos por lado."
    assert sum("Dora el queso" in p for p in _etiquetar(dorado)) == 1, "el suyo, no otro"
    assert not any("Dora el queso" in p for p in _etiquetar(_GUISO, {"medicalConditions": ["Lactancia"]}))
    # el queso mezclado en la masa se cuece con ella (pastelitos al airfryer de la batería)
    pastelitos = {**_GUISO, "name": "Pastelitos de nabo al airfryer con queso blanco pasteurizado y edamame",
                  "recipe": ["Mise en place: pela y corta 250 g de nabo; desmenuza 20 g de queso blanco fresco.",
                             "El Toque de Fuego: cocina el nabo en agua hirviendo 12-15 minutos y májalo. Integra el queso "
                             "y el perejil, forma pastelitos y pincélalos con aceite. Cocina en el airfryer a 200 °C "
                             "durante 8-10 minutos, hasta que estén dorados.",
                             "Montaje: presenta los pastelitos recién hechos."]}
    assert not any("Dora el queso" in p for p in _etiquetar(pastelitos))


def test_cottage_se_queda_con_la_nota():
    cottage = {**_GUISO, "name": "Lechosa con maní y queso cottage",
               "ingredients": ["150 g de lechosa", "10 g de maní", "70 g de queso cottage pasteurizado"],
               "recipe": ["Mise en place: corta la lechosa.", "Montaje: sirve la lechosa con el maní y el cottage."]}
    rec = _etiquetar(cottage)
    assert not any("Dora el queso" in p for p in rec) and any("humee" in p for p in rec)
