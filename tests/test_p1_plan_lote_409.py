# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-409 · 2026-09-26] El huevo duro que ningún paso hierve trae su hervor.

Replay de la cola: «pela 3 huevos y 2 claras de huevo cocidos», «ten listos 2 huevos enteros bien cocidos» con los huevos
crudos en la lista y ningún paso que los hierva (9 comidas)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_huevo_duro_trae_su_hervor():
    m = {"ingredients": ["3 huevos", "2 claras de huevo", "½ plátano verde"],
         "recipe": ["Mise en place: corta el plátano; pela 3 huevos y 2 claras de huevo cocidos.", "Montaje: sirve."]}
    assert pc.huevo_duro_de_la_lista(m) == 1
    assert m["recipe"][1] == ("💡 Cocción previa: hierve los huevos 10-12 min, pásalos a agua fría y pélalos; para las claras "
                              "de huevo de la lista, hierve también un huevo por cada clara y quítale la yema al pelarlo.")
    assert pc.huevo_duro_de_la_lista(m) == 0
    s = {"ingredients": ["2 huevos"], "recipe": ["Mise en place: ten listos 2 huevos enteros bien cocidos."]}
    assert pc.huevo_duro_de_la_lista(s) == 1
    assert s["recipe"][1] == "💡 Cocción previa: hierve los huevos 10-12 min, pásalos a agua fría y pélalos."


def test_lo_que_ya_se_hierve_no_se_toca():
    casos = [
        (["2 huevos"], ["El Toque de Fuego: hierve los huevos 10 min; pela los huevos cocidos."]),
        (["2 huevos cocidos"], ["Mise en place: pela los huevos cocidos."]),
        (["2 huevos"], ["El Toque de Fuego: cuaja los huevos en la sartén hasta que estén cocidos."]),
    ]
    for lista, pasos in casos:
        m = {"ingredients": list(lista), "recipe": list(pasos)}
        assert pc.huevo_duro_de_la_lista(m) == 0 and m["recipe"] == pasos, (lista, m["recipe"])


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").huevo_duro_de_la_lista(meal)  # [P1-PLAN-LOTE-409]' in src
