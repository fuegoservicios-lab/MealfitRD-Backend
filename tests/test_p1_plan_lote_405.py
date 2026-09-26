# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-405 · 2026-09-26] Las claras de la lista también se cocinan.

Batería REAL sobre el 399 (embarazo, día 2): «3 huevos» + «3 claras de huevo» en la lista y «hierve el huevo en agua
durante 10-12 min» — las claras no las cocina ningún paso (corpus: 119). Además «pon 3 huevos y 2 claras de huevo a
hervir… y pela» (12, el 390 no leía «hervir») y, con sólo claras, «Cocina huevo a la plancha o hervido» (17)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_las_claras_del_huevo_duro():
    m = {"ingredients": ["3 huevos", "3 claras de huevo", "300 g de remolacha"],
         "recipe": ["Mise en place: mide 3 huevos y 3 claras de huevo y el agua para cocinar el huevo.",
                    "El Toque de Fuego: hierve el huevo en agua durante 10-12 min, hasta que estén completamente cocidos. "
                    "Mientras tanto, asa la remolacha 18-20 min.",
                    "Montaje: agrega el huevo bien cocido cortado en cuartos."]}
    assert pc.claras_de_la_lista(m) == 1
    assert m["recipe"][1].startswith("El Toque de Fuego: hierve el huevo en agua durante 10-12 min, hasta que estén "
                                     "completamente cocidos; para las claras de huevo de la lista, hierve también con "
                                     "cáscara un huevo por cada clara y, al pelarlos, quítales la yema. Mientras tanto"), m["recipe"][1]
    assert pc.claras_de_la_lista(m) == 0


def test_a_hervir_lo_ve_el_390():
    m = {"ingredients": ["3 huevos", "2 claras de huevo"],
         "recipe": ["Mise en place: pon 3 huevos y 2 claras de huevo a hervir 9 minutos, enfría y pela; corta el casabe."]}
    assert pc.claras_en_su_huevo(m) == 1
    assert "2 claras de huevo (hiérvelas dentro de su huevo entero, con cáscara) a hervir 9 minutos" in m["recipe"][0]
    assert pc.claras_de_la_lista(m) == 0                                          # ya resuelto por el 390


def test_solo_claras_el_cerrador_las_cuaja():
    m = {"ingredients": ["2 claras de huevo", "1 torta pequeña de casabe"],
         "recipe": ["El Toque de Fuego: calienta el casabe 1-2 minutos por lado. Cocina huevo a la plancha o hervido y "
                    "sírvelo como proteína del plato.", "Montaje: sirve."]}
    assert pc.claras_de_la_lista(m) == 1
    assert m["recipe"][0].endswith("Cuaja las claras de huevo en la sartén, revueltas, hasta que estén firmes y opacas, y "
                                   "sírvelas como proteína del plato.")


def test_las_claras_ya_cocinadas_no_se_tocan():
    casos = [
        (["3 huevos", "2 claras de huevo"], ["El Toque de Fuego: bate 3 huevos y 2 claras de huevo y cuájalos en la sartén."]),
        (["3 huevos", "2 claras de huevo"], ["El Toque de Fuego: hierve 3 huevos 10 min. Cuaja las 2 claras en la sartén."]),
        (["3 huevos"], ["El Toque de Fuego: hierve los huevos 10 min y pélalos."]),
    ]
    for lista, pasos in casos:
        m = {"ingredients": list(lista), "recipe": list(pasos)}
        assert pc.claras_de_la_lista(m) == 0 and m["recipe"] == pasos, (lista, m["recipe"])


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").claras_de_la_lista(meal)  # [P1-PLAN-LOTE-405]' in src
    assert "|\\bherv(?:ir|id[oa]s?)\\b" in (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
