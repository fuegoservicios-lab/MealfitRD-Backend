# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-804 · 2026-09-28] «cocina 3 huevos y 2 claras…; quita la yema de cada huevo pelado» quitaba también la
yema a los 3 enteros (batería real de embarazo, almuerzo del día 2). Con enteros en la misma frase, la cola cuenta las
claras."""
from __future__ import annotations

import pasos_cantidades as pc


def _meal(paso, lista):
    return {"name": "Arroz blanco con huevo duro y ensalada fresca", "ingredients": list(lista), "recipe": [paso]}


def test_con_enteros_la_yema_se_quita_solo_a_las_claras():
    m = _meal("El Toque de Fuego: cocina 3 huevos y 2 claras de huevo en agua hirviendo 10-12 minutos hasta que estén "
              "bien cocidos; enfríalos y pélalos.", ["3 huevos", "2 claras de huevo"])
    assert pc.claras_en_su_huevo(m) == 1
    paso = m["recipe"][0]
    assert "de cada huevo" not in paso, paso
    assert "quita la yema a 2 de ellos: usa solo su clara" in paso, paso
    assert "hiérvelas dentro de su huevo entero" in paso


def test_una_clara_con_un_entero():
    m = _meal("El Toque de Fuego: hierve 1 huevo y 1 clara de huevo 10 minutos; pélalos.", ["1 huevo", "1 clara de huevo"])
    assert pc.claras_en_su_huevo(m) == 1
    assert "quita la yema a 1 de ellos" in m["recipe"][0], m["recipe"][0]


def test_solo_claras_como_siempre():
    m = _meal("El Toque de Fuego: hierve 3 claras de huevo 10-11 minutos, enfríalas y pélalas.", ["3 claras de huevo"])
    assert pc.claras_en_su_huevo(m) == 1
    assert "quita la yema de cada huevo pelado: usa solo la clara" in m["recipe"][0], m["recipe"][0]
