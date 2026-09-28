# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-735 · 2026-09-28] Las claras de la lista se cocinan también cuando el huevo va a la sartén.

Batería real sobre el 659 (adulto mayor con HTA): «2 huevos» + «1 clara de huevo» con «casca 2 huevos en la misma sartén
y cocínalos 3-4 minutos hasta que la clara y la yema cuajen», y «3 huevos» + «3 claras de huevo» con «casca los huevos
dentro … hasta que la clara y la yema cuajen»: ningún paso cocinaba las claras de la lista.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import claras_que_se_cocinan as cq  # noqa: E402

_NOTA = ("⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes, sin partes líquidas) antes "
         "de servir; evita el huevo crudo o poco cocido.")


def test_la_clara_se_cuaja_junto_a_los_huevos_cascados():
    m = {"name": "Yautía majada con huevos al orégano",
         "ingredients": ["½ pedazo de yautía (≈132 g)", "2 huevos", "1½ tomates medianos", "1 clara de huevo"],
         "recipe": [
             "Mise en place: pela y corta la yautía en cubos pequeños.",
             "El Toque de Fuego: hierve la yautía 12-15 minutos y májala. Aparta el sofrito, casca 2 huevos en la misma "
             "sartén y cocínalos 3-4 minutos hasta que la clara y la yema cuajen. Sirve caliente.",
             "Montaje: sirve la yautía majada con los huevos encima.",
             _NOTA,
         ]}
    assert cq.integrar(m) == 1
    assert m["recipe"][1] == (
        "El Toque de Fuego: hierve la yautía 12-15 minutos y májala. Aparta el sofrito, casca 2 huevos en la misma sartén "
        "y cocínalos 3-4 minutos hasta que la clara y la yema cuajen. Vierte también la clara de huevo junto a los huevos "
        "y cocínala hasta que esté firme y opaca. Sirve caliente.")
    assert m["recipe"][3] == _NOTA


def test_las_claras_se_cuajan_en_la_salsa():
    m = {"name": "Huevos guisados en salsa de tomate",
         "ingredients": ["3 huevos", "1⅔ tazas de tomate picado", "3 claras de huevo"],
         "recipe": [
             "Mise en place: pica el tomate. Mide 3 huevos y 3 claras de huevo.",
             "El Toque de Fuego: sofríe el tomate 4 minutos. Abre tres huecos en la salsa, casca los huevos dentro, tapa y "
             "cocina 6-8 minutos hasta que la clara y la yema cuajen.",
             "Montaje: sirve los huevos con su salsa.",
         ]}
    assert cq.integrar(m) == 1
    assert m["recipe"][1].endswith("hasta que la clara y la yema cuajen. Vierte también las 3 claras de huevo junto a "
                                   "los huevos y cocínalas hasta que estén firmes y opacas.")


def test_las_claras_se_baten_con_los_huevos():
    m = {"name": "Revoltillo con palmito", "ingredients": ["2 huevos", "¾ taza de palmito", "3 claras de huevo"],
         "recipe": ["Mise en place: corta el palmito. Bate los 2 huevos con una pizca de orégano.",
                    "El Toque de Fuego: vierte los huevos batidos y revuelve 4-5 minutos hasta que queden cuajados."]}
    assert cq.integrar(m) == 1
    assert m["recipe"][0] == ("Mise en place: corta el palmito. Bate los 2 huevos y las 3 claras de huevo con una pizca "
                              "de orégano.")


def test_el_revoltillo_las_echa_con_los_huevos():
    m = {"name": "Mangú ligero con revoltillo", "ingredients": ["1 plátano verde", "2 huevos", "4 claras de huevo"],
         "recipe": ["Mise en place: mide 2 huevos y 4 claras de huevo.",
                    "El Toque de Fuego: cocina la cebolla 2 min, agrega los huevos y revuelve hasta que cuajen."]}
    assert cq.integrar(m) == 1
    assert m["recipe"][1] == ("El Toque de Fuego: cocina la cebolla 2 min, agrega los huevos y las 4 claras de huevo y "
                              "revuelve hasta que cuajen.")


def test_cascadas_y_batidas_con_los_huevos_ya_estan_usadas():
    m = {"name": "Revoltillo de cundeamor", "ingredients": ["3 huevos", "2 claras de huevo"],
         "recipe": ["Mise en place: casca 3 huevos y 2 claras de huevo y bátelos con sal.",
                    "El Toque de Fuego: vierte los huevos batidos y revuelve 4 minutos."]}
    antes = list(m["recipe"])
    assert cq.integrar(m) == 0 and m["recipe"] == antes


def test_bata_con_numero_ya_las_usa():
    m = {"name": "Tostadas criollas con huevo", "ingredients": ["3 huevos", "5 claras de huevo"],
         "recipe": ["Mise en place: pique la cebolla; bata 3 huevos y 5 claras de huevo y prepare el pan.",
                    "El Toque de Fuego: vierta la mezcla en la sartén y revuelva 3-4 minutos."]}
    antes = list(m["recipe"])
    assert cq.integrar(m) == 0 and m["recipe"] == antes


def test_los_huevos_ya_cocidos_no_arrastran_la_clara_cruda():
    m = {"name": "Mangú con huevos guisados", "ingredients": ["2 huevos", "1 clara de huevo"],
         "recipe": ["El Toque de Fuego: cocina el pimiento 6-8 minutos; añade los huevos ya bien cocidos y cortados, y "
                    "calienta 2 minutos."]}
    antes = list(m["recipe"])
    assert cq.integrar(m) == 0 and m["recipe"] == antes


def test_en_la_mise_en_place_no_se_cocina():
    # el «casca» de la mise en place no se toca; las claras van al paso que cocina los huevos
    m = {"name": "Tostadas con huevo", "ingredients": ["3 huevos", "2 claras de huevo"],
         "recipe": ["Mise en place: corta el tomate; casca 3 huevos y mide el aceite.",
                    "El Toque de Fuego: cocina 3 huevos en una sartén 3-4 min, hasta que la clara esté cuajada."]}
    assert cq.integrar(m) == 1
    assert m["recipe"][0] == "Mise en place: corta el tomate; casca 3 huevos y mide el aceite."
    assert m["recipe"][1].endswith("Cuaja también las 2 claras de huevo en la sartén, revueltas, hasta que estén firmes "
                                   "y opacas.")


def test_un_solo_huevo_concuerda():
    m = {"name": "Bowl de casabe con huevo", "ingredients": ["1 huevo", "2 claras de huevo"],
         "recipe": ["El Toque de Fuego: casca el huevo entero y cocínalo 3-4 minutos hasta que la clara y la yema "
                    "cuajen."]}
    assert cq.integrar(m) == 1
    assert m["recipe"][0].endswith("Vierte también las 2 claras de huevo junto al huevo y cocínalas hasta que estén "
                                   "firmes y opacas.")


def test_con_huevos_a_la_plancha_las_claras_se_cuajan_aparte():
    # batería real sobre el 636 (estudiante económico, día 1): «3 huevos» + «3 claras de huevo» y sólo «Cocina huevos a
    # la plancha… hasta que las claras estén cuajadas» (esa es la clara del huevo entero)
    m = {"name": "Tostadas integrales con huevo a la plancha", "ingredients": ["3 huevos", "3 claras de huevo"],
         "recipe": ["Mise en place: prepara 3 huevos y 3 claras de huevo.",
                    "El Toque de Fuego: calienta el aceite. Cocina huevos a la plancha, añadiendo sal al gusto, hasta que "
                    "las claras estén cuajadas.",
                    "Montaje: coloca los huevos a la plancha sobre las tostadas."]}
    assert cq.integrar(m) == 1
    assert m["recipe"][1].endswith("hasta que las claras estén cuajadas. Cuaja también las 3 claras de huevo en la "
                                   "sartén, revueltas, hasta que estén firmes y opacas.")


def test_revuelve_y_usted_tambien_las_echan():
    for paso, esperado in (
        ("El Toque de Fuego: calienta el aceite y revuelve los huevos durante 3-4 minutos.",
         "El Toque de Fuego: calienta el aceite y revuelve los huevos y las 2 claras de huevo durante 3-4 minutos."),
        ("El Toque de Fuego: sofría la cebolla, agregue los huevos batidos y cocine 3 minutos.",
         "El Toque de Fuego: sofría la cebolla, agregue los huevos batidos y las 2 claras de huevo y cocine 3 minutos."),
    ):
        m = {"name": "Revoltillo", "ingredients": ["3 huevos", "2 claras de huevo"], "recipe": [paso]}
        assert cq.integrar(m) == 1
        assert m["recipe"][0] == esperado


def test_cocina_los_huevos_en_la_sarten_cuaja_las_claras_aparte():
    m = {"name": "Tostadas integrales con huevo", "ingredients": ["3 huevos", "2 claras de huevo"],
         "recipe": ["El Toque de Fuego: cocina 3 huevos en una sartén con el aceite a fuego medio 3-4 min, hasta que la "
                    "clara esté cuajada."]}
    assert cq.integrar(m) == 1
    assert m["recipe"][0].endswith("Cuaja también las 2 claras de huevo en la sartén, revueltas, hasta que estén firmes "
                                   "y opacas.")


def test_si_un_paso_ya_cocina_las_claras_no_se_toca():
    m = {"name": "Revoltillo", "ingredients": ["2 huevos", "3 claras de huevo"],
         "recipe": ["El Toque de Fuego: bate los huevos con las claras y cuájalos revueltos 4 minutos."]}
    antes = list(m["recipe"])
    assert cq.integrar(m) == 0 and m["recipe"] == antes


def test_el_huevo_duro_es_del_405():
    m = {"name": "Huevo duro", "ingredients": ["2 huevos", "2 claras de huevo"],
         "recipe": ["El Toque de Fuego: hierve los huevos 10-12 minutos; para las claras de huevo de la lista, hierve "
                    "también con cáscara un huevo por cada clara y, al pelarlos, quítales la yema."]}
    antes = list(m["recipe"])
    assert cq.integrar(m) == 0 and m["recipe"] == antes


def test_enganchado_tras_el_405():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i = src.index('__import__("claras_que_se_cocinan").integrar(meal)')
    assert i > src.index('__import__("pasos_cantidades").claras_de_la_lista(meal)')
    assert i < src.index('__import__("claras_pochadas").alinear(meal)')
