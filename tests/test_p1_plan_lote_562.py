# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-562 · 2026-09-27] El huevo que se BATE no está «bien cocido» todavía.

Batería real (embarazo): «bate 1 huevo bien cocido con 5 ml de leche» en unos panqueques; en el corpus, «bate 2 huevos
bien cocidos con sal al gusto» (el revisor rechazó uno: un reintento).
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import huevo_que_se_bate as hb  # noqa: E402

_NOTA = ("🤰 Seguridad alimentaria (embarazo/lactancia): cocina el huevo POR COMPLETO (yema y clara firmes, sin puntos "
         "líquidos).")


def test_panqueques_de_embarazo():
    m = {"ingredients": ["30 g de avena", "1 huevo bien cocido", "5 ml de leche pasteurizada"],
         "ingredients_raw": ["30 g de avena", "1 huevo bien cocido", "5 ml de leche pasteurizada"],
         "recipe": ["Mise en place: Muele 30 g de avena; bate 1 huevo bien cocido con 5 ml de leche pasteurizada.",
                    "El Toque de Fuego: cocina la masa en la sartén 2-3 min por lado, hasta que no quede líquida.", _NOTA]}
    assert hb.limpiar(m) >= 3
    assert m["ingredients"][1] == "1 huevo" and m["ingredients_raw"][1] == "1 huevo"
    assert "bate 1 huevo con 5 ml" in m["recipe"][0]
    assert m["recipe"][2] == _NOTA, "la nota de seguridad no se toca"


def test_el_punto_de_coccion_del_toque_no_se_toca():
    # replay del corpus: la primera versión convertía «(huevo BIEN cocido para el embarazo)» en «(huevo para el embarazo)»
    toque = ("El Toque de Fuego: agrega los huevos batidos y cocina 4-5 min hasta que cuajen por completo (huevo BIEN "
             "cocido para el embarazo).")
    m = {"ingredients": ["3 huevos"], "recipe": ["Mise en place: bate 3 huevos con sal.", toque]}
    hb.limpiar(m)
    assert m["recipe"][1] == toque


def test_ricotta_batida_y_huevos_duros_en_mitades():
    m = {"ingredients": ["2 huevos bien cocidos", "22 g de ricotta"],
         "recipe": ["Montaje: bate la ricotta con el cebollín y úntala; corona con los huevos bien cocidos en mitades."]}
    assert hb.limpiar(m) == 0 and m["ingredients"][0] == "2 huevos bien cocidos"


def test_la_masa_de_harina_y_huevo():
    m = {"ingredients": ["2 huevos bien cocidos", "35 g de harina de maíz"],
         "recipe": ["El Toque de Fuego: incorpora la harina de maíz a los huevos con un poco de agua hasta obtener una "
                    "masa espesa; cocina la tortilla 5 min por lado."]}
    assert hb.limpiar(m) == 1 and m["ingredients"][0] == "2 huevos"


def test_huevos_duros_que_no_se_baten_se_quedan():
    m = {"ingredients": ["3 huevos bien cocidos"],
         "recipe": ["Mise en place: corta 3 huevos bien cocidos por la mitad.", "Montaje: sirve con el mangú."]}
    assert hb.limpiar(m) == 0 and m["ingredients"] == ["3 huevos bien cocidos"]


def test_ancla_en_la_cola():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("huevo_que_se_bate").limpiar(meal)  # [P1-PLAN-LOTE-562]' in src
