# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-311 · 2026-09-25] La avena COCIDA lleva líquido para cocinarse.

Baterías del 25-sep: 9 desayunos en 25 planes con «30 g de avena» y «15 ml de leche descremada» (el motor encoge la leche
para cuadrar el día). Con el paso ya fiel a la lista (lote 302) la receta sería imposible; lo que falta es AGUA (0 kcal,
fuera de la compra y del guard de coherencia)."""
from __future__ import annotations

import copy
import pathlib

import avena_liquido as al
import recipe_contract as rc
from culinary_coherence import build_culinary_index

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _desayuno():
    ings = ["30 g de avena", "15 ml de leche descremada", "½ pera mediana", "2 huevos", "1 cebolla roja",
            "¾ cdta de aceite de oliva", "Canela en polvo al gusto", "2 claras de huevo"]
    return {"name": "Avena cremosa con pera y huevos revueltos con cebolla roja", "meal": "Desayuno",
            "ingredients": list(ings), "ingredients_raw": list(ings),
            "recipe": ["Mise en place: corta la pera en cubos; mide 30 g de avena y 15 ml de leche descremada; corta 1 "
                       "cebolla roja y bate 2 huevos y 2 claras de huevo.",
                       "El Toque de Fuego: cocina la avena con la leche descremada y la canela en una olla a fuego medio "
                       "durante 6-8 minutos, removiendo hasta que esté cremosa; aparte, revuelve los huevos.",
                       "Montaje: sirve la avena con los cubos de pera."]}


def test_quince_ml_de_leche_no_cocinan_treinta_gramos_de_avena():
    m = _desayuno()
    assert al.completar(m) == 110                        # 4 ml/g − 15 ml, a la decena hacia arriba
    assert m["ingredients"][-1] == "110 ml de agua" and m["ingredients_raw"][-1] == "110 ml de agua"
    assert "mide 30 g de avena y 15 ml de leche descremada y 110 ml de agua;" in m["recipe"][0], m["recipe"][0]
    assert m["_avena_liquido"] == 110
    assert al.completar(m) == 0, "idempotente: ya hay 125 ml para 30 g"


def test_la_linea_de_agua_que_ya_existe_crece():
    m = _desayuno()
    m["ingredients"].append("20 ml de agua")
    m["ingredients_raw"].append("20 ml de agua")
    m["recipe"][0] = m["recipe"][0].replace("15 ml de leche descremada;", "15 ml de leche descremada y 20 ml de agua;")
    assert al.completar(m) == 90
    assert "110 ml de agua" in m["ingredients"] and "20 ml de agua" not in m["ingredients"]
    assert "15 ml de leche descremada y 110 ml de agua;" in m["recipe"][0], m["recipe"][0]


def test_el_peso_entre_parentesis_manda_sobre_la_taza():
    m = _desayuno()
    m["ingredients"][0] = m["ingredients_raw"][0] = "1 taza de avena (70 g)"
    m["recipe"][0] = m["recipe"][0].replace("mide 30 g de avena", "mide 1 taza de avena (70 g)")
    assert al.completar(m) == 270                         # 4 × 70 − 15 = 265 → 270 (con la taza, 85 g, serían 330)


def test_sin_medida_en_los_pasos_el_paso_que_la_cocina_lo_dice():
    m = _desayuno()
    m["recipe"][0] = "Mise en place: corta la pera en cubos."
    al.completar(m)
    assert m["recipe"][1].endswith("Completa el líquido con 110 ml de agua para que la avena se cocine."), m["recipe"][1]


def test_lo_que_no_se_toca():
    suficiente = _desayuno()
    suficiente["ingredients"][1] = suficiente["ingredients_raw"][1] = "150 ml de leche descremada"
    remojada = _desayuno()
    remojada["recipe"][1] = "Deja la avena remojando toda la noche en la nevera con la leche y la canela."
    panqueque = _desayuno()
    panqueque["name"] = "Panqueques de avena con pera"
    cruda = _desayuno()
    cruda["recipe"][1] = "Montaje: sirve la avena con la leche fría y la pera por encima."
    # replay de las baterías: la avena TOSTADA como topping («tuesta la avena … a fuego medio; déjala enfriar») no es gacha
    tostada = _desayuno()
    tostada["recipe"][1] = "El Toque de Fuego: tuesta la avena en una sartén seca a fuego medio hasta que quede aromática."
    # una línea de agua rota («½ agua») no se deja medir: no se adivina
    rota = _desayuno()
    rota["ingredients"].append("½ agua")
    for m in (suficiente, remojada, panqueque, cruda, tostada, rota):
        antes = copy.deepcopy(m)
        assert al.completar(m) == 0 and m == antes, m["name"]
    assert al.completar({"ingredients": None, "recipe": "x"}) == 0


def test_corre_en_el_contrato_final():
    m = _desayuno()
    idx = build_culinary_index([{"name": "Avena", "aliases": ["avena en hojuelas"], "category": "Granos"},
                                {"name": "Leche descremada", "aliases": ["leche"], "category": "Lácteos"}])
    rc._aplicar_meal(m, idx, "repair")
    assert "110 ml de agua" in m["ingredients"], m["ingredients"]


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("avena_liquido").completar(meal)               # [P1-PLAN-LOTE-311]' in src
