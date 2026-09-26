# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-325 · 2026-09-25] Los pasos siguen a la lista que reescribe el tope de yemas.

El tope de yemas (lote 235) corre en la cola del escudo DESPUÉS del contrato final de la receta, que ya había sincronizado
los pasos. Batería de cierre del 25-sep (colesterol + estatina): «56 g de huevo» + «6 claras» pasan a «1 huevo» + «7 claras»
y el paso seguía diciendo «prepara 1 huevo y 6 claras de huevo» — 3 de 231 comidas, las tres de este tope."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import yemas_colesterol as yc  # noqa: E402

FORM = {"medicalConditions": ["Colesterol Alto"]}


def _plan(ings, raw, pasos):
    return {"days": [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Mangú con huevos revueltos", "ingredients": list(ings),
                                           "ingredients_raw": list(raw), "recipe": list(pasos)}]}]}


def test_la_clara_que_anade_el_tope_llega_al_paso():
    # estado REAL antes del tope (batería de cierre, día 1 del perfil con estatina)
    p = _plan(["½ plátano verde", "56 g de huevo", "½ cebolla", "1 cdta de aceite de oliva", "6 claras de huevo"],
              ["0.5 plátano verde", "0.5 cebolla", "1 cdta de aceite de oliva", "56 g de Huevo", "198g de clara de huevo"],
              ["Mise en place: pela y corta ½ plátano verde en trozos; corta ½ cebolla en tiras; mide 1 cdta de aceite de "
               "oliva; prepara 1 huevo y 6 claras de huevo.",
               "El Toque de Fuego: hierve el plátano 18-20 min; revuelve el huevo y las claras 3 min.",
               "Montaje: sirve el mangú con el revoltillo al lado."])
    assert yc.topar_yemas(p, FORM) == 1
    m = p["days"][0]["meals"][0]
    assert "7 claras de huevo" in m["ingredients"] and "1 huevo" in m["ingredients"], m["ingredients"]
    assert "prepara 1 huevo y 7 claras de huevo." in m["recipe"][0], m["recipe"][0]


def test_el_paso_que_cuenta_los_huevos_no_queda_con_dos_cuentas_de_claras():
    p = _plan(["2 huevos", "3 claras de huevo"], ["100g de huevo", "99g de clara de huevo"],
              ["Mise en place: bate 2 huevos y 3 claras de huevo con sal.", "Montaje: sirve."])
    yc.topar_yemas(p, FORM)
    m = p["days"][0]["meals"][0]
    assert [i for i in m["ingredients"] if "clara" in i] == ["5 claras de huevo"], m["ingredients"]
    assert m["recipe"][0] == "Mise en place: bate 1 huevo y 5 claras de huevo con sal.", m["recipe"][0]


def test_sin_tope_los_pasos_no_se_tocan():
    pasos = ["Mise en place: prepara 1 huevo y 6 claras de huevo.", "Montaje: sirve."]
    p = _plan(["1 huevo", "5 claras de huevo"], ["50g de huevo", "165g de clara de huevo"], pasos)
    assert yc.topar_yemas(p, FORM) == 0
    assert p["days"][0]["meals"][0]["recipe"] == pasos       # el tope no toca lo que no reescribió


def test_ancla():
    src = (_BACKEND / "yemas_colesterol.py").read_text(encoding="utf-8")
    assert "_pasos_siguen_la_lista(meal)  # [P1-PLAN-LOTE-325]" in src
