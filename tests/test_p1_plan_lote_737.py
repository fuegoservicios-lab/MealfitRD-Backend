# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-737 · 2026-09-28] El ave o el cerdo crudos de la lista siempre dicen cuándo están hechos.

Batería real sobre el 636 (estudiante económico, día 3, cena): «💪 Cocina pechuga de pollo a la plancha o hervida y
sírvela como proteína del plato.» con «60 g de pechuga de pollo» cruda en la lista y ni 74 °C ni nota en el plato; en el
corpus reciente, 88 de 698 comidas con ave cruda sin ningún criterio de cocción.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import ave_con_su_punto as ap  # noqa: E402

_CERRADOR = "💪 Cocina pechuga de pollo a la plancha o hervida y sírvela como proteína del plato."


def test_el_pollo_del_cerrador_lleva_la_nota_y_la_frase_intacta():
    m = {"name": "Canoas de plátano verde con edamame y pechuga de pollo",
         "ingredients": ["½ plátano verde mediano", "300 g de edamame cocido", "60 g de pechuga de pollo"],
         "recipe": ["El Toque de Fuego: cocina las canoas en la airfryer 15-18 min.", _CERRADOR,
                    "Montaje: sirve las canoas. Acompaña con edamame."]}
    assert ap.asegurar(m) == 1
    assert m["recipe"][1] == _CERRADOR                 # la frase del cerrador la reconocen otros por su texto exacto
    assert m["recipe"][-1] == ap.NOTA_AVE
    assert ap.asegurar(m) == 0 and m["recipe"].count(ap.NOTA_AVE) == 1


def test_dorar_3_minutos_sin_punto_lleva_la_nota():
    m = {"name": "Pechuga planchada al limón", "ingredients": ["½ pechuga de pollo (≈98 g)", "1 taza de repollo"],
         "recipe": ["El Toque de Fuego: dora la pechuga de pollo con el limón durante 3-4 min, volteando para calentar "
                    "de manera uniforme.", "Montaje: sirve la pechuga con la ensalada."]}
    assert ap.asegurar(m) == 1 and m["recipe"][-1] == ap.NOTA_AVE


def test_con_el_criterio_ya_dicho_no_se_toca():
    for paso in ("El Toque de Fuego: saltea el pollo 8-10 minutos, hasta que alcance 74 °C por dentro.",
                 "El Toque de Fuego: cocina el pollo hasta que no quede rosado por dentro y los jugos salgan claros.",
                 "⚠️ Seguridad alimentaria: el pollo/cerdo debe cocinarse por completo (interior sin partes rosadas).") :
        m = {"name": "Pollo salteado", "ingredients": ["150 g de pechuga de pollo"], "recipe": [paso, "Montaje: sirve."]}
        antes = list(m["recipe"])
        assert ap.asegurar(m) == 0 and m["recipe"] == antes


def test_el_ave_ya_cocida_o_de_lata_no_se_toca():
    for linea in ("75 g de pechuga de pollo cocida desmenuzada", "1 lata de pollo en agua", "50 g de jamón de pavo"):
        m = {"name": "Wrap", "ingredients": [linea], "recipe": ["Montaje: rellena el wrap con el pollo y sirve."]}
        antes = list(m["recipe"])
        assert ap.asegurar(m) == 0 and m["recipe"] == antes


def test_sin_ave_ni_cerdo_no_se_toca():
    m = {"name": "Pescado a la plancha", "ingredients": ["120 g de filete de pescado"],
         "recipe": ["💪 Cocina filete de pescado a la plancha o hervido y sírvelo como proteína del plato."]}
    antes = list(m["recipe"])
    assert ap.asegurar(m) == 0 and m["recipe"] == antes


def test_enganchado_tras_ave_hasta_74():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i = src.index('__import__("ave_con_su_punto").asegurar(meal)')
    assert i > src.index('__import__("pasos_cantidades").ave_hasta_74(meal)')
