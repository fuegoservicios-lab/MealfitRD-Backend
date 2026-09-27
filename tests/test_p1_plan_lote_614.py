# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-614 · 2026-09-27] La leche de una cucharadita no es un ingrediente.

Validación del 592 (adulto mayor con hipertensión, día 2): «20 g de avena, 5 ml de leche descremada… 80 ml de agua… cocina
la avena con la leche descremada, el agua y la canela». Corpus: 126 comidas con ≤ 10 ml de leche; replay: 62 se limpian y
el resto (sin otra base líquida, remojo con la leche como único líquido, «gotas de leche», café) queda igual.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import leche_infima as li  # noqa: E402

_AVENA = {
    "meal": "Desayuno", "name": "Avena cremosa con lechosa, granada, canela y yogurt",
    "ingredients": ["20 g de avena", "5 ml de leche descremada", "35 g de lechosa", "30 g de granada",
                    "¼ cdta de canela en polvo", "1 taza de yogurt", "80 ml de agua"],
    "ingredients_raw": ["20 g de avena", "5 ml de leche descremada", "35 g de lechosa", "30 g de granada",
                        "¼ cdta de canela en polvo", "230 g de Yogurt", "80 ml de agua"],
    "recipe": ["Mise en place: mide 20 g de avena, 5 ml de leche descremada y 80 ml de agua y ¼ cdta de canela en polvo; "
               "corta 35 g de lechosa en cubos.",
               "El Toque de Fuego: cocina la avena con la leche descremada, el agua y la canela a fuego medio 7-8 minutos, "
               "removiendo hasta que quede cremosa.",
               "Montaje: sirve la avena tibia con la lechosa y la granada por encima y acompaña con yogurt.",
               "⚠️ Sodio (hipertensión/riñón): elige versiones bajas en sodio; el queso, mejor fresco y bajo en sal."],
}


def _q(meal):
    m = copy.deepcopy(meal)
    return li.quitar(m), m


def test_la_avena_de_la_validacion():
    n, m = _q(_AVENA)
    assert n == 1
    assert "5 ml de leche descremada" not in m["ingredients"] and "5 ml de leche descremada" not in m["ingredients_raw"]
    assert m["recipe"][0].startswith("Mise en place: mide 20 g de avena y 80 ml de agua y ¼ cdta de canela en polvo;")
    assert "cocina la avena con el agua y la canela a fuego medio" in m["recipe"][1]
    assert m["recipe"][3] == _AVENA["recipe"][3]                                          # las notas no se tocan


def test_la_leche_era_el_liquido_nombrado_pasa_a_el_agua():
    m = copy.deepcopy(_AVENA)
    m["recipe"][1] = ("El Toque de Fuego: hierve el huevo en agua 10-12 min; en otra olla cocina la avena con la leche y la "
                      "canela a fuego medio durante 6-8 min.")
    n, m = _q(m)
    assert n == 1 and "en otra olla cocina la avena con el agua y la canela a fuego medio" in m["recipe"][1]


def test_batido_con_yogur():
    n, m = _q({"name": "Batido de mango con yogur griego",
               "ingredients": ["⅔ taza de yogurt griego sin azúcar", "315 g de mango", "5 ml de leche descremada"],
               "recipe": ["Mise en place: pela y corta 315 g de mango; mide ⅔ taza de yogurt griego sin azúcar (75 g) y 5 ml "
                          "de leche descremada.",
                          "Montaje: licúa el mango, el yogurt griego y la leche descremada hasta obtener un batido uniforme."]})
    assert n == 1
    assert m["recipe"][0].endswith("mide ⅔ taza de yogurt griego sin azúcar (75 g).")
    assert m["recipe"][1] == "Montaje: licúa el mango y el yogurt griego hasta obtener un batido uniforme."


def test_masa_de_panqueques():
    n, m = _q({"name": "Panqueques suaves de avena con lechosa",
               "ingredients": ["30 g de avena", "1 huevo", "1.1 ml de leche descremada", "100 g de lechosa"],
               "recipe": ["Mise en place: muele los 30 g de avena; mide 1 ml de leche descremada, separa 100 g de lechosa.",
                          "El Toque de Fuego: bate el huevo con la leche descremada, incorpora la avena molida y cocina."]})
    assert n == 1
    assert m["recipe"][0] == "Mise en place: muele los 30 g de avena; separa 100 g de lechosa."
    assert m["recipe"][1] == "El Toque de Fuego: bate el huevo, incorpora la avena molida y cocina."


@pytest.mark.parametrize("cambio", [
    {"name": "Café con leche y pan con queso"},                                            # un chorrito es la receta
    {"ingredients": ["20 g de avena", "5 ml de leche descremada", "35 g de lechosa"]},     # avena cocida sin su agua
    {"ingredients": ["20 g de avena", "5 ml de leche descremada", "150 ml de leche entera", "80 ml de agua"]},  # dos leches
    {"ingredients": ["20 g de avena", "15 ml de leche descremada", "80 ml de agua"]},      # 1 cda: sobre el umbral
])
def test_no_se_toca(cambio):
    m = copy.deepcopy(_AVENA)
    m.update(cambio)
    antes = copy.deepcopy(m)
    assert li.quitar(m) == 0 and m == antes


def test_remojo_con_la_leche_como_unico_liquido():
    m = {"name": "Avena remojada de melón y manzana",
         "ingredients": ["30 g de avena en hojuelas", "5 ml de leche", "190 g de melón", "80 g de yogurt griego entero"],
         "recipe": ["Montaje: mezcla la avena con la leche y la canela y déjala reposar 3 min hasta que espese; acompaña "
                    "con yogurt natural entero."]}
    antes = copy.deepcopy(m)
    assert li.quitar(m) == 0 and m == antes


def test_gotas_de_leche_es_la_receta():
    m = copy.deepcopy(_AVENA)
    m["recipe"][2] = "Montaje: bate el queso ricotta con unas gotas de leche y colócalo encima de la avena."
    antes = copy.deepcopy(m)
    assert li.quitar(m) == 0 and m == antes


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("leche_infima").quitar(meal)  # [P1-PLAN-LOTE-614]' in src
