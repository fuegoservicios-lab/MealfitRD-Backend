# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-612 · 2026-09-27] Tras una sustitución, la concordancia sigue al alimento NUEVO.

Corpus de baterías (421 planes): 201 pasos en 120 comidas con el género o el número del alimento viejo — «arroz integral…
y escúrrela», «espinacas… hasta que se ablande», «sirve junto al espinacas salteado», «filete de pescado blanco… cocínalos»,
«tuesta maní… hasta que doren; retíralas», y «maní fileteado» en los pasos (el lote 287 sólo lo quitaba de la lista).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import concordancia_sustituto as cs  # noqa: E402
import graph_orchestrator as go  # noqa: E402

_PRECIOS = {"quinoa": 300.0, "arroz integral": 70.0, "arroz blanco": 45.0, "kale": 250.0, "espinaca": 60.0,
            "camaron": 400.0, "filete de pescado": 150.0, "almendra": 450.0, "mani": 120.0}


@pytest.fixture(autouse=True)
def _precios(monkeypatch):
    monkeypatch.setattr(go, "_budget_build_master_price_map", lambda: {"catalogo": 1})

    def _precio(nombre, _mm):
        n = cs._sa(nombre)
        return next((p for k, p in _PRECIOS.items() if k in n), 0.0)

    monkeypatch.setattr(go, "_budget_master_price_per_lb", _precio)


_FORM = {"cookingTime": "1hour", "medicalConditions": ["Ninguna"], "country": "DO", "budget": "low",
         "allergies": [], "dislikes": []}


def _pase(meal):
    dias = [{"day": 1, "meals": [copy.deepcopy(meal)]}]
    assert go._apply_budget_cheapen_pass(dias, _FORM, force=True) >= 1
    return dias[0]["meals"][0]


def test_quinoa_a_arroz_integral_escurrelo_tierno_y_la_linea():
    m = _pase({"meal": "Almuerzo", "name": "Pollo con quinoa", "ingredients": ["150 g de pechuga de pollo",
                                                                               "½ taza de quinoa seca"],
               "recipe": ["Mise en place: enjuaga ½ taza de quinoa seca (80 g) y escúrrela.",
                          "El Toque de Fuego: cocina la quinoa en agua hirviendo durante 12-15 minutos, hasta que esté "
                          "tierna; escúrrela muy bien y tuéstala en una sartén seca, removiendo para que quede suelta."]})
    assert m["ingredients"][1] == "½ taza de Arroz integral seco", m["ingredients"]
    assert m["recipe"][0] == "Mise en place: enjuaga ½ taza de arroz integral seco (80 g) y escúrrelo."
    assert ("hasta que esté tierno; escúrrelo muy bien y tuéstalo en una sartén seca, removiendo para que quede "
            "suelto.") in m["recipe"][1], m["recipe"][1]
    assert "35-45 minutos" in m["recipe"][1]                                   # lote 611


def test_kale_a_espinacas_numero_y_articulo():
    m = _pase({"meal": "Cena", "name": "Pescado con kale", "ingredients": ["150 g de tilapia", "2 tazas de kale"],
               "recipe": ["El Toque de Fuego: saltea el kale 2-3 min hasta que se ablande ligeramente.",
                          "Montaje: sirve la tilapia junto al kale salteado."]})
    assert m["recipe"][0] == "El Toque de Fuego: saltea espinacas 2-3 min hasta que se ablanden ligeramente."
    assert m["recipe"][1] == "Montaje: sirve la tilapia junto a las espinacas salteadas."


def test_camarones_a_filete_singular():
    m = _pase({"meal": "Almuerzo", "name": "Guiso de berenjena con camarones",
               "ingredients": ["1 berenjena", "150 g de camarones"],
               "recipe": ["El Toque de Fuego: añade los camarones al guiso y cocínalos 2-3 minutos, hasta que estén "
                          "opacos; incorpóralos con cuidado."]})
    assert ("añade filete de pescado blanco al guiso y cocínalo 2-3 minutos, hasta que esté opaco; incorpóralo con "
            "cuidado.") in m["recipe"][0], m["recipe"][0]


def test_almendras_a_mani_verbo_pronombre_y_corte():
    m = _pase({"meal": "Merienda", "name": "Yogurt con almendras", "ingredients": ["1 taza de yogurt natural",
                                                                                  "15 g de almendras fileteadas"],
               "recipe": ["El Toque de Fuego: tuesta las almendras fileteadas en una sartén seca hasta que doren y "
                          "suelten aroma; retíralas de inmediato para que no se quemen."]})
    assert m["recipe"][0] == ("El Toque de Fuego: tuesta maní picado en una sartén seca hasta que dore y suelte aroma; "
                              "retíralo de inmediato para que no se queme."), m["recipe"][0]


def test_el_pronombre_de_otro_sustantivo_no_se_toca():
    m = _pase({"meal": "Cena", "name": "Wrap de pollo", "ingredients": ["1 tortilla integral", "15 g de almendras"],
               "recipe": ["Montaje: reparte el pollo entre la tortilla integral, agrega las almendras y ciérralas como "
                          "wraps."]})
    assert m["recipe"][0] == "Montaje: reparte el pollo entre la tortilla integral, agrega maní y ciérralas como wraps."


def test_la_ruta_clinica_tambien_concuerda():
    meal = {"ingredients": ["2 tazas de espinacas"], "recipe": ["Saltea el kale 2 minutos hasta que se ablande."]}
    assert go._rewrite_recipe_steps_after_subs(meal, [(["kale"], "espinacas")])
    assert meal["recipe"] == ["Saltea espinacas 2 minutos hasta que se ablanden."]


@pytest.mark.parametrize("nombre, gn", [("quinoa", ("f", False)), ("Arroz integral", ("m", False)),
                                        ("almendras", ("f", True)), ("nueces", ("f", True)), ("Maní", ("m", False)),
                                        ("camarones", ("m", True)), ("Filete de pescado blanco", ("m", False)),
                                        ("Espinacas", ("f", True)), ("espárragos", ("m", True)),
                                        ("½ taza de quinoa seca", ("f", False)), ("xyz", None)])
def test_genero_numero(nombre, gn):
    assert cs.genero_numero(nombre) == gn


def test_mismo_genero_no_toca_nada():
    t = "cocina la chía hasta que esté hidratada; revuélvela."
    assert cs.concordar(t, "linaza", "chía") == t
