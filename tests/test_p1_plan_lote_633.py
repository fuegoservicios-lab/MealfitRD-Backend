# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-633 · 2026-09-28] El víver que el complemento incorpora llega cocido.

Validación del 592 (adulto mayor con hipertensión, día 2): «Tortitas saladas de trigo… ensalada fresca y batata al
vapor» terminaba el Toque con «Incorpora también batata durante la preparación» y ningún paso la cocía.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

_TORTITAS = {
    "name": "Tortitas saladas de trigo con revuelto de huevo y palmito, ensalada fresca y batata al vapor",
    "ingredients": ["25 g de harina de trigo", "105 ml de agua", "6 claras de huevo", "55 g de palmito bajo en sodio",
                    "1 cda de aceite de oliva", "½ tomate", "½ pepino", "½ taza de lechuga", "½ limón",
                    "1 diente de ajo", "½ batata mediana"],
    "recipe": [
        "Mise en place: mide 25 g de harina de trigo y 105 ml de agua; pica 1 diente de ajo; corta 55 g de palmito en "
        "rodajas, ½ tomate y ½ pepino; lava y corta ½ taza de lechuga; exprime ½ limón; saca 6 claras de huevo.",
        "El Toque de Fuego: mezcla la harina de trigo con el agua y el ajo hasta lograr una masa fluida. Calienta la "
        "sartén con el aceite de oliva y cocina tortitas finas 2-3 minutos por lado. En la misma sartén, revuelve las "
        "claras con el palmito 3-4 minutos hasta que cuajen.",
        "Montaje: sirve las tortitas junto al revuelto y la ensalada de tomate, pepino y lechuga con jugo de limón.",
    ],
}


def _pasos(m):
    return [s for s in m["recipe"] if isinstance(s, str)]


def test_la_batata_prometida_al_vapor_se_cuece_al_vapor():
    m = copy.deepcopy(_TORTITAS)
    assert go._ensure_ingredients_used_in_recipe(m) >= 1
    assert _pasos(m)[1] == ("💡 Cocción previa: cocina la batata pelada al vapor 15-20 min, hasta que el cuchillo "
                            "entre sin fuerza.")
    assert go._ensure_ingredients_used_in_recipe(m) == 0          # idempotente: ya la usa un paso


@pytest.mark.parametrize("linea, nota", [
    ("½ batata mediana", "💡 Cocción previa: hierve la batata pelada en agua 15-20 min, hasta que el cuchillo entre sin "
                         "fuerza."),
    ("1 papa pequeña", "💡 Cocción previa: hierve la papa pelada en agua 15-20 min, hasta que el cuchillo entre sin "
                       "fuerza."),
    ("½ plátano verde", "💡 Cocción previa: hierve el plátano verde pelado en agua 20-25 min, hasta que el cuchillo "
                        "entre sin fuerza."),
])
def test_sin_vapor_en_el_nombre_se_hierve(linea, nota):
    m = copy.deepcopy(_TORTITAS)
    m["name"] = "Tortitas saladas de trigo con revuelto de huevo y palmito"
    m["ingredients"][-1] = linea
    go._ensure_ingredients_used_in_recipe(m)
    assert _pasos(m)[1] == nota


@pytest.mark.parametrize("linea", ["½ batata asada", "1 guineo maduro", "20 g de harina de plátano", "½ taza de puré de papa"])
def test_lo_que_ya_viene_cocido_o_se_come_crudo_no_recibe_nota(linea):
    m = copy.deepcopy(_TORTITAS)
    m["ingredients"][-1] = linea
    go._ensure_ingredients_used_in_recipe(m)
    assert not any("Cocción previa" in s for s in _pasos(m))
