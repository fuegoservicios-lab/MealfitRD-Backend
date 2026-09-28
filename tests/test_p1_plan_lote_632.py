# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-632 · 2026-09-28] Un participio («las lentejas guisadas») no cuece la verdura de al lado.

Validación del 592 (estudiante, día 3): «Lentejas guisadas… con plátano maduro, brócoli al vapor y pechuga de pollo»
—el Mise separa el brócoli, ningún paso lo cocina y el Montaje lo sirve «al lado»— quedaba sin la «💡 Cocción previa»
del lote 540 porque la frase del Montaje decía «las lentejas guisadas». Replay de 5.042 comidas: 5 cambios, los 5
verduras que ningún paso cocinaba (brócoli, vainitas, tayota, berenjena, espinacas).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import verdura_sin_coccion as vsc  # noqa: E402

_LENTEJAS = {
    "name": "Lentejas guisadas en salsa natural de vegetales con plátano maduro, brócoli al vapor y pechuga de pollo",
    "ingredients": ["⅔ taza de lentejas cocidas", "½ plátano maduro", "1⅓ tazas de brócoli en floretes",
                    "½ tomate mediano", "½ cebolla", "½ diente de ajo", "1¼ cdtas de aceite de oliva",
                    "⅓ taza de agua", "Sal al gusto", "1¼ pechugas de pollo (≈208 g)"],
    "recipe": [
        "Mise en place: enjuaga y escurre ⅔ taza de lentejas cocidas; corta ½ plátano maduro (75 g) en trozos, separa "
        "1⅓ tazas de brócoli en floretes, pica ½ tomate mediano, ½ cebolla y ½ diente de ajo.",
        "El Toque de Fuego: calienta el aceite de oliva en una sartén profunda a fuego medio. Sofríe la cebolla y el ajo "
        "2 minutos, añade el tomate y cocina 3 minutos. Calienta las lentejas cocidas 2-3 minutos. Añade pechuga de "
        "pollo al guiso y cocínala a fuego medio 12-15 minutos, hasta que esté cocida por dentro.",
        "🍠 Añade plátano maduro al guiso y cocínalo 15-20 minutos, hasta que esté tierno por dentro, antes de servir.",
        "Montaje: sirve las lentejas guisadas con el plátano maduro en salsa y el brócoli al lado; toma agua con la "
        "comida. Acompaña con pechuga de pollo.",
    ],
}
_NOTA_BROCOLI = "💡 Cocción previa: hierve el brócoli 4-5 minutos (o cocínalo al vapor), hasta que esté tierno, y escúrrelo."


def test_el_adjetivo_de_otro_alimento_no_cuece_el_brocoli():
    m = copy.deepcopy(_LENTEJAS)
    assert vsc.cocer(m) == 1
    assert m["recipe"][1] == _NOTA_BROCOLI            # tras el Mise en place
    assert vsc.cocer(m) == 0                           # idempotente: la nota lo cuece


@pytest.mark.parametrize("montaje", [
    "Montaje: sirve las tortitas horneadas con el brócoli al lado.",
    "Montaje: sirve el pollo asado con el brócoli al vapor y agua.",
    "Montaje: sirve caliente con el brócoli.",
])
def test_participio_adjetivo_o_eco_del_nombre_no_cuecen(montaje):
    m = {"name": "Pollo con brócoli al vapor", "ingredients": ["1 taza de brócoli", "150 g de pechuga de pollo"],
         "recipe": ["Mise en place: separa 1 taza de brócoli en floretes.",
                    "El Toque de Fuego: cocina la pechuga a la plancha 6-7 minutos por lado.", montaje]}
    assert vsc.cocer(m) == 1 and _NOTA_BROCOLI in m["recipe"]


@pytest.mark.parametrize("toque", [
    "El Toque de Fuego: saltea el brócoli 3 minutos y cocina la pechuga a la plancha.",
    "El Toque de Fuego: pon el brócoli al vapor hasta que esté tierno; cocina la pechuga a la plancha.",
    "El Toque de Fuego: cocina la pechuga a la plancha. Añade el brócoli al guiso y cocina 4 minutos.",
])
def test_un_verbo_de_verdad_sigue_contando(toque):
    m = {"name": "Pollo con brócoli", "ingredients": ["1 taza de brócoli", "150 g de pechuga de pollo"],
         "recipe": ["Mise en place: separa 1 taza de brócoli en floretes.", toque,
                    "Montaje: sirve la pechuga asada con el brócoli."]}
    antes = list(m["recipe"])
    assert vsc.cocer(m) == 0 and m["recipe"] == antes
