# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-586 · 2026-09-27] La proteína que el plato ya cuece no se cuece otra vez.

Batería real (DM2 + HTA; bariátrica + SOP): «…tapa y guisa 8-10 min, hasta que el pescado alcance 63 °C. Añade tilapia
fresca al guiso y cocínala…» y «…hornea a 180 °C 10-12 minutos, hasta que la clara esté cuajada y la yema firme… Cocina
huevo a la plancha o hervido y sírvelo como proteína del plato.»
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import coccion_sin_repetir as csr  # noqa: E402

_MISE = "Mise en place: corta los vegetales."


def _meal(toque, montaje="Montaje: sirve caliente."):
    return {"recipe": [_MISE, toque, montaje]}


@pytest.mark.parametrize("toque, queda", [
    ("El Toque de Fuego: cocina la cebolla 2 min. Añade la tilapia, el vinagre blanco y el cilantro; tapa y guisa a fuego "
     "medio-bajo 8-10 min, hasta que el pescado alcance 63 °C en el centro. Añade tilapia fresca al guiso y cocínala a "
     "fuego medio 5-7 minutos, hasta que se desmenuce fácilmente (63 °C al centro); incorpórala con cuidado para no "
     "deshacer el resto.",
     "hasta que el pescado alcance 63 °C en el centro."),
    ("El Toque de Fuego: Casca 1 huevo dentro del molde engrasado sobre el aceite, sala al gusto; hornea a 180 °C durante "
     "10-12 minutos, hasta que la clara esté completamente cuajada y la yema firme (nunca líquida). Aparte, tuesta el pan "
     "2 minutos por lado. Cocina huevo a la plancha o hervido y sírvelo como proteína del plato.",
     "tuesta el pan 2 minutos por lado."),
])
def test_la_segunda_coccion_se_va(toque, queda):
    m = _meal(toque)
    assert csr.quitar(m) == 1
    assert m["recipe"][1].endswith(queda), m["recipe"][1]


def test_la_copia_exacta_se_va():
    frase = "Cocina Filete de pescado blanco a la plancha o hervido y sírvelo como proteína del plato."
    m = _meal(f"El Toque de Fuego: dora el queso 1-2 min por lado. {frase} {frase}")
    assert csr.quitar(m) == 1 and m["recipe"][1].count(frase) == 1


@pytest.mark.parametrize("toque, montaje", [
    # «incorpora también» en la misma oración que otra cocción NO cuece el pavo (el pavo se quedaba crudo)
    ("El Toque de Fuego: sofríe la cebolla y el ajo 2 min, añade el tomate y cocina 3 min; Incorpora también pechuga de "
     "pavo durante la preparación. Añade Pechuga de pavo al guiso y cocínala a fuego medio 12-15 minutos, hasta que esté "
     "cocida por dentro; incorpórala con cuidado para no deshacer el resto.", "Montaje: sirve caliente."),
    # batir los huevos y luego cocinar la cebolla no cuece los huevos
    ("El Toque de Fuego: bate los huevos con la pimienta; cocina la cebolla y el ajo durante 1 minuto. Cocina huevos a la "
     "plancha o hervidos y sírvelos como proteína del plato.", "Montaje: sirve caliente."),
    # «a la plancha» en el Montaje no es cocinarlo
    ("El Toque de Fuego: hierve el plátano 12-15 min. Cocina Filete de pescado blanco a la plancha o hervido y sírvelo "
     "como proteína del plato.", "Montaje: sirve filete de pescado blanco a la plancha sobre el plátano."),
    # «apto para consumo crudo»: «para» no es el alimento
    ("El Toque de Fuego: cocina el arroz en un recipiente para microondas 10 min. Cocina salmón apto para consumo crudo a "
     "la plancha o hervido y sírvelo como proteína del plato.", "Montaje: sirve."),
])
def test_si_nadie_mas_la_cuece_la_frase_se_queda(toque, montaje):
    m = _meal(toque, montaje)
    antes = list(m["recipe"])
    assert csr.quitar(m) == 0 and m["recipe"] == antes


def test_ancla_en_la_cola():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("coccion_sin_repetir").quitar(meal, index)  # [P1-PLAN-LOTE-586]' in src
