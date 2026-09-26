# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-407 · 2026-09-26] La proteína que la lista compra cruda y el paso usa cocida trae su cocción.

Batería real sobre el 331 (bariátrica, día 1): «ten lista la pechuga de pollo cocida y desmenuzada» con «¼ pechuga de pollo
(≈63 g)» cruda en la lista y ningún paso que la cocine (corpus: 52 comidas)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_pollo_crudo_usado_cocido_trae_su_coccion():
    m = {"ingredients": ["¼ pechuga de pollo (≈63 g)", "¼ plátano verde", "125 g de vainitas"],
         "recipe": ["Mise en place: pica la cebolla; ten lista la pechuga de pollo cocida y desmenuzada.",
                    "El Toque de Fuego: sofríe la cebolla 3 minutos y agrega el pollo desmenuzado con el orégano.",
                    "Montaje: sirve."]}
    assert pc.proteina_cocida_de_la_lista(m) == 1
    assert m["recipe"][1] == ("💡 Cocción previa: cocina la pechuga de pollo en agua con sal 15-18 min, o a la plancha 5-7 min "
                              "por lado, hasta que no quede rosada por dentro (74 °C al centro); déjala reposar y "
                              "desmenúzala o córtala como pide la receta."), m["recipe"][1]
    assert pc.proteina_cocida_de_la_lista(m) == 0                                   # idempotente
    p = {"ingredients": ["1 filete de pescado"],
         "recipe": ["Mise en place: añade filete de pescado blanco ya cocido al bowl con el limón.", "Montaje: sirve."]}
    assert pc.proteina_cocida_de_la_lista(p) == 1
    assert p["recipe"][1].startswith("💡 Cocción previa: cocina el filete de pescado a la plancha o al vapor 3-4 min")


def test_lo_que_ya_se_cocina_o_se_compra_cocido_no_se_toca():
    casos = [
        (["1½ pechugas de pollo (255 g)"], ["El Toque de Fuego: sella la pechuga de pollo 5-6 min por lado; desmenuza el "
                                            "pollo cocido."]),
        (["150 g de pechuga de pollo cocida"], ["Mise en place: desmenuza la pechuga de pollo cocida."]),
        (["65 g de atún en agua"], ["Mise en place: agrega el atún ya cocido."]),
        (["1 pechuga de pollo"], ["El Toque de Fuego: cocina el pollo hasta que esté completamente cocido."]),
    ]
    for lista, pasos in casos:
        m = {"ingredients": list(lista), "recipe": list(pasos)}
        assert pc.proteina_cocida_de_la_lista(m) == 0 and m["recipe"] == pasos, (lista, m["recipe"])


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").proteina_cocida_de_la_lista(meal)  # [P1-PLAN-LOTE-407]' in src
