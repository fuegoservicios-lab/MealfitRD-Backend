# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-880 · 2026-09-29] «sirve la papa majadas» → «sirve la papa majada»: el participio sigue al alimento.

Batería real (mujer que pierde grasa, 29-sep, código 868). Corpus: 15 de 8.392 comidas únicas («la papa escurridas /
doradas / horneadas», «el plátano asados», «el huevo cuajados»), leídas.
"""
from __future__ import annotations

import pathlib

import participio_concuerda as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_participio_pegado_toma_el_numero_y_genero_del_alimento():
    casos = {
        "Montaje: sirve la papa majadas con el revoltillo encima.": "Montaje: sirve la papa majada con el revoltillo encima.",
        "Montaje: reparte la papa doradas y el queso blanco sobre las tortillas.":
            "Montaje: reparte la papa dorada y el queso blanco sobre las tortillas.",
        "Montaje: sirve la ensalada junto a el plátano asados.": "Montaje: sirve la ensalada junto a el plátano asado.",
        "El Toque de Fuego: incorpora la papa escurridas y saltea 1-2 minutos más.":
            "El Toque de Fuego: incorpora la papa escurrida y saltea 1-2 minutos más.",
    }
    for antes, despues in casos.items():
        assert pc.concordar(antes) == despues, pc.concordar(antes)


def test_la_pareja_y_lo_que_no_es_alimento_no_se_tocan():
    for t in ("Montaje: sirve la ensalada con el pepino y el tomate aliñados.",
              "El Toque de Fuego: sofríe el ajo, la cebolla y el ají salteados 3 minutos.",
              "Montaje: sirve 2-3 arepitas en el plato acompañadas de la ensalada fresca.",
              "Montaje: sirve la mezcla batidos.",
              "Montaje: sirve los plátanos majados."):
        assert pc.concordar(t) == t, t


def test_notas_no_y_ancla():
    m = {"recipe": ["⚠️ Seguridad alimentaria: la papa cocidas.", "Montaje: sirve la papa majadas."]}
    assert pc.concordar_pasos(m) == 1 and m["recipe"][0] == "⚠️ Seguridad alimentaria: la papa cocidas."
    assert pc.concordar_pasos(m) == 0
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("participio_concuerda").concordar_pasos(meal)  # [P1-PLAN-LOTE-880]' in src
