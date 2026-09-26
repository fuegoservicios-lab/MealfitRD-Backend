# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-371 · 2026-09-26] «los 40 g», no «las 40 g»: el artículo concuerda con los gramos.

Replay de la cola real: «escurre las 40 g de habichuelas negras cocidas», «ten listas las 135 g de habichuelas rojas
cocidas», «corta la 20 g de queso blanco fresco» — el sincronizador mete la cifra detrás del artículo del alimento."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_articulo_concuerda():
    m = {"recipe": ["Mise en place: escurre las 40 g de habichuelas negras cocidas; ten listas las 135 g de habichuelas rojas.",
                    "El Toque de Fuego: corta la 20 g de queso blanco fresco y tuesta las 15 g 3-4 minutos.",
                    "⚠️ Nota: las 40 g de habichuelas se remojan."]}
    assert pc.articulo_de_los_gramos(m) == 2
    assert m["recipe"][0] == ("Mise en place: escurre los 40 g de habichuelas negras cocidas; ten listos los 135 g de "
                              "habichuelas rojas."), m["recipe"][0]
    assert m["recipe"][1] == "El Toque de Fuego: corta los 20 g de queso blanco fresco y tuesta los 15 g 3-4 minutos."
    assert m["recipe"][2] == "⚠️ Nota: las 40 g de habichuelas se remojan."                    # las notas no se tocan


def test_lo_que_no_se_toca():
    pasos = ["Mise en place: corta las habichuelas y la cebolla; mide 1 taza de arroz y las 2 tazas de agua."]
    m = {"recipe": list(pasos)}
    assert pc.articulo_de_los_gramos(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").articulo_de_los_gramos(meal)  # [P1-PLAN-LOTE-371]' in src
