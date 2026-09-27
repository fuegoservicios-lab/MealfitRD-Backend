# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-442 · 2026-09-26] Lo fresco no se escurre: se seca.

Replay de la cola sobre 322 planes: «escurre ½ pechuga de pollo (≈100 g)», «escurre 150 g de filete de pescado blanco» (7
comidas): el atún o las sardinas de lata pasaron a pollo o pescado fresco y el verbo de la lata quedó."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_pieza_fresca_se_seca():
    m = {"ingredients": ["1 filete de pescado (≈150 g)", "½ pechuga de pollo (≈100 g)"],
         "recipe": ["Mise en place: escurre 150 g de filete de pescado blanco y mide el aceite; escurre ½ pechuga de pollo "
                    "(≈100 g)."]}
    assert pc.fresco_no_se_escurre(m) == 1
    assert m["recipe"][0] == ("Mise en place: seca 150 g de filete de pescado blanco con papel de cocina y mide el aceite; "
                              "seca ½ pechuga de pollo (≈100 g) con papel de cocina.")
    assert pc.fresco_no_se_escurre(m) == 0


def test_la_lata_se_sigue_escurriendo():
    pasos = ["Mise en place: escurre 150 g de atún en agua y escurre 2 filetes de pescado en lata."]
    m = {"ingredients": ["150 g de atún en agua", "2 filetes de pescado en lata"], "recipe": list(pasos)}
    assert pc.fresco_no_se_escurre(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").fresco_no_se_escurre(meal)  # [P1-PLAN-LOTE-442]' in src
