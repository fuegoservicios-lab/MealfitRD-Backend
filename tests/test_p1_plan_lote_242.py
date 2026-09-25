# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-242 · 2026-09-25] El tope de yemas deja UNA línea de claras por plato, con su número gramatical, y no
cuenta dos veces las claras en la compra.

Batería real con el 235 (colesterol + ganar músculo): «1 huevo · 1 claras de huevo · 3 claras de huevo» en el mismo plato.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import yemas_colesterol as yc  # noqa: E402

FORM = {"medicalConditions": ["Colesterol Alto"]}


def _plan(ings, raw=None):
    meal = {"meal": "Desayuno", "name": "Revoltillo", "ingredients": list(ings), "recipe": ["Mise en place: bate."]}
    if raw is not None:
        meal["ingredients_raw"] = list(raw)
    return {"days": [{"day": 1, "meals": [meal]}]}


def test_una_sola_linea_de_claras_y_en_plural():
    p = _plan(["2 huevos", "3 claras de huevo", "1 taza de espinaca"])
    yc.topar_yemas(p, FORM)
    ings = p["days"][0]["meals"][0]["ingredients"]
    assert [i for i in ings if "clara" in i] == ["5 claras de huevo"], ings
    assert "1 huevo" in ings


def test_una_clara_en_singular():
    p = _plan(["1½ huevos"])
    yc.topar_yemas(p, FORM)
    ings = p["days"][0]["meals"][0]["ingredients"]
    assert ings == ["1 huevo", "1 clara de huevo"], ings


def test_la_compra_no_cuenta_dos_veces_las_claras():
    p = _plan(["2 huevos", "3 claras de huevo"], raw=["100g de huevo", "99g de clara de huevo"])
    yc.topar_yemas(p, FORM)
    raw = p["days"][0]["meals"][0]["ingredients_raw"]
    assert sorted(raw) == ["165g de clara de huevo", "50g de huevo"], raw


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 242
