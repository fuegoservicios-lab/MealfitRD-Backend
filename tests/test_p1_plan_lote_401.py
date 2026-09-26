# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-401 · 2026-09-26] La frase del guiso concuerda.

Batería REAL sobre el 379 (adulto mayor con HTA, día 2): «Agrega yautía al guiso y cocínalos a fuego medio 12-15 minutos,
hasta que esté cocidos por dentro; Incorpóralos con cuidado» (corpus: «esté cocidos» 18, «; Incorpóralo» 54)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_PASO = ("El Toque de Fuego: cocina la cebolla 3 min. Agrega yautía al guiso y cocínalos a fuego medio 12-15 minutos, hasta "
         "que esté cocidos por dentro; Incorpóralos con cuidado para no deshacer el resto. Añade filete de pescado blanco al "
         "guiso y cocínalo a fuego medio 12-15 minutos, hasta que esté cocido por dentro; Incorpóralo con cuidado para no "
         "deshacer el resto.")


def test_la_frase_del_guiso_concuerda():
    m = {"recipe": [_PASO, "⚠️ Nota: hasta que esté cocidos; Incorpóralos (las notas no se tocan)."]}
    assert pc.guiso_concuerda(m) == 1
    assert "hasta que estén cocidos por dentro; incorpóralos con cuidado" in m["recipe"][0]
    assert "hasta que esté cocido por dentro; incorpóralo con cuidado" in m["recipe"][0]
    assert m["recipe"][1] == "⚠️ Nota: hasta que esté cocidos; Incorpóralos (las notas no se tocan)."
    assert pc.guiso_concuerda(m) == 0


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").guiso_concuerda(meal)  # [P1-PLAN-LOTE-401]' in src
