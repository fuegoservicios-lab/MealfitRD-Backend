# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-341 · 2026-09-25] El número y el participio concuerdan en la lista cuando son varios.

Corpus de 308 planes: «1½ ají cubanela», «4 ají cubanela», «2 nabo mediano» (26 líneas) y «2½ tazas de espinacas
picado» (38 líneas) en la lista que lee el usuario."""
from __future__ import annotations

import pulido_lineas as pl


def test_varios_en_plural():
    assert pl.pulir_linea("1½ ají cubanela") == "1½ ajíes cubanela"
    assert pl.pulir_linea("4 ají cubanela") == "4 ajíes cubanela"
    assert pl.pulir_linea("2 nabo mediano") == "2 nabos medianos"
    assert pl.pulir_linea("2½ tazas de espinacas picado") == "2½ tazas de espinacas picadas"
    assert pl.pulir_linea("1 taza de pimientos picado") == "1 taza de pimientos picados"


def test_lo_que_ya_concuerda_no_se_toca():
    for s in ("1 ají cubanela", "½ nabo mediano", "1 taza de espinacas picadas", "2 dientes de ajo",
              "1 taza de hojas de lechuga picada", "3 tazas de lechuga picada"):
        assert pl.pulir_linea(s) == s, s
