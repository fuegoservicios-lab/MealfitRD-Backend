# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-491 · 2026-09-27] Un caldo de pescado no es pescado.

Replay forzado de los días 21+ (perfil DM2 con insulina, «Sardinas estilo ropa vieja»): la línea «½ taza de caldo de
pescado o agua» casaba «pescado» y se sustituía por «½ taza de atún en agua»; el paso quedaba «incorpora las sardinas y
el caldo… hasta que el atún se impregne». El caldo (cubito, tetrabrik) es despensa; lo mismo una crema de leche o un
jugo: productos hechos DEL alimento, no el fresco.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402

_REQ = {"need_days": 21, "allow_frozen": False}


def test_el_caldo_no_se_sustituye():
    for linea in ("½ taza de caldo de pescado o agua", "1 taza de caldo de pollo", "½ taza de crema de leche",
                  "1 taza de consomé de res"):
        assert cu.sustituir_linea(linea, 20, _REQ, gramos_de=lambda _t: None) is None, linea


def test_el_alimento_fresco_si():
    r = cu.sustituir_linea("150 g de filete de pescado", 20, _REQ, gramos_de=lambda _t: None)
    assert r and r[1] in ("atun en agua", "sardinas en lata"), r
    r = cu.sustituir_linea("1 taza de leche", 20, _REQ, gramos_de=lambda _t: None)
    assert r and r[1] == "leche UHT", r
