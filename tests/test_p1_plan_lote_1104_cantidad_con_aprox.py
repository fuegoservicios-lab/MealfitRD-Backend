"""[P1-PLAN-LOTE-1104 · 2026-11-18] «Jugo de naranja Rica 100% sin azúcar añadida (2 vasos, ≈500 ml)» no tenía −/+.

Al duplicar la cantidad quedaba «4 vasos, ≈500 ml»: el catálogo leía los mismos 500 ml (la misma energía) y la regla «al
duplicarla, el doble de energía» lo declaraba no editable. El «≈N ml» acompaña a la cantidad y se escala con ella.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gramos_editables as g  # noqa: E402

JUGO = "Jugo de naranja Rica 100% sin azúcar añadida (2 vasos, ≈500 ml)"


def test_el_aprox_se_escala_con_la_cantidad():
    assert g.con_cantidad(JUGO, 4) == "Jugo de naranja Rica 100% sin azúcar añadida (4 vasos, ≈1000 ml)"
    assert g.con_cantidad(JUGO, 1) == "Jugo de naranja Rica 100% sin azúcar añadida (1 vasos, ≈250 ml)"
    assert g.con_cantidad(JUGO, 2) == JUGO


def test_sin_aprox_o_con_el_aprox_como_cantidad_no_cambia_nada_mas():
    assert g.con_cantidad("2 tazas de arroz", 3) == "3 tazas de arroz"
    assert g.con_cantidad("Leche (≈1 vaso)", 2) == "Leche (≈2 vaso)"
    assert g.con_cantidad("Té (1 taza) con 2 g de azúcar", 2).startswith("Té (2 taza)")


def test_lo_que_va_fuera_del_parentesis_no_se_toca():
    t = "Batido (1 vaso, ≈300 ml) y 200 ml de agua aparte"
    assert g.con_cantidad(t, 2) == "Batido (2 vaso, ≈600 ml) y 200 ml de agua aparte"


def test_con_el_aprox_escalado_el_jugo_ya_es_editable():
    class Db:
        pass

    def macros(texto, db):
        import re
        m = re.search(r"≈(\d+) ml", texto)
        return {"kcal": int(m.group(1)) * 0.39} if m else None

    orig = g.macros_de_renglon
    g.macros_de_renglon = macros
    try:
        assert g.cantidades([JUGO], Db()) == {0: {"valor": 2.0, "unidad": "vasos"}}
    finally:
        g.macros_de_renglon = orig
