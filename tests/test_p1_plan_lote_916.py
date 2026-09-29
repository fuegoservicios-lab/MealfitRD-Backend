# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-916 · 2026-09-29] Lo «opcional» que se pesa conserva su cifra.

La línea sin número que el 880-887 dejó abierta («G de queso blanco fresco (extensor opcional)», batería rdb524): el
redondeo de cantidades (`quantize_ingredient_string`) trata «al gusto» y «opcional» como cantidad espuria y quita el
NÚMERO, pero deja la unidad: «30 g de queso blanco fresco (extensor opcional)» → «G de queso…», en la lista visible y en la
del motor (la compra y los macros perdían el queso). La regla nació para «0.91 sal y pimienta al gusto», que no lleva
unidad. Con gramos, mililitros, tazas o cucharas la cantidad es real y se redondea como cualquier otra.
"""
from __future__ import annotations

import pathlib

from nutrition_db import quantize_ingredient_string as q

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_queso_opcional_que_se_pesa_conserva_sus_gramos():
    assert q("30 g de queso blanco fresco (extensor opcional)") == ("30 g de queso blanco fresco (extensor opcional)", 1.0)


def test_y_se_redondea_como_cualquier_otra_linea_en_gramos():
    linea, factor = q("31.72 g de queso blanco fresco (extensor opcional)")
    assert linea == "30 g de queso blanco fresco (extensor opcional)"
    assert abs(factor - 30 / 31.72) < 1e-6


def test_con_cualquier_unidad_de_medida():
    assert q("5 g de sal al gusto")[0] == "5 g de sal al gusto"
    assert q("10 ml de vinagre, opcional")[0] == "10 ml de vinagre, opcional"
    assert q("1 cdta de orégano al gusto")[0] == "1 cdta de orégano al gusto"
    assert q("2 cdas de cilantro picado (opcional)")[0] == "2 cdas de cilantro picado (opcional)"
    assert q("½ taza de lechuga (opcional)")[0] == "½ taza de lechuga (opcional)"


def test_nunca_sale_una_linea_que_empieza_por_la_unidad():
    for linea in ("30 g de queso blanco fresco (extensor opcional)", "15 g de maní tostado sin sal (opcional)",
                  "0.4 g de pimienta al gusto", "120 ml de leche (opcional)", "1.3 tazas de berro (opcional)"):
        salida = q(linea)[0]
        assert salida.split()[0].lower() not in ("g", "gr", "ml", "kg", "taza", "tazas", "cda", "cdas", "cdta", "cdtas"), salida


def test_la_cantidad_sin_unidad_sigue_siendo_espuria():
    """Lo que la regla ya hacía (test_p3_portion_quantize): un número suelto ante «al gusto» no es una medida."""
    assert q("0.91 sal y pimienta al gusto") == ("Sal y pimienta al gusto", 1.0)
    assert q("1 sal al gusto") == ("Sal al gusto", 1.0)
    assert q("Sal al gusto") == ("Sal al gusto", 1.0)
    assert q("1 pizca de sal al gusto")[0] == "Pizca de sal al gusto"


def test_ancla():
    src = (_BACKEND / "nutrition_db.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-916" in src
    assert b"\x08" not in (_BACKEND / "nutrition_db.py").read_bytes()
