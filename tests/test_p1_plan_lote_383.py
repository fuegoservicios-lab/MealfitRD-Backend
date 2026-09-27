# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-383 · 2026-09-26] Lo que «Descríbelo» separa en gramos también baja de la Nevera.

La auditoría de «Registrar comida»: en el componedor, las partes de «Descríbelo y lo calculo» («Lechosa · 150 g»,
«Leche entera · 240 g») viajaban como `custom` y un `custom` NUNCA descontaba (`pantry_lines: []`), aunque el
interruptor dijera «Resta estos alimentos». En el escáner, la MISMA respuesta del servidor sí descontaba (va como
«150 g de Lechosa» y la Nevera la resuelve por nombre). Ahora un `custom` CON GRAMOS viaja igual que en el escáner;
sin gramos («Macros a mano», un plato entero) sigue sin descontar: no hay cantidad que restar.
Tooltip-anchor: P1-PLAN-LOTE-383
"""
from __future__ import annotations


def test_un_custom_con_gramos_descuenta_como_en_el_escaner():
    import food_search
    r = food_search.resolve_line({"ref": "custom", "qty": 1, "unit": "g", "name": "Lechosa", "grams": 150,
                                  "macros": {"kcal": 60}}, [])
    assert r["pantry_lines"] == ["150 g de Lechosa"]
    assert r["grams"] == 150


def test_sin_gramos_no_hay_nada_que_restar():
    import food_search
    r = food_search.resolve_line({"ref": "custom", "qty": 1, "unit": "g", "name": "Batida de lechosa",
                                  "macros": {"kcal": 280}}, [])
    assert r["pantry_lines"] == []
    for malo in (0, -5, 99999, "x"):
        r = food_search.resolve_line({"ref": "custom", "qty": 1, "name": "Lechosa", "grams": malo, "macros": {}}, [])
        assert r["pantry_lines"] == []


def test_la_linea_del_componedor_acepta_gramos():
    from routers.diary import ManualMealLine
    assert ManualMealLine(ref="custom", name="Lechosa", grams=150).grams == 150
    assert ManualMealLine(ref="custom", name="Lechosa").grams is None
