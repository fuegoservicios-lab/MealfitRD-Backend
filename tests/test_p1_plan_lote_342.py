# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-342 · 2026-09-25] En unas «tortitas» el huevo también es aglutinante.

Batería real sobre el 331 (suplementos + estatina, día 2): «Tortitas crujientes de cebada con queso blanco fresco» — las
claras ligan la masa, pero «tortitas» no contiene «torta», así que el autocorrector de proteína repetida
(P1-PROTEIN-REPEAT-AUTOFIX) las tomó por un segundo plato-huevo del día y las cambió por pechuga, también en los pasos y
en las notas: «bate ¾ pechuga de pollo», «pechuga de pollo batidas», «usa solo pechuga de pollo — NO botes pechuga de
pollo: guárdalas tapadas»."""
from __future__ import annotations

import graph_orchestrator as go


def test_las_tortitas_son_aglutinante_para_el_gate_y_el_autocorrector():
    for nombre in ("tortitas crujientes de cebada con queso blanco fresco", "tortitas de avena con guineo",
                   "torticas de yuca"):
        assert go._egg_counts_for_same_day_gate(nombre) is False, nombre
    assert go._egg_counts_for_same_day_gate("revoltillo de claras con cebolla") is True
    assert go._egg_counts_for_same_day_gate("tortitas con huevo frito") is True      # el modo de cocción del huevo manda


def test_ancla():
    assert "tortita" in go._EGG_BINDER_DISH_TOKENS and "tortica" in go._EGG_BINDER_DISH_TOKENS
