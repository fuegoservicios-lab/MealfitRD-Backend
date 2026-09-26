# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-351 · 2026-09-26] Dos quesos en la lista: la mención se ata por sus dos primeras palabras.

Plan del dueño (batería real sobre el 331, día 1): «20 g de queso blanco fresco» y «70 g de queso mozzarella» en la lista y
el paso «ten listos 15 g de queso blanco fresco» — el sincronizador descartaba las dos líneas por empezar igual."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_cada_queso_toma_los_gramos_de_su_linea():
    m = {"ingredients": ["20 g de queso blanco fresco", "70 g de queso mozzarella", "½ torta de casabe"],
         "recipe": ["Mise en place: ten listos 15 g de queso blanco fresco y ½ torta de casabe.",
                    "Montaje: sirve con 50 g de queso mozzarella rallado."]}
    assert pc.gramos_por_dos_palabras(m) == 2
    assert "ten listos 20 g de queso blanco fresco" in m["recipe"][0]
    assert "sirve con 70 g de queso mozzarella" in m["recipe"][1]


def test_sin_ambiguedad_o_con_reparto_no_se_toca():
    pasos = ["Mise en place: ten 15 g de queso blanco fresco."]
    m = {"ingredients": ["20 g de queso blanco fresco", "150 g de pechuga de pollo"], "recipe": list(pasos)}
    assert pc.gramos_por_dos_palabras(m) == 0 and m["recipe"] == pasos      # sin ambigüedad: es del sincronizador
    pasos2 = ["Mise en place: reserva la mitad, 10 g de queso blanco fresco, para el final.", "Montaje: sirve."]
    m2 = {"ingredients": ["20 g de queso blanco fresco", "70 g de queso mozzarella"], "recipe": list(pasos2)}
    assert pc.gramos_por_dos_palabras(m2) == 0 and m2["recipe"] == pasos2


def test_ancla():
    assert "+ gramos_por_dos_palabras(meal)" in (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
