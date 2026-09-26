# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-344 · 2026-09-26] Una tostada no se cierra.

Batería real sobre el 331 (bariátrica, día 1): «Tostada Integral con Hummus y Tomate» recibía un SEGUNDO Montaje —
«Montaje: rellena la pan integral 30 g con lo que cierra (unos 110 g del relleno) y sirve el resto del relleno (~59 g)
al lado, como ensalada: misma compra, un wrap que se puede cerrar»—: el detector del lote 27 trata igual tostadas y wraps,
y una tostada es un plato ABIERTO."""
from __future__ import annotations

import dish_structure as ds


def test_la_tostada_abierta_no_es_un_wrap():
    meal = {"name": "Tostada Integral con Hummus y Tomate",
            "ingredients": ["1 rebanada de pan integral (30 g)", "20 g de hummus", "1½ tomates (147 g)",
                            "1 cdta de aceite de oliva virgen extra"],
            "recipe": ["Mise en place: mide 20 g de hummus y corta 1½ tomates (147 g) en rodajas finas.",
                       "Montaje: unta el hummus sobre la rebanada de pan integral y coloca encima el tomate."]}
    assert not any(r["tipo"] == "wrap_desproporcionado" for r in ds.relaciones(meal))


def test_el_wrap_de_verdad_sigue_acusado():
    meal = {"name": "Wrap de pollo con lechuga",
            "ingredients": ["1 tortilla de trigo (40 g)", "200 g de pechuga de pollo", "100 g de lechuga"],
            "recipe": ["Montaje: rellena la tortilla con el pollo y la lechuga."]}
    assert any(r["tipo"] == "wrap_desproporcionado" for r in ds.relaciones(meal))
