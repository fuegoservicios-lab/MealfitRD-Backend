# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-313 · 2026-09-25] «hasta que el cuchillo entre» también dice hasta cuándo.

Plan real del perfil del dueño («Nada» de tiempo), 25-sep: «cocina la auyama en un recipiente apto para microondas con 1
cda de agua durante 5-7 minutos, hasta que el cuchillo entre sin fuerza». V7f exige ≥ 8 min a un víver salvo que el paso
diga hasta cuándo; no reconocía la prueba del cuchillo, daba la auyama por cruda y el reparador del lote 68 añadía
«🍠 Corta Auyama en cubos pequeños (1 cm) y hiérvelos 8-10 minutos» a un plato que ya la cocía."""
from __future__ import annotations

import pathlib

import culinary_coherence as cc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_IDX = cc.build_culinary_index([
    {"name": "Auyama", "aliases": ["calabaza", "auyamas"], "category": "Víveres", "prep_methods": ["hervido", "horneado"]},
    {"name": "Cebolla", "aliases": ["cebollas"], "category": "Vegetales", "prep_methods": ["crudo", "sofrito"]},
])


def _plato(fuego):
    return {"name": "Tortilla con auyama", "ingredients": ["250 g de auyama", "½ cebolla"],
            "recipe": ["Mise en place: corta 250 g de auyama en cubos pequeños y ½ cebolla en plumas.", fuego,
                       "Montaje: reparte la auyama en la tortilla y sirve."]}


def test_la_prueba_del_cuchillo_cuece():
    m = _plato("El Toque de Fuego: cocina la auyama en un recipiente apto para microondas con 1 cda de agua durante 5-7 "
               "minutos, hasta que el cuchillo entre sin fuerza.")
    assert cc.alimentos_sin_coccion(m, _IDX) == []
    m2 = _plato("El Toque de Fuego: cuece la auyama en el microondas 6 minutos, hasta que un tenedor la atraviese.")
    assert cc.alimentos_sin_coccion(m2, _IDX) == []


def test_sin_decir_hasta_cuando_sigue_sin_cocer():
    m = _plato("El Toque de Fuego: dora la auyama 3 minutos en la sartén con la cebolla.")
    assert cc.alimentos_sin_coccion(m, _IDX) == [("Auyama", "viver")]


def test_ancla():
    assert "tooltip-anchor: P1-PLAN-LOTE-313-PRUEBA-DEL-CUCHILLO" in (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
