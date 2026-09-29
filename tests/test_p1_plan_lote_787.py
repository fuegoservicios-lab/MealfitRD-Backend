# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-787 · 2026-09-28] Los gramos del paso siguen a la pieza de la lista aunque empiece por su unidad.

Bloque 3 real del plan 6594aae1: «½ pedazo mediano de yuca (≈172 g)» con «corta 205 g de yuca en trozos pequeños», y
«1¼ filetes de pescado (≈199 g)» con «mide 215 g de filete de pescado blanco». El lote 328 emparejaba por la primera
palabra («pedazo» ≠ «yuca»; «filetes» ≠ «filete»).
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cantidades as pc  # noqa: E402


def _cena():
    return {"name": "Filete de pescado blanco con majado rápido de yuca y zanahoria",
            "ingredients": ["1¼ filetes de pescado (≈199 g)", "½ pedazo mediano de yuca (≈172 g)", "1 zanahoria mediana"],
            "recipe": ["Mise en place: corta 205 g de yuca en trozos pequeños y 1 zanahoria en rodajas; mide 215 g de "
                       "filete de pescado blanco.",
                       "El Toque de Fuego: cocina la yuca en el microondas 8-10 minutos; cocina el pescado 3-4 minutos "
                       "por lado."]}


def test_pedazo_de_yuca_y_filetes():
    m = _cena()
    assert pc.gramos_de_la_pieza(m) == 1
    assert "corta 172 g de yuca" in m["recipe"][0] and "mide 199 g de filete de pescado" in m["recipe"][0], m["recipe"][0]


def test_un_reparto_sigue_sin_tocarse():
    m = _cena()
    m["recipe"][0] = "Mise en place: corta la mitad de los 205 g de yuca en trozos."
    before = list(m["recipe"])
    pc.gramos_de_la_pieza(m)
    assert m["recipe"] == before


def test_la_pechuga_de_siempre_igual():
    m = {"name": "Pollo", "ingredients": ["1 pechuga de pollo (≈158 g)"],
         "recipe": ["Mise en place: corta 205 g de pechuga en tiras."]}
    assert pc.gramos_de_la_pieza(m) == 1 and "158 g de pechuga" in m["recipe"][0]
