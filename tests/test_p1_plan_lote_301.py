# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-301 · 2026-09-25] La proteína COCIDA contra su fila CRUDA.

«150 g de pechuga de pollo cocida» contaba 150 g de pechuga cruda (161 kcal, 34 g de proteína); cocida, salió de
~200 g crudos (217 kcal, 45 g). 98 líneas así en las baterías del 25-sep, casi todas en perfiles «Nada»."""
from __future__ import annotations

import pathlib

import cocido_en_catalogo as cc
from nutrition_db import IngredientNutritionDB

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_FILAS = [
    {"name": "Pechuga de pollo", "kcal_per_100g": 107.4, "protein_g_per_100g": 22.5, "carbs_g_per_100g": 0.0,
     "fats_g_per_100g": 1.9, "density_g_per_unit": 170, "category": "Proteínas"},
    {"name": "Filete de pescado blanco", "kcal_per_100g": 95.7, "protein_g_per_100g": 20.1, "carbs_g_per_100g": 0.0,
     "fats_g_per_100g": 1.5, "density_g_per_unit": 150, "category": "Proteínas"},
    {"name": "Camarones", "kcal_per_100g": 85.0, "protein_g_per_100g": 20.1, "carbs_g_per_100g": 0.0,
     "fats_g_per_100g": 0.5, "category": "Proteínas"},
    {"name": "Jamón de pavo", "kcal_per_100g": 106.8, "protein_g_per_100g": 18.4, "carbs_g_per_100g": 2.0,
     "fats_g_per_100g": 3.0, "density_g_per_unit": 28, "category": "Proteínas"},
    {"name": "Arroz blanco", "kcal_per_100g": 358.6, "protein_g_per_100g": 7.1, "carbs_g_per_100g": 79.5,
     "fats_g_per_100g": 0.7, "density_g_per_cup": 185, "category": "Despensa"},
]


def _db():
    return IngredientNutritionDB(rows=[dict(r) for r in _FILAS])


def _m(db, linea):
    return db.macros_from_ingredient_string(linea) or {}


def test_la_pechuga_cocida_cuenta_lo_que_pesaba_cruda():
    db = _db()
    f = cc.factor_proteina()
    assert f == 1.35
    m = _m(db, "150 g de pechuga de pollo cocida")
    assert abs(m["grams"] - 150 * f) < 0.5 and abs(m["protein"] - 150 * f * 0.225) < 0.5, m     # antes 34 g
    assert abs(_m(db, "150 g de pechuga de pollo cocida y desmenuzada")["grams"] - 150 * f) < 0.5
    assert abs(_m(db, "120 g de camarones cocidos")["grams"] - 120 * f) < 0.5
    assert abs(_m(db, "100 g de filete de pescado blanco a la plancha, asado")["grams"] - 100 * f) < 0.5


def test_lo_que_no_se_toca():
    db = _db()
    assert _m(db, "150 g de pechuga de pollo")["grams"] == 150                                  # sin cocción
    assert _m(db, "150 g de pechuga de pollo a la plancha")["grams"] == 150                     # la lista tampoco
    assert _m(db, "1 pechuga de pollo cocida")["grams"] == 170                                  # una pieza es una pieza
    assert _m(db, "150 g de pechuga de pollo cocida (200 g en crudo)")["grams"] == 200         # el paréntesis ya es crudo
    assert _m(db, "100 g de jamón de pavo cocido")["grams"] == 100                              # embutido: se vende hecho
    assert abs(_m(db, "60 g de arroz blanco frito")["grams"] - 60) < 0.01                        # un grano no es proteína


def test_ganchos():
    src = (_BACKEND / "cocido_en_catalogo.py").read_text(encoding="utf-8")
    assert "return _proteina_cocida(linea, gramos, info, unidad) if (prot and info) else gramos" in src
    assert "tooltip-anchor: P1-PLAN-LOTE-301-PROTEINA-COCIDA" in src
