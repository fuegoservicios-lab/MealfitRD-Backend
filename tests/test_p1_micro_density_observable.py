"""[P1-MICRO-DENSITY-OBSERVABLE · 2026-06-26] (auditoría gap #6) El descarte SILENCIOSO de micros por
densidad faltante ahora es OBSERVABLE.

Modo de fallo: un ingrediente medido en "taza"/volumen SIN density_g_per_cup (o unidad discreta sin
density_g_per_unit) resuelve por nombre pero NO por gramos → micros_from_ingredient_string descarta TODOS
sus micros y devuelve None, sin rastro. Eso producía falsos-bajos (Vit K/fibra) que solo se cazaban con
auditorías manuales reactivas. Ahora emite un WARN dedup-por-ingrediente para backfillar la densidad
proactivamente. Complementa el backfill de datos (scripts/micro_backfill_gap6_2026_06_26.py).
"""
from __future__ import annotations

import logging

import nutrition_db
from nutrition_db import IngredientNutritionDB

# Ingrediente SIN density_g_per_cup ni density_g_per_unit → "1 taza" no resuelve a gramos.
_NODENSITY = {
    "name": "Frutamística", "aliases": ["frutamistica"],
    "kcal_per_100g": 50, "protein_g_per_100g": 1, "carbs_g_per_100g": 12, "fats_g_per_100g": 0.2,
    "fiber_g_per_100g": 2, "vitamin_k_mcg_per_100g": 100,
}


def test_dropped_micros_emit_observability_warning(caplog):
    nutrition_db._MICRO_DENSITY_GAP_WARNED.clear()
    db = IngredientNutritionDB(rows=[_NODENSITY])
    with caplog.at_level(logging.WARNING, logger="nutrition_db"):
        out = db.micros_from_ingredient_string("1 taza de frutamistica")
    assert out is None, "sin densidad volumétrica → micros descartados (None)"
    assert "P1-MICRO-DENSITY-OBSERVABLE" in caplog.text, "debe emitir el WARN de gap de densidad"
    assert "Frutamística" in nutrition_db._MICRO_DENSITY_GAP_WARNED


def test_warning_dedups_per_ingredient(caplog):
    nutrition_db._MICRO_DENSITY_GAP_WARNED.clear()
    db = IngredientNutritionDB(rows=[_NODENSITY])
    with caplog.at_level(logging.WARNING, logger="nutrition_db"):
        db.micros_from_ingredient_string("1 taza de frutamistica")
        db.micros_from_ingredient_string("2 tazas de frutamistica")
    n = sum(1 for r in caplog.records if "P1-MICRO-DENSITY-OBSERVABLE" in r.getMessage())
    assert n == 1, f"debe deduplicar: 1 WARN por ingrediente por proceso (vio {n})"


def test_no_warning_when_grams_resolve(caplog):
    """Con hint de gramos explícito '(50g)' resuelve → micros computados, sin WARN."""
    nutrition_db._MICRO_DENSITY_GAP_WARNED.clear()
    db = IngredientNutritionDB(rows=[_NODENSITY])
    with caplog.at_level(logging.WARNING, logger="nutrition_db"):
        out = db.micros_from_ingredient_string("1 taza de frutamistica (50g)")
    assert out is not None and out["grams"] == 50.0
    assert "P1-MICRO-DENSITY-OBSERVABLE" not in caplog.text


def test_no_warning_for_unresolved_name(caplog):
    """Si el nombre NO resuelve (no es del catálogo), NO se warnea (no es un gap de densidad)."""
    nutrition_db._MICRO_DENSITY_GAP_WARNED.clear()
    db = IngredientNutritionDB(rows=[_NODENSITY])
    with caplog.at_level(logging.WARNING, logger="nutrition_db"):
        out = db.micros_from_ingredient_string("1 taza de ingrediente_inexistente_xyz")
    assert out is None
    assert "P1-MICRO-DENSITY-OBSERVABLE" not in caplog.text
