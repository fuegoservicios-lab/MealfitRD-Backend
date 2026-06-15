"""[P2-CARB-TARGET-TRIM · 2026-06-15] Trim de carbos consciente de porción, POST-cuantización.

Causa raíz (instrumentada, 1 plan, pool abierto): el reconcile multi-macro nivela los carbos al target
(verificado 12/12 calls), pero la cuantización de porciones (FS2, Guard 4 del clinical layer) recomputa
los macros del meal desde los ingredientes redondeados — y su piso mínimo ("no puedes servir 13g de arroz")
infla las porciones pequeñas de perfiles de bajas calorías → carbos +17.5% sobre target en flota
(aware.json, 19 planes). El recompute es HONESTO (el plan REALMENTE tiene esos carbos).

Fix: `_trim_day_carbs_to_target` corre DESPUÉS de la cuantización. Escala los ingredientes CARBO-dominantes
hacia el target y RE-CUANTIZA cada uno → porciones cocinables (las flexibles g/taza/cda bajan; las de
unidad-entera —huevo/wrap— se auto-protegen al re-snapear a entero, evitando "0.73 huevos"). Recompute
honesto de macros. El residual 'at-min' es irreducible (cota de porción cocinable) y se acepta. Knob
`MEALFIT_CARB_TARGET_TRIM` (default OFF → A/B antes de prod).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GO = _REPO_ROOT / "backend" / "graph_orchestrator.py"


# ───────────────────────── parser-based ─────────────────────────

def test_knob_defined_default_off():
    src = _GO.read_text(encoding="utf-8")
    assert "P2-CARB-TARGET-TRIM" in src
    m = re.search(r'CARB_TARGET_TRIM_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_CARB_TARGET_TRIM"\s*,\s*(\w+)\)', src)
    assert m, "knob CARB_TARGET_TRIM_ENABLED no definido"
    assert m.group(1) == "False", "default DEBE ser False (A/B antes de encender en prod)"


def test_trim_called_after_quantization_in_clinical_layer():
    """El trim DEBE correr DESPUÉS de _apply_portion_quantization (si no, la cuantización lo revierte)."""
    src = _GO.read_text(encoding="utf-8")
    idx_quant = src.find("_q_n = _apply_portion_quantization(plan, _db)")
    idx_trim = src.find("_trim_day_carbs_to_target(_d.get(\"meals\", []) or [], _tc, _db")
    assert idx_quant != -1 and idx_trim != -1, "callsites no encontrados"
    assert idx_quant < idx_trim, "el trim debe ejecutarse DESPUÉS de la cuantización"


# ───────────────────────── funcional ─────────────────────────

def _db():
    from nutrition_db import IngredientNutritionDB
    return IngredientNutritionDB(rows=[
        {"name": "Arroz blanco", "aliases": ["arroz"], "kcal_per_100g": 130,
         "protein_g_per_100g": 2.7, "carbs_g_per_100g": 28, "fats_g_per_100g": 0.3, "density_g_per_cup": 158},
        {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 165,
         "protein_g_per_100g": 31, "carbs_g_per_100g": 0, "fats_g_per_100g": 3.6},
    ])


def _meal_over():
    return [{"name": "Almuerzo", "protein": 47, "carbs": 112, "fats": 6,
             "ingredients": ["400g de arroz blanco", "150g de pechuga de pollo"],
             "ingredients_raw": ["400g de arroz blanco", "150g de pechuga de pollo"]}]


def test_trims_overtarget_carbs_toward_target():
    import graph_orchestrator as go
    meals = _meal_over()
    c_before = sum(go._meal_macro_num(m.get("carbs")) for m in meals)
    assert go._trim_day_carbs_to_target(meals, 50.0, _db(), tol=0.10) is True
    c_after = sum(go._meal_macro_num(m.get("carbs")) for m in meals)
    assert c_after < c_before, f"carbos deben bajar ({c_after} vs {c_before})"
    assert abs(c_after - 50.0) < abs(c_before - 50.0), "carbos deben acercarse al target"


def test_non_carb_dominant_ingredient_untouched():
    """La fuente proteica principal (pollo, NO carbo-dominante) NO debe escalarse."""
    import graph_orchestrator as go
    meals = _meal_over()
    go._trim_day_carbs_to_target(meals, 50.0, _db(), tol=0.10)
    assert "150g de pechuga de pollo" in meals[0]["ingredients"], "el pollo (proteína) debe quedar intacto"
    # el arroz (carbo-dominante) sí se reduce
    assert not any("400g de arroz" in s for s in meals[0]["ingredients"]), "el arroz debe haberse reducido"


def test_noop_when_already_in_band():
    """Día ya cerca del target (dentro de tol) → no-op (no toca porciones cocinables)."""
    import graph_orchestrator as go
    meals = [{"name": "C", "protein": 30, "carbs": 52, "fats": 10,
              "ingredients": ["180g de arroz blanco"], "ingredients_raw": ["180g de arroz blanco"]}]
    # 52 carbos, target 50, tol 0.10 → 52 <= 55 → no-op
    assert go._trim_day_carbs_to_target(meals, 50.0, _db(), tol=0.10) is False
    assert meals[0]["ingredients"] == ["180g de arroz blanco"], "no debe tocar un día ya en banda"
