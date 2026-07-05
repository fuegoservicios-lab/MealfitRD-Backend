"""[P1-FATS-POSTCLOSER-RELEVEL · 2026-07-05] Re-trim de GRASAS post-micro-recheck P2-2.

Causa raíz (medida en vivo, corrida 7f99b955): el re-fire P2-2 del micro-closer es la ÚLTIMA
pasada macro-mutante del motor y solo compensaba CARBOS (`_trim_day_carbs_to_target`), pero lo
que el closer siembra/escala es GRASA-denso (girasol 51% grasa, linaza 42%, maní 49%). El
cal-reconcile había nivelado las kcal exactas ANTES de la capa clínica → la grasa añadida por
el P2-2 quedaba sin compensar → plan aprobado en intento 1 con per_macro fats 0.333 + kcal
0.333 → banner low_band_macro:fats ("Calidad por debajo del óptimo").

Fix: `_trim_day_fats_to_target` (espejo del carb-trim) corre dentro del bloque P2-2, tras el
re-trim de carbos. Recorta grasa de fuentes grasa-dominantes NO-portadoras (aceite/mantequilla/
aguacate/queso) y PROTEGE las líneas con `_SEED_NUT_TOKENS` (el cierre de vit E/omega-3 que el
closer acaba de lograr). La grasa embebida en proteína-dominantes queda intacta (contrato del
carb-trim). Knob `MEALFIT_FATS_POSTCLOSER_RELEVEL` (default ON) + TOL (default 0.08).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GO_PATH = _REPO_ROOT / "backend" / "graph_orchestrator.py"
_GO = _GO_PATH.read_text(encoding="utf-8")


# ───────────────────────── parser-based ─────────────────────────

def test_knob_defined_default_on():
    assert "P1-FATS-POSTCLOSER-RELEVEL" in _GO
    m = re.search(r'FATS_POSTCLOSER_RELEVEL_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_FATS_POSTCLOSER_RELEVEL"\s*,\s*(\w+)\)', _GO)
    assert m and m.group(1) == "True", "knob default ON (cierra un banner medido en prod)"
    assert '_env_float("MEALFIT_FATS_POSTCLOSER_RELEVEL_TOL", 0.08)' in _GO


def test_wired_inside_p2_2_block_after_carb_trim_before_quantize():
    """El trim de grasas corre DENTRO del bloque P2-2, tras el carb-trim y antes del re-quantize
    (el quantize final ve el estado ya trimado)."""
    _blk_start = _GO.index("(P2-2) Micro-recheck post-motor")
    _blk = _GO[_blk_start:_blk_start + 3600]
    i_carb = _blk.index("_trim_day_carbs_to_target")
    i_fat = _blk.index("_trim_day_fats_to_target")
    i_quant = _blk.index("_apply_portion_quantization")
    assert i_carb < i_fat < i_quant, "orden: carb-trim → fats-trim → re-quantize"
    assert "FATS_POSTCLOSER_RELEVEL_ENABLED" in _blk, "el seam debe estar gateado por el knob"


def test_seed_tokens_ssot_shared_with_line_clamp():
    """`_SEED_NUT_TOKENS` es el SSOT: lo usa el clamp de línea del closer Y la protección del trim."""
    assert "_SEED_NUT_TOKENS = (" in _GO
    i_clamp = _GO.index("[P1-CLOSER-LINE-CLAMP · 2026-07-05] techo ABSOLUTO")
    assert "_SEED_NUT_TOKENS" in _GO[i_clamp:i_clamp + 900]


# ───────────────────────── funcional ─────────────────────────

def _db():
    from nutrition_db import IngredientNutritionDB
    return IngredientNutritionDB(rows=[
        {"name": "Aceite de oliva", "aliases": ["aceite"], "kcal_per_100g": 884,
         "protein_g_per_100g": 0, "carbs_g_per_100g": 0, "fats_g_per_100g": 100},
        {"name": "Semillas de girasol", "aliases": ["girasol"], "kcal_per_100g": 584,
         "protein_g_per_100g": 20.8, "carbs_g_per_100g": 20, "fats_g_per_100g": 51.5},
        {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 165,
         "protein_g_per_100g": 31, "carbs_g_per_100g": 0, "fats_g_per_100g": 3.6},
    ])


def _meals_over():
    # grasa real: aceite 30g + girasol ~10.3g + pollo ~5.4g ≈ 46g
    return [{"name": "Almuerzo", "protein": 51, "carbs": 4, "fats": 46,
             "ingredients": ["30g de aceite de oliva", "20g de semillas de girasol",
                             "150g de pechuga de pollo"],
             "ingredients_raw": ["30g de aceite de oliva", "20g de semillas de girasol",
                                 "150g de pechuga de pollo"]}]


def test_trims_overtarget_fats_toward_target():
    import graph_orchestrator as go
    meals = _meals_over()
    f_before = sum(go._meal_macro_num(m.get("fats")) for m in meals)
    assert go._trim_day_fats_to_target(meals, 30.0, _db(), tol=0.08) is True
    f_after = sum(go._meal_macro_num(m.get("fats")) for m in meals)
    assert f_after < f_before, f"las grasas deben bajar ({f_after} vs {f_before})"
    assert abs(f_after - 30.0) < abs(f_before - 30.0), "las grasas deben acercarse al target"


def test_seed_and_protein_lines_protected():
    """El girasol (portador del cierre de vit E, grasa-dominante) y el pollo (proteína-dominante)
    quedan INTACTOS — la grasa se recorta SOLO del aceite (fuente no-portadora)."""
    import graph_orchestrator as go
    meals = _meals_over()
    go._trim_day_fats_to_target(meals, 30.0, _db(), tol=0.08)
    assert "20g de semillas de girasol" in meals[0]["ingredients"], \
        "la semilla del closer es el cierre de micros — jamás encogerla"
    assert "150g de pechuga de pollo" in meals[0]["ingredients"], "la proteína queda intacta"
    assert not any("30g de aceite" in s for s in meals[0]["ingredients"]), \
        "el aceite (grasa-dominante no-portadora) debe haberse reducido"


def test_raw_lockstep():
    import graph_orchestrator as go
    meals = _meals_over()
    go._trim_day_fats_to_target(meals, 30.0, _db(), tol=0.08)
    assert meals[0]["ingredients"] == meals[0]["ingredients_raw"] or \
        len(meals[0]["ingredients"]) == len(meals[0]["ingredients_raw"]), \
        "ingredients_raw debe escalarse en lockstep"
    assert not any("30g de aceite" in s for s in meals[0]["ingredients_raw"]), \
        "el raw del aceite también debe reducirse (lista de compras coherente)"


def test_noop_when_already_in_band():
    import graph_orchestrator as go
    meals = [{"name": "C", "protein": 30, "carbs": 50, "fats": 31,
              "ingredients": ["31g de aceite de oliva"], "ingredients_raw": ["31g de aceite de oliva"]}]
    # 31g de grasa, target 30, tol 0.08 → 31 <= 32.4 → no-op
    assert go._trim_day_fats_to_target(meals, 30.0, _db(), tol=0.08) is False
    assert meals[0]["ingredients"] == ["31g de aceite de oliva"]


def test_noop_when_only_protected_sources():
    """Día cuyo exceso de grasa viene SOLO de semillas/frutos secos → no hay masa movible → no-op
    honesto (jamás romper el cierre de micros para cerrar la banda)."""
    import graph_orchestrator as go
    meals = [{"name": "M", "protein": 21, "carbs": 20, "fats": 52,
              "ingredients": ["100g de semillas de girasol"],
              "ingredients_raw": ["100g de semillas de girasol"]}]
    assert go._trim_day_fats_to_target(meals, 30.0, _db(), tol=0.08) is False
    assert meals[0]["ingredients"] == ["100g de semillas de girasol"]
