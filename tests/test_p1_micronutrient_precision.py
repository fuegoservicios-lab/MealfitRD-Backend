"""[P1-MICRONUTRIENT-PRECISION · 2026-06-24] Precisión + coherencia de micronutrientes.

Tres piezas que componen para que los micros sean tan "precisos y coherentes" como los macros:

  A. P1-MICRONUTRIENT-CATALOG-BACKFILL — rellena los micros del panel que estaban NULL en
     `master_ingredients` (un NULL contaba como 0 → subestimaba el panel → falso "magnesio bajo").
     Migración SSOT idempotente en migrations/ Y backend/migrations/.
  B. P1-FLOOR-COVERAGE-AWARE — la rama de PISOS usa la cobertura POR-NUTRIENTE (como la de techos,
     P1-CEILING-COVERAGE-AWARE): un piso incumplido cuyo micro tiene cobertura parcial → 'estimado_bajo'
     (incierto), no 'bajo' confiado.
  C. P1-MICRONUTRIENT-STEER — directiva CUANTITATIVA de micros alcanzables (magnesio/calcio/hierro/
     fibra/potasio) inyectada al prompt del day-generator + knob MEALFIT_MICRONUTRIENT_STEER.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nutrition_db import IngredientNutritionDB
from micronutrients import (
    build_micronutrient_report, build_micronutrient_targets_directive, dri_targets,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BACKEND)


# ───────────────────────── A. Backfill del catálogo (migración SSOT) ─────────────────────────

def _migration_paths():
    return [
        os.path.join(_ROOT, "migrations", "p1_micronutrient_catalog_backfill_2026_06_24.sql"),
        os.path.join(_BACKEND, "migrations", "p1_micronutrient_catalog_backfill_2026_06_24.sql"),
    ]


def test_backfill_migration_existe_en_ambos_dirs():
    # SSOT: la migración vive en migrations/ (root) Y backend/migrations/ (P3-MIGRATIONS-SSOT).
    for p in _migration_paths():
        assert os.path.exists(p), f"falta la migración del backfill en {p}"


def test_backfill_migration_es_idempotente_y_con_sanity():
    for p in _migration_paths():
        with open(p, encoding="utf-8") as f:
            sql = f.read()
        assert "P1-MICRONUTRIENT-CATALOG-BACKFILL" in sql
        # Idempotente: rellena NULL sin pisar valores existentes.
        assert "COALESCE(" in sql
        # Sanity: 0 NULL del panel tras correr (patrón DO $$ RAISE EXCEPTION del repo).
        assert "RAISE EXCEPTION" in sql
        # Corrige explícitamente el stub de Ajo (k/ca/fe eran 0 erróneo).
        assert "WHERE name = 'Ajo'" in sql or "WHERE name='Ajo'" in sql


# ───────────────────────── B. Floor coverage-aware ─────────────────────────

# 'Arroz' resuelve (tiene macros + calcio) pero NO trae magnesium_mg_per_100g → magnesio cuenta 0
# y su cobertura POR-NUTRIENTE = 0, mientras la cobertura GLOBAL es 1.0 (todo resuelve).
_ROWS_B = [
    {"name": "Arroz", "aliases": ["arroz", "arroz blanco"],
     "kcal_per_100g": 130, "protein_g_per_100g": 2.7, "carbs_g_per_100g": 28, "fats_g_per_100g": 0.3,
     "fiber_g_per_100g": 0.4, "sodium_mg_per_100g": 1, "calcium_mg_per_100g": 10, "iron_mg_per_100g": 0.2,
     "potassium_mg_per_100g": 35, "vitamin_d_mcg_per_100g": 0, "vitamin_b12_mcg_per_100g": 0,
     "sugars_g_per_100g": 0.1},  # ← SIN magnesium_mg_per_100g a propósito (columna ausente = NULL)
]


def _db_b():
    return IngredientNutritionDB(rows=_ROWS_B)


def test_floor_con_cobertura_por_nutriente_pobre_es_estimado_bajo():
    # Plan 100% arroz: cobertura GLOBAL alta (todo resuelve) pero el MAGNESIO no tiene dato en
    # ningún ingrediente → antes reportaba 'bajo' confiado; ahora 'estimado_bajo' (incierto).
    plan = {"days": [{"meals": [{"ingredients": [
        "150g de arroz (150g)", "150g de arroz (150g)", "150g de arroz (150g)"]}]}]}
    rep = build_micronutrient_report(plan, _db_b(), sex="female")
    assert rep["coverage"] >= 0.6  # cobertura GLOBAL alta → NO es el disparador global
    mag = next(e for e in rep["panel"] if e["key"] == "magnesium_mg")
    assert mag["status"] == "estimado_bajo", f"esperaba estimado_bajo, fue {mag['status']}"


def test_floor_con_cobertura_por_nutriente_completa_sigue_bajo_confiado():
    # El calcio SÍ tiene dato en todos los ingredientes (cobertura por-nutriente = 1.0) y queda
    # debajo del piso → debe seguir 'bajo' CONFIADO (no degradar la señal real a estimado).
    plan = {"days": [{"meals": [{"ingredients": [
        "150g de arroz (150g)", "150g de arroz (150g)", "150g de arroz (150g)"]}]}]}
    rep = build_micronutrient_report(plan, _db_b(), sex="female")
    cal = next(e for e in rep["panel"] if e["key"] == "calcium_mg")
    assert cal["status"] == "bajo", f"esperaba bajo confiado, fue {cal['status']}"


# ───────────────────────── C. Steering numérico ─────────────────────────

def test_steer_incluye_pisos_numericos_alcanzables():
    d = build_micronutrient_targets_directive(sex="male", age=20)
    t = dri_targets("male", 20)
    # Magnesio/potasio/fibra del hombre adulto presentes como números citables.
    assert f"≥{int(round(t['magnesium_mg']['floor']))} mg" in d   # 420 mg
    assert f"≥{int(round(t['potassium_mg']['floor']))} mg" in d   # 3400 mg
    assert f"≥{int(round(t['fiber_g']['floor']))} g" in d         # 38 g
    assert "Magnesio" in d and "Potasio" in d and "Hierro" in d and "Fibra" in d and "Calcio" in d


def test_steer_excluye_vitamina_d_como_objetivo():
    # Vit D NO debe figurar como piso a alcanzar (se cubre con suplemento, no se fuerza el plan).
    d = build_micronutrient_targets_directive(sex="male", age=20)
    assert not any(l.startswith("• Vitamina D") for l in d.splitlines())  # sin bullet de objetivo
    assert "≥15 mcg" not in d and "≥20 mcg" not in d  # los pisos de vit D van en mcg (vit E va en mg)
    assert "NO la fuerces" in d  # mensaje explícito de no forzar vit D


def test_steer_sex_aware_hombre_vs_mujer():
    dm = build_micronutrient_targets_directive(sex="male", age=30)
    df = build_micronutrient_targets_directive(sex="female", age=30)
    assert "≥38 g" in dm and "≥25 g" in df   # fibra 38 (M) vs 25 (F)
    assert "≥420 mg" in dm and "≥320 mg" in df  # magnesio 420 (M) vs 320 (F)


def test_steer_med_que_eleva_potasio_no_maximiza():
    # Con un fármaco que eleva el potasio sérico, la directiva NO empuja a maximizar potasio
    # (coherente con el guard del panel P1-POTASSIUM-PANEL-MED-AWARE → evita hiperkalemia).
    d = build_micronutrient_targets_directive(sex="male", age=60, k_elevating_med=True)
    assert "MODERADAS" in d
    assert "≥3400 mg" not in d and "guineo, plátano, batata" not in d


def test_steer_nunca_lanza():
    # Robustez: inputs basura → "" sin excepción.
    assert build_micronutrient_targets_directive(sex=None, age="no-num", conditions=None) is not None


# ───────────────────────── C. Cableado al orquestador (parser-based anchor) ─────────────────────────

def test_steer_knob_y_inyeccion_en_el_orquestador():
    # Anchor: si alguien renombra el knob o quita la inyección al day-gen, este test falla ANTES
    # de cambiar producción (convención del repo: tooltip-anchor para parser-based tests).
    with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
        src = f.read()
    assert 'MICRONUTRIENT_STEER_ENABLED = _env_bool("MEALFIT_MICRONUTRIENT_STEER"' in src
    # Llave en el dict de contexto + inyección en el f-string del prompt del day-generator.
    assert '"micronutrient_targets_context":' in src
    assert "ctx['micronutrient_targets_context']" in src


# ───────────────────────── #1. Recompute de micros en el chunk worker ─────────────────────────

def test_chunk_worker_recomputa_micros_tras_merge():
    # Anchor: el chunk worker debe recalcular el panel de micros tras anexar días, para que un plan
    # chunked (15/30d) no quede con el panel clavado en la semana 1. Si alguien quita la llamada, falla.
    with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
        orch = f.read()
    with open(os.path.join(_BACKEND, "cron_tasks.py"), encoding="utf-8") as f:
        cron = f.read()
    # Knob kill-switch definido + recompute invocado en el chunk worker DESPUÉS del merge de días.
    assert 'MICRONUTRIENT_CHUNK_RECOMPUTE_ENABLED = _env_bool("MEALFIT_MICRONUTRIENT_CHUNK_RECOMPUTE"' in orch
    merge_pos = cron.find("plan_data['days'] = merged_days")
    recompute_pos = cron.find("recompute_micronutrient_report_for_plan", merge_pos)
    assert merge_pos >= 0, "no se encontró el merge `plan_data['days'] = merged_days` en el chunk worker"
    assert recompute_pos > merge_pos, "el recompute de micros debe ir DESPUÉS del merge de días"


# ───────────── P1-MICRONUTRIENT-STEER-PROTEIN-AWARE: gain_muscle prioriza proteína ─────────────

def _line(directive, prefix):
    return next(l for l in directive.splitlines() if l.startswith(prefix))


def test_steer_gain_muscle_antepone_proteina_animal():
    d = build_micronutrient_targets_directive(sex="male", age=20, goal="gain_muscle")
    # Línea de PRIORIDAD que blinda el piso de proteína / fuente animal de alta densidad.
    assert "PRIORIDAD" in d and "proteína" in d.lower()
    assert "NUNCA reemplaces la proteína principal" in d
    # Magnesio: lidera con semillas/hoja verde; leguminosa relegada a guarnición (no plato-base).
    mag = _line(d, "• Magnesio").lower()
    assert "guarnición" in mag and mag.index("nueces") < mag.index("legum")
    # Hierro: hemo animal primero, sin liderar con legumbres.
    fe = _line(d, "• Hierro").lower()
    assert "hemo" in fe and "legum" not in fe


def test_steer_no_muscle_mantiene_legumbres_como_fuente_principal():
    d = build_micronutrient_targets_directive(sex="female", age=30, goal="lose_fat")
    assert "PRIORIDAD" not in d  # solo gain_muscle blinda la proteína
    assert "legumbres (habichuelas)" in _line(d, "• Magnesio")
    # hierro mantiene legumbres en la lista (objetivo no muscular).
    assert "legumbres" in _line(d, "• Hierro").lower()


def test_steer_goal_none_es_no_muscle():
    # Sin goal (default) → comportamiento no-muscular (legumbres lideran), nunca lanza.
    d = build_micronutrient_targets_directive(sex="male", age=20)
    assert "PRIORIDAD" not in d and "• Magnesio" in d


# ───────────── P1-MICRONUTRIENT-RAW-INGREDIENTS: el reporte prefiere ingredients_raw ─────────────

from micronutrients import compute_plan_micronutrient_totals  # noqa: E402

_QUESO_ROW = {
    "name": "Queso Mozzarella", "aliases": ["queso mozzarella"],
    "kcal_per_100g": 280, "protein_g_per_100g": 28, "carbs_g_per_100g": 3, "fats_g_per_100g": 17,
    "calcium_mg_per_100g": 500, "magnesium_mg_per_100g": 27, "sodium_mg_per_100g": 600,
    "fiber_g_per_100g": 0, "iron_mg_per_100g": 0.4, "potassium_mg_per_100g": 76,
    "vitamin_b12_mcg_per_100g": 1.0, "vitamin_d_mcg_per_100g": 0, "sugars_g_per_100g": 1,
}


def test_micros_prefiere_ingredients_raw_cuando_display_no_resuelve():
    # display vago ("lonja/pedazo de queso") NO resuelve; raw canónico SÍ → el reporte usa raw.
    db = IngredientNutritionDB(rows=[_QUESO_ROW])
    plan = {"days": [{"meals": [{
        "ingredients": ["¾ lonja/pedazo de queso"],            # display → 0 resueltos
        "ingredients_raw": ["100g de queso mozzarella (100g)"],  # raw → 1 resuelto
    }]}]}
    tot = compute_plan_micronutrient_totals(plan, db)
    assert tot["resolved_ings"] == 1, "debe resolver vía ingredients_raw, no el display"
    assert tot["daily"]["calcium_mg"] > 0 and tot["daily"]["magnesium_mg"] > 0  # micros del queso contados


def test_micros_fallback_a_ingredients_si_no_hay_raw():
    # Sin ingredients_raw (plan viejo) → cae a `ingredients` (no rompe).
    db = IngredientNutritionDB(rows=[_QUESO_ROW])
    plan = {"days": [{"meals": [{"ingredients": ["100g de queso mozzarella (100g)"]}]}]}
    tot = compute_plan_micronutrient_totals(plan, db)
    assert tot["resolved_ings"] == 1


def test_micros_raw_anchor_en_source():
    # Anchor: si alguien revierte a `meal.get("ingredients")` directo, el recompute volvería a
    # sub-contar el protein-closer y a divergir de assemble. Este test lo bloquea.
    with open(os.path.join(_BACKEND, "micronutrients.py"), encoding="utf-8") as f:
        src = f.read()
    assert 'meal.get("ingredients_raw") or meal.get("ingredients")' in src
