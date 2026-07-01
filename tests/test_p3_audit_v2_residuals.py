"""[P3-AUDIT-V2-RESIDUALS · 2026-07-01] Residuos + P3 del audit objetivo v2 — cierre final.

7 con código/datos:
  S1  avena al blocklist name-based de ALMUERZO (con excludes: costra/empanizado/harina/leche de avena).
  S3  carb-floor closer jamás escala arroz en la CENA (el "toque de arroz" tangencial gate-excluido
      podía inflarse ×2.5 a arroz nocturno pleno).
  S4  plato nombrado "desayuno" servido en la cena → violación soft (honestidad nombre↔slot).
  M1  cron diario `_micro_floor_kpi_job`: cobertura de pisos DRI a nivel flota → pipeline_metrics.
  M2  caveat "dato estimado" en la nota de suplemento cuando status=estimado_bajo (asimetría de
      honestidad vs la rama ceiling, que ya tenía nota dedicada).
  M3  migración p3_density_backfill: 15 filas sin NINGUNA densidad backfilleadas (porción típica
      curada) + DO-block cero-ambas-NULL (una mención "1 unidad de X" descartaba micros en silencio).
  MA2 `_quality_degraded_band_per_macro_low` propagado en el overlay T2 de chunks (semanas 2+).
  C1  comentarios "los 119" stale actualizados al catálogo dinámico (~202).

4 cerrados COMO DECISIÓN (no código — que un audit futuro no los re-flagee):
  D-1 Fallback matemático NO corre el macro-solver: decisión de review clínica adversaria
      (P1-FALLBACK-BAND-AWARE: "NO escalamos hacia banda a propósito — un plan de contingencia
      ligeramente bajo banda es el lado SEGURO post-bariátrico"). Ya tiene physical-macros +
      rotación + capa clínica + `_is_fallback` honesto y está excluido del band-score.
  D-2 `meal["cals"]` = kcal de CATÁLOGO, no 4·P+4·C+9·F: Atwater genérico es MENOS preciso.
      La UI no debe recomputar (ancla P3-CALS-CATALOG-DECISION).
  D-3 `dominican_dish_recipes.json` sigue muerto POR DISEÑO (el LLM itemiza ingredientes, no emite
      'locrío' como línea → un lookup de macros por plato compuesto no encaja en el pipeline; la
      creatividad va por pools+prompts y se MIDE con el cron creativity_kpi).
  D-4 Detección slot ingredient-level NO se generaliza a merienda/almuerzo: "arroz con leche" es
      merienda legítima (falso positivo directo) y "plato fuerte desde ingredients" no es
      determinizable sin fuzzy. El caso con daño real (arroz oculto en DESAYUNO, hard) ya se cerró
      en P2-SLOT-INGREDIENT-RICE.
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as go
from constants import slot_violations_for_meal_name

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_MICRO = (_BACKEND / "micronutrients.py").read_text(encoding="utf-8")
_SHOP = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


# ── S1 · avena en almuerzo ──────────────────────────────────────────────────
# NB: `slot_violations_for_meal_name` espera el slot YA canonicalizado (minúscula) —
# los callers de producción pasan por `canonical_slot_key` antes.
def test_s1_avena_lunch_main_flagged():
    v = slot_violations_for_meal_name("Avena cremosa con frutas y maní", "almuerzo")
    assert v and v[0]["hard"] is False, "avena como plato principal del almuerzo debe ser violación soft"


def test_s1_avena_excludes_protect_legit_uses():
    assert slot_violations_for_meal_name("Pollo con costra de avena", "almuerzo") == []
    assert slot_violations_for_meal_name("Pescado empanizado de avena", "almuerzo") == []
    # en desayuno/merienda la avena es legítima (el rule vive solo en almuerzo)
    assert slot_violations_for_meal_name("Avena caliente con canela", "desayuno") == []


# ── S3 · carb-floor no escala arroz en cena ─────────────────────────────────
def test_s3_carbfloor_night_rice_skip_anchored():
    assert "P3-CARBFLOOR-NIGHT-RICE-SKIP" in _GRAPH
    i = _GRAPH.find("def _close_carb_gap_for_day")
    seg = _GRAPH[i:i + 4500]
    assert "_is_cena_cf" in seg and "_SLOT_RICE_TOKENS" in seg, \
        "el closer de carbos no excluye arroz-en-cena como target de escalado"


# ── S4 · honestidad nombre↔slot ─────────────────────────────────────────────
def test_s4_dinner_named_breakfast_flagged():
    v = slot_violations_for_meal_name("Desayuno criollo completo", "cena")
    assert any("desayuno" in x["label"].lower() for x in v)
    assert slot_violations_for_meal_name("Desayuno criollo completo", "desayuno") == []


# ── M1 · KPI de pisos DRI ───────────────────────────────────────────────────
def test_m1_micro_floor_kpi_cron_registered():
    assert "def _micro_floor_kpi_job" in _CRON
    assert 'id="micro_floor_kpi_job"' in _CRON
    assert "MEALFIT_MICRO_KPI_INTERVAL_MIN" in _CRON
    assert "all_floors_ok_ratio" in _CRON


# ── M2 · caveat estimado_bajo ───────────────────────────────────────────────
def test_m2_estimado_bajo_caveat():
    assert "P3-FLOOR-ESTIMADO-CAVEAT" in _MICRO
    assert "Dato estimado" in _MICRO, "falta el caveat de dato incierto en la nota del piso"
    i = _MICRO.find('if status == "estimado_bajo" and entry.get("nota")')
    assert i != -1, "el caveat debe anexarse solo cuando el status es estimado_bajo"


# ── M3 · densidades ─────────────────────────────────────────────────────────
def test_m3_density_migration_both_dirs():
    mig = _BACKEND / "migrations" / "p3_density_backfill_2026_07_01.sql"
    assert mig.exists()
    sql = mig.read_text(encoding="utf-8")
    assert "RAISE EXCEPTION" in sql and "density_g_per_cup IS NULL AND density_g_per_unit IS NULL" in sql
    assert "COALESCE(density_g_per_unit" in sql, "el backfill debe ser COALESCE (idempotente)"
    root = _BACKEND.parent / "migrations" / "p3_density_backfill_2026_07_01.sql"
    if root.exists():
        assert sql == root.read_text(encoding="utf-8"), "drift SSOT migrations"


# ── MA2 · propagación per-macro en chunks ───────────────────────────────────
def test_ma2_band_per_macro_low_propagated_in_chunks():
    assert _CRON.count("_quality_degraded_band_per_macro_low") >= 2, \
        "el flag debe estar en P0_4_T2_INCREMENTAL_KEYS Y en el loop de propagación P2-10"


# ── C1 · comentarios "119" stale ────────────────────────────────────────────
def test_c1_no_stale_119_comments():
    assert "los 119 de master_ingredients" not in _SHOP
    assert "solo los 119" not in _SHOP


# ── D-1..D-4 · decisiones ancladas ──────────────────────────────────────────
def test_d1_fallback_no_solver_is_a_safety_decision():
    assert "NO escalamos hacia banda a propósito" in _GRAPH, \
        "la decisión de seguridad del fallback (P1-FALLBACK-BAND-AWARE) desapareció — leer el docstring de este test antes de cablear el solver al fallback"


def test_d2_cals_catalog_decision_anchored():
    assert "P3-CALS-CATALOG-DECISION" in _GRAPH


def test_d3_dish_library_stays_dead_by_design():
    dq = (_BACKEND / "tests" / "test_p2_dish_quality.py").read_text(encoding="utf-8")
    assert "muerto POR DISEÑO" in dq, \
        "si se decide cablear dominican_dish_recipes.json, actualizar AMBAS anclas con la nueva decisión"


def test_d4_no_ingredient_scan_for_merienda():
    from constants import slot_ingredient_violations
    # arroz con leche = merienda legítima → el detector ingredient-level NO debe cubrir merienda
    assert slot_ingredient_violations(["1 taza de arroz", "leche"], "Merienda") == []
