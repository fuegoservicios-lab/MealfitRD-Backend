"""[P1-PANTRY-FIRST-PLAN · 2026-07-11] F3 — "Crear plan desde mi Nevera":

1. Pre-flight determinista `POST /api/plans/pantry-feasibility` (cero LLM): kcal/proteína
   disponibles en user_inventory (gramos vía master: masa directa, unidad×density,
   paquete×container, taza×density_cup) vs necesidades de N días del objetivo; gaps con
   sugerencias de compra a precio del catálogo (proteína por densidad/precio, kcal por
   staples). Umbral 0.85 (la lista de compras cubre el residuo con honestidad).
2. Generación pantry-first: planSource='pantry' en el SSE → inyección SERVER-SIDE del
   inventario real como current_pantry_ingredients (formato _parse_quantity, espejo del
   id_string del frontend) → build_pantry_context emite el bloque Zero-Waste EXISTENTE
   y la validación pantry del review aplica. update_reason=None (generación de
   formulario) NO choca con el guard variety-ignora-pantry.
3. Frontend: QPlanSource como step 0 del wizard (default 'scratch' — nadie se bloquea) +
   pre-flight best-effort en el submit (toasts honestos; jamás bloquea).

tooltip-anchor: P1-PANTRY-FIRST-PLAN
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_FLOW = (_BACKEND.parent / "frontend" / "src" / "components" / "assessment" / "InteractiveAssessmentFlow.jsx")


# ---------------------------------------------------------------------------
# 1. Conversión inventario → gramos
# ---------------------------------------------------------------------------

def test_pantry_item_grams_units():
    from routers.plans import _pantry_item_grams
    assert _pantry_item_grams(2, "lb", {}) == 907.184
    assert _pantry_item_grams(500, "g", {}) == 500.0
    assert _pantry_item_grams(3, "unidad", {"density_g_per_unit": 120}) == 360.0
    assert _pantry_item_grams(2, "paquete", {"container_weight_g": 454}) == 908.0
    assert _pantry_item_grams(1, "taza", {"density_g_per_cup": 240}) == 240.0
    # fail-open: unidad sin density / unit desconocida → 0 (no cuenta, no explota)
    assert _pantry_item_grams(3, "unidad", {}) == 0.0
    assert _pantry_item_grams(1, "chin", {}) == 0.0


# ---------------------------------------------------------------------------
# 2. Reporte de factibilidad (funcional con DB stub)
# ---------------------------------------------------------------------------

def _fake_esq_factory(inv_rows, master_rows, sugg_rows):
    def _fake_esq(sql, params=(), fetch_all=False, fetch_one=False, **kw):
        if "FROM user_inventory" in sql:
            return inv_rows
        if "density_g_per_unit" in sql and "FROM master_ingredients" in sql:
            return master_rows
        return sugg_rows
    return _fake_esq


def test_feasibility_report_feasible_and_infeasible():
    import db_core
    from routers.plans import _pantry_feasibility_report
    master = [{"name": "Pechuga de pollo", "aliases": [], "kcal_per_100g": 165,
               "protein_g_per_100g": 31, "density_g_per_unit": None,
               "density_g_per_cup": None, "container_weight_g": None, "price_per_lb": 95},
              {"name": "Arroz blanco", "aliases": ["arroz"], "kcal_per_100g": 360,
               "protein_g_per_100g": 7, "density_g_per_unit": None,
               "density_g_per_cup": 185, "container_weight_g": 2000, "price_per_lb": 30}]
    # Nevera abundante: 4 lb de pollo + 1 paquete (2kg) de arroz → 2 días de 2000 kcal
    inv = [{"ingredient_name": "Pechuga de pollo", "quantity": 4, "unit": "lb"},
           {"ingredient_name": "arroz", "quantity": 1, "unit": "paquete"}]
    with patch.object(db_core, "execute_sql_query", _fake_esq_factory(inv, master, [])):
        r = _pantry_feasibility_report("u1", 2, 2000, 100)
    assert r["pantry_items_counted"] == 2
    assert r["feasible"] is True and r["days_supported"] >= 2
    # Nevera corta para 30 días → infeasible con gaps + sugerencias
    sugg = [{"name": "Huevo", "price_per_lb": 60, "protein_g_per_100g": 13, "kcal_per_100g": 155,
             "carbs_g_per_100g": 1}]
    with patch.object(db_core, "execute_sql_query", _fake_esq_factory(inv, master, sugg)):
        r30 = _pantry_feasibility_report("u1", 30, 2000, 100)
    assert r30["feasible"] is False
    assert r30["gaps"] and all(g["suggestions"] for g in r30["gaps"])
    assert 0 < r30["days_supported"] < 30


# ---------------------------------------------------------------------------
# 3. Endpoint + inyección SSE (parser)
# ---------------------------------------------------------------------------

def test_endpoint_wiring():
    i = _PLANS.find('@router.post("/pantry-feasibility")')
    assert i > 0
    blk = _PLANS[i: i + 2600]
    assert "Depends(_PANTRY_FEAS_LIMITER)" in blk, "RateLimiter, NO paywall"
    assert "raise HTTPException(status_code=403" in blk, "guests fuera (la Nevera vive en el perfil)"
    assert "calculate_bmr" in blk and "apply_goal_adjustment" in blk, "targets con el SSOT de la calculadora"


def test_sse_injection_pantry_mode():
    i = _PLANS.find('str(data.get("planSource") or "").strip().lower() == "pantry"')
    assert i > 0, "la inyección pantry-first desapareció del SSE"
    blk = _PLANS[i - 600: i + 2400]
    assert "MEALFIT_PANTRY_FIRST_MODE" in blk, "knob de rollback"
    assert 'pipeline_data["current_pantry_ingredients"] = _pf_items' in blk, (
        "el inventario REAL entra como current_pantry_ingredients (Zero-Waste + validación pantry)"
    )
    assert "FROM user_inventory" in blk, "fuente server-side, jamás el request"
    # orden: después del strip de untrusted keys
    i_strip = _PLANS.find('_strip_untrusted_internal_keys(pipeline_data, allow_set=None, log_prefix="ROUTER /analyze/stream")')
    assert 0 < i_strip < i


def test_zero_waste_context_fires_for_form_generation():
    # update_reason=None (formulario) + current_pantry → build_pantry_context emite bloque
    from prompts.plan_generator import build_pantry_context
    ctx = build_pantry_context({
        "current_pantry_ingredients": ["2 lb de Pechuga de pollo", "1 paquete de Arroz blanco"],
    })
    assert ctx and "pechuga" in ctx.lower() or "Pechuga" in ctx, "el bloque Zero-Waste debe emitirse"
    # variety (renovación) sigue ignorando la pantry por diseño
    assert build_pantry_context({
        "update_reason": "variety",
        "current_pantry_ingredients": ["2 lb de Pechuga de pollo"],
    }) == ""


# ---------------------------------------------------------------------------
# 4. Frontend (parser)
# ---------------------------------------------------------------------------

def test_frontend_wizard_and_preflight():
    src = _FLOW.read_text(encoding="utf-8")
    assert "QPlanSource" in src, "el step 0 del wizard desapareció"
    assert "pantry-feasibility" in src, "el pre-flight del submit desapareció"
    assert "formData.planSource === 'pantry'" in src
    i_pf = src.find("pantry-feasibility")
    i_nav = src.find("navigate('/plan')", i_pf)
    assert 0 < i_pf < i_nav, "el pre-flight corre ANTES de navegar a la generación"
    q = (_FLOW.parent / "questions" / "QPlanSource.jsx").read_text(encoding="utf-8")
    assert "planSource" in q and "scratch" in q and "pantry" in q


def test_marker_anchored_in_source():
    assert _PLANS.count("P1-PANTRY-FIRST-PLAN") >= 3
