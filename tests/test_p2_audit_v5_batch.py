"""[P2-AUDIT-V5-BATCH · 2026-07-02] Los 16 P2 del audit objetivo v5.

Origen: memoria project_objective_audit_v5_2026_07_02.md (los 2 P1 cerraron en
P1-AUDIT-V5-BATCH el mismo día). Sub-fixes cubiertos:
  GAP-14 STRICT_ALL_REASONS ON-en-código (test_p4_update_dishes_strict_all_reasons.py)
  GAP-C2 duración SSOT en cron T2/recovery · GAP-C1 cost summary en sitios cron
  GAP-C3 expand rebuild inline · GAP-04 re-agregación post review-patch
  GAP-06 partial_pricing · GAP-07 budget tier en chat-modify · GAP-08 cheapen reescribe pasos
  GAP-09 backstop tiempo/temp (|horno + persist boundary) · GAP-10 qty-sync regen-day
  GAP-11 same-day protein en chat-modify · GAP-13 raw-staple en swap
  GAP-05 piso cocinable SHRINK · GAP-03 _day_fallback honesto
  GAP-M1 per_day_ceilings · GAP-M2 micro-closer per-día (knob OFF)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_CT = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_TO = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_AG = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_MI = (_BACKEND / "micronutrients.py").read_text(encoding="utf-8")
_NC = (_BACKEND / "nutrition_calculator.py").read_text(encoding="utf-8")
_PL = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_DASH = (_BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


@pytest.fixture(scope="module")
def nc():
    import nutrition_calculator as _nc
    return _nc


class _FakeDB:
    """DB mínima para la mecánica del shrink-floor (sin Neon)."""
    def macros_from_ingredient_string(self, s):
        return {"protein": 2.0, "carbs": 0.0, "fats": 1.0, "kcal": 15.0}


def test_marker_bumped():
    """Supersession-proof: este batch o uno posterior (fecha ≥)."""
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P2-AUDIT-V5-BATCH" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-02", f"marker {m.group(1)!r} anterior al batch"


# ── GAP-C2 + GAP-C1: sitios cron ─────────────────────────────────────────────
def test_c2_cron_duration_uses_ssot():
    """T2 y recovery prefieren calc_grocery_duration sobre el form crudo."""
    hits = re.findall(
        r'full_plan_data\.get\("calc_grocery_duration"\)|plan_data\.get\("calc_grocery_duration"\)', _CT)
    assert len(hits) >= 4, f"esperaba ≥4 usos de calc_grocery_duration en cron_tasks (got {len(hits)})"
    assert not re.search(
        r'grocery_duration\s*=\s*form_data\.get\("groceryDuration",\s*"weekly"\)\n', _CT
    ), "regresión: groceryDuration crudo como única fuente en un sitio cron"


def test_c1_cron_cost_summary_wired():
    assert "'shopping_cost_summary'" in _CT and "'budget_reconciliation'" in _CT, \
        "faltan las keys de costo en P0_4_T2_INCREMENTAL_KEYS"
    assert "_compute_cost_summary_jsonb_extras" in _CT and "_wrap_jsonb_set_expr" in _CT
    # helper invocado en los 3 sitios jsonb_set (recovery + pantry persist + pantry clear)
    assert len(re.findall(r"_compute_cost_summary_jsonb_extras\(", _CT)) >= 4  # def + 3 callsites
    # T2 inline refresh
    assert "refresh_budget_reconciliation as _rbr_t2" in _CT


def test_c1_wrap_jsonb_expr_mechanics():
    import cron_tasks as ct
    expr = ct._wrap_jsonb_set_expr("plan_data", ["shopping_cost_summary", "budget_reconciliation"])
    assert expr == ("jsonb_set(jsonb_set(plan_data, '{shopping_cost_summary}', %s::jsonb, true), "
                    "'{budget_reconciliation}', %s::jsonb, true)")
    assert ct._wrap_jsonb_set_expr("x", []) == "x"


# ── GAP-C3: expand rebuild inline ────────────────────────────────────────────
def test_c3_expand_rebuilds_lists_inline():
    start = _PL.index("def _apply_recipe_expansion")
    body = _PL[start:start + 12000]
    assert "_expand_list_dirty" in body, "falta el flag de dirty del append de veg"
    assert re.search(r'_rebuild_plan_shopping_lists_inline\(\s*plan_data_fresh', body), \
        "expand debe invocar el rebuild inline de listas"
    assert 'surface="recipe_expand"' in body


# ── GAP-04: re-agregación post review-patch ──────────────────────────────────
def test_gap04_review_patch_reaggregates():
    assert "_review_patch_removed_ingredients" in _GO
    # el patch acumula los removidos
    patch_start = _GO.index("def _auto_patch_ingredient_coherence")
    patch_body = _GO[patch_start:patch_start + 6000]
    assert "_review_patch_removed_ingredients" in patch_body
    # el seam re-agrega cuando no hubo swap
    # [P1-CRITIQUE-SLOT-PARITY · 2026-07-11] 400→1600: el re-cierre de banda
    # (P1-POST-PATCH-BAND-RECLOSE) vive entre el if y la re-agregación.
    assert re.search(
        r"not _swapped[\s\S]{0,300}_review_patch_removed_ingredients[\s\S]{0,1600}_recompute_aggregates_after_swap",
        _GO,
    ), "el seam debe re-agregar listas si el review-patch removió ingredientes sin swap"


# ── GAP-06: partial_pricing ──────────────────────────────────────────────────
def _summary(cycle_total, duration="weekly", priced=10, total=12):
    return {"active_duration": duration,
            "by_duration": {duration: {"cycle_total_rd": cycle_total,
                                       "items_priced": priced, "items_total": total}}}


def test_gap06_partial_pricing_flag(nc):
    ref = {"tier": "custom", "basis": "custom", "reference_rd": 10000, "floor_rd": 4000}
    low_cov = nc.reconcile_budget_with_cost(ref, _summary(9000, priced=5, total=28))
    assert low_cov["partial_pricing"] is True
    assert low_cov["price_coverage"] == round(5 / 28, 3)
    assert low_cov["status"] == "dentro", "el flag NO cambia el status"
    high_cov = nc.reconcile_budget_with_cost(ref, _summary(9000, priced=27, total=28))
    assert "partial_pricing" not in high_cov


def test_gap06_coverage_knob(nc, monkeypatch):
    monkeypatch.setenv("MEALFIT_BUDGET_RECONCILE_MIN_COVERAGE", "0.95")
    ref = {"tier": "low", "basis": "low", "reference_rd": 9000, "floor_rd": 4000}
    r = nc.reconcile_budget_with_cost(ref, _summary(5000, priced=25, total=28))
    assert r["partial_pricing"] is True  # 0.89 < 0.95


def test_gap06_frontend_consumes_flag():
    assert "partial_pricing" in _DASH and "price_coverage" in _DASH, \
        "el Dashboard debe consumir el caveat de cobertura parcial"


# ── GAP-07: budget tier en chat-modify ───────────────────────────────────────
def test_gap07_chatmodify_budget_wired():
    fn = _TO.index("def execute_modify_single_meal")
    body = _TO[fn:]
    assert "budget_prefers_economy" in body
    assert "build_budget_context" in body
    assert "_apply_budget_cheapen_pass" in body, "falta el backstop cheapen-pass en chat-modify"
    # cheapen corre solo con expansión + economía, antes de colocar el meal
    assert re.search(r"allow_pantry_expansion and _bud_economy", body)
    # hidratación del budget desde health_profile
    assert re.search(r'"budget",\s*"budgetAmount",\s*"budgetCurrency"', body)


# ── GAP-08: cheapen-pass reescribe pasos ─────────────────────────────────────
def test_gap08_cheapen_rewrites_recipe_steps_anchor():
    cp = _GO.index("def _apply_budget_cheapen_pass")
    body = _GO[cp:cp + 9000]
    assert "_rewrite_recipe_steps_after_subs" in body, \
        "el cheapen-pass debe reescribir los pasos tras sustituir (GAP-08)"
    assert "SUBST_RECIPE_REWRITE_ENABLED" in body


def test_gap08_cheapen_rewrites_steps_functional(go, monkeypatch):
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_PASS_ENABLED", True)
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_MAX_SUBS", 3)
    monkeypatch.setattr(go, "_budget_build_master_price_map",
                        lambda: {"salmon": 600.0, "filete de pescado blanco": 127.0})
    import nutrition_calculator as _nc
    monkeypatch.setattr(_nc, "budget_prefers_economy", lambda fd: True)
    days = [{"meals": [{
        "name": "Salmón a la Plancha",
        "ingredients": ["150g de salmón", "100g de brócoli"],
        "recipe": ["Mise en place: pesa todo.", "El Toque de Fuego: sella el salmón 4 min por lado.",
                   "Montaje: sirve."],
    }]}]
    n = go._apply_budget_cheapen_pass(days, {"budget": "low"})
    assert n >= 1
    steps = " ".join(days[0]["meals"][0]["recipe"]).lower()
    assert "salmón" not in steps and "salmon" not in steps, \
        f"los pasos siguen mencionando el premium sustituido: {steps!r}"
    assert "pescado blanco" in steps


# ── GAP-09: backstop tiempo/temp ─────────────────────────────────────────────
def test_gap09a_horno_token_removed(go):
    assert not go._CONTRACT_TIME_RE.search("cocina el pollo en el horno hasta dorar"), \
        "'horno' suelto NO debe satisfacer el contrato de tiempo/temp"
    assert go._CONTRACT_TIME_RE.search("hornea 18-20 min a 180 °C")
    assert go._CONTRACT_TIME_RE.search("cocina a fuego medio")


def test_gap09a_horno_gets_default_injected(go):
    meal = {"name": "Pollo al Horno",
            "ingredients": ["120g de pollo"],
            "recipe": ["Mise en place: prepara.", "El Toque de Fuego: cocina el pollo en el horno hasta dorar.",
                       "Montaje: sirve."]}
    changed = go._inject_recipe_time_temp_defaults(meal)
    assert changed, "el paso de horno sin tiempo debe recibir el default"
    assert "180 °C" in " ".join(meal["recipe"])


def test_gap09b_persist_boundary_runs_timetemp():
    fb = _GO.index("def finalize_plan_data_coherence")
    body = _GO[fb:fb + 16000]
    assert "_inject_recipe_time_temp_defaults" in body, \
        "el persist boundary debe correr el backstop tiempo/temp (paridad con los otros 13 guards)"
    assert "timetemp=" in body


# ── GAP-10: qty-sync en regen-day ────────────────────────────────────────────
def test_gap10_regen_day_syncs_step_quantities():
    mu = _PL.index("def _day_mutator")
    body = _PL[mu:mu + 8000]
    assert "_sync_recipe_step_quantities" in body, "regen-day debe re-sincronizar pasos"
    # después del trim renal y antes del pop de listas
    renal = body.index("renal_protein_trim_for_update")
    sync = body.index("_sync_recipe_step_quantities")
    pop = body.index('"aggregated_shopping_list_monthly"')
    assert renal < sync < pop, "el qty-sync debe correr tras el trim renal y antes del strip de listas"


# ── GAP-11: same-day protein en chat-modify ──────────────────────────────────
def test_gap11_detector_mechanics():
    import tools as t
    assert t._detect_same_day_protein_repeat("Pechuga a la Plancha", ["Pollo Guisado con Yuca"]) == "pollo"
    assert t._detect_same_day_protein_repeat("Fresas con Yogur", ["Bistec de Res"]) is None, \
        "anti falso-positivo 'res' en 'fresas'"
    assert t._detect_same_day_protein_repeat("Bistec Encebollado", ["Avena con Frutas"]) is None


def test_gap11_backstop_and_advisory_wired():
    assert "SAME_DAY_PROTEIN_REPEAT" in _TO, "falta el ValueError del backstop en chat-modify"
    assert "_same_day_protein_advisory" in _TO, "falta el advisory final (espejo de swap)"
    assert "RETRY VARIEDAD DEL DÍA" in _TO
    # skip pantry-strict (mismo criterio que swap)
    assert re.search(r"not \(clean_ingredients and not allow_pantry_expansion\)[\s\S]{0,2500}SAME_DAY_PROTEIN_REPEAT", _TO)


# ── GAP-13: raw-staple en swap ───────────────────────────────────────────────
def test_gap13_swap_raw_staple_pressure():
    assert "MEALFIT_SWAP_RAW_STAPLE_PRESSURE" in _AG, "falta el knob del gate raw-staple en swap"
    assert re.search(r'MEALFIT_SWAP_RAW_STAPLE_PRESSURE",\s*"true"', _AG), "default ON en el os.environ.get"
    assert 'ValueError("RAW_STAPLE: "' in _AG
    assert "RETRY PLATO TRANSFORMADO" in _AG
    # orden: después de dish-quality, antes de same-day variety
    dq = _AG.index("P2-UPDATE-DISHQUALITY-PRESSURE")
    rs = _AG.index("MEALFIT_SWAP_RAW_STAPLE_PRESSURE")
    sd = _AG.index("P2-UPDATE-SAMEDAY-VARIETY")
    assert dq < rs < sd


# ── GAP-05: piso cocinable SHRINK ────────────────────────────────────────────
def test_gap05_knob_defaults(go):
    assert go.PORTION_SHRINK_FLOOR_ENABLED is True or isinstance(go.PORTION_SHRINK_FLOOR_ENABLED, bool)
    assert 5.0 <= go.PORTION_SHRINK_FLOOR_G <= 40.0


def test_gap05_bump_with_headroom(go, monkeypatch):
    monkeypatch.setattr(go, "_ingredient_macro_group", lambda s, db: "protein")
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: True)
    days = [{"day": 1, "meals": [{"name": "T", "cals": 400,
        "ingredients": ["5g de pechuga de pollo", "80g de arroz blanco", "10g de aceite de oliva", "sal al gusto"],
        "ingredients_raw": ["5g de pechuga de pollo", "80g de arroz blanco", "10g de aceite de oliva", "sal al gusto"],
        "recipe": [], "protein": 10, "carbs": 60, "fats": 12}]}]
    n = go._floor_subservible_portions(days, day_kcal_target=2000, db=_FakeDB())
    m = days[0]["meals"][0]
    assert n == 1
    assert m["ingredients"][0].startswith("15g"), "la línea sub-servible debe bumpearse al piso"
    assert m["ingredients_raw"][0].startswith("15g"), "lockstep de ingredients_raw"
    assert m["ingredients"][2] == "10g de aceite de oliva", "el aceite (cucharadita legítima) queda exento"
    assert m.get("_portion_floor_adjusted") is True


def test_gap05_drop_without_headroom(go, monkeypatch):
    monkeypatch.setattr(go, "_ingredient_macro_group", lambda s, db: "protein")
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: True)
    days = [{"day": 1, "meals": [{"name": "T2", "cals": 2000,
        "ingredients": ["5g de queso mozzarella", "100g de arroz", "120g de pollo"],
        "recipe": [], "protein": 40, "carbs": 80, "fats": 30}]}]
    n = go._floor_subservible_portions(days, day_kcal_target=2000, db=_FakeDB())
    assert n == 1
    assert len(days[0]["meals"][0]["ingredients"]) == 2, "sin headroom la línea se dropea"


def test_gap05_never_leaves_meal_near_empty(go, monkeypatch):
    monkeypatch.setattr(go, "_ingredient_macro_group", lambda s, db: "protein")
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: True)
    days = [{"day": 1, "meals": [{"name": "T3", "cals": 2000,
        "ingredients": ["5g de queso", "10g de pollo"], "recipe": [],
        "protein": 5, "carbs": 5, "fats": 5}]}]
    go._floor_subservible_portions(days, day_kcal_target=2000, db=_FakeDB())
    assert len(days[0]["meals"][0]["ingredients"]) == 2, "jamás dejar la comida casi vacía por drops"


def test_gap05_wired_in_assemble():
    assert len(re.findall(r"_floor_subservible_portions\(", _GO)) >= 3, \
        "el shrink-floor debe correr en assemble (post-quantize) y post-recheck"


# ── GAP-03: _day_fallback honesto ────────────────────────────────────────────
def test_gap03_collector_mechanics(go):
    plan = {"days": [{"day": 2, "_day_fallback": True}, {"day": 1}, {"day": 3, "_day_fallback": False}]}
    assert go._collect_day_fallback_days(plan) == [2]
    assert go._collect_day_fallback_days({}) == []


def test_gap03_wired_in_should_retry_and_surgical():
    assert "DAY_FALLBACK_HONESTY" in _GO
    assert 'reason="day_fallback"' in _GO, "el residual debe marcar _quality_degraded(day_fallback)"
    # surgical incluye los días fallback en su colecta
    sg = _GO.index("marker_day_nums = _collect_unresolved_marker_days(plan_result)")
    assert "_collect_day_fallback_days" in _GO[sg:sg + 800]
    # el prompt del corrector excluye el marker del JSON
    assert '"_day_fallback"' in _GO


# ── GAP-M1: per_day_ceilings ─────────────────────────────────────────────────
def test_gapm1_report_has_per_day_ceilings():
    assert "per_day_ceilings" in _MI
    assert "MEALFIT_MICRO_PERDAY_CEILING_RATIO" in _MI
    # el bloque evalúa SOLO targets con ceiling (el de floors los salta)
    assert re.search(r'if "ceiling" not in _ptgt:\s*\n\s*continue', _MI)


def test_gapm1_consumer_banner_only():
    assert "micro_worst_day_ceiling" in _GO, "falta el reason del banner de techos worst-day"
    # jamás retry: el reason vive en _maybe_mark_panel_degraded (banner) — no en gates de retry
    idx = _GO.index("micro_worst_day_ceiling")
    ctx = _GO[max(0, idx - 3000):idx + 500]
    assert "_maybe_mark_panel_degraded" in ctx or "MICRO_PERDAY_DEGRADE_ENABLED" in ctx


# ── GAP-M2: micro-closer per-día ─────────────────────────────────────────────
def test_gapm2_knob_born_off(go):
    # [P1-GATES-FLIP-ON · 2026-07-03] (audit v6 · P1-4) el playbook medir→actuar se COMPLETÓ:
    # el gym baseline midió 9/20 planes con worst-day floor → default de CÓDIGO ahora True.
    # El runtime de la suite sigue OFF (baseline conftest) — este assert verifica ambos lados.
    import re as _re_m2
    _go_src_m2 = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert _re_m2.search(r'MICRO_CLOSER_PERDAY_ENABLED\s*=\s*_env_bool\("MEALFIT_MICRO_CLOSER_PERDAY",\s*True\)',
                         _go_src_m2), "default de código ON tras P1-GATES-FLIP-ON"
    assert go.MICRO_CLOSER_PERDAY_ENABLED is False, \
        "baseline de la suite OFF (conftest) — los fixtures históricos asumen closer promedio-only"


def test_gapm2_floors_extension_wired():
    assert "MICRO_CLOSER_PERDAY_ENABLED" in _GO
    cl = _GO.index("def _close_micro_gaps_for_plan")
    body = _GO[cl:cl + 14000]
    assert "per_day_floors" in body, "el closer debe poder leer el worst-day del panel"
    assert "_MICRO_CLOSER_RENAL_EXCLUDED" in body, "exclusiones clínicas compartidas (SSOT)"
    # los umbrales del detector NO se tocan
    assert "MEALFIT_MICRO_PERDAY_RATIO" not in body


def test_gapm2_renal_excluded_ssot(go):
    assert "potassium_mg" in go._MICRO_CLOSER_RENAL_EXCLUDED
    assert "calcium_mg" not in go._MICRO_CLOSER_RENAL_EXCLUDED, \
        "calcio sigue cerrable en renal (comportamiento previo)"
