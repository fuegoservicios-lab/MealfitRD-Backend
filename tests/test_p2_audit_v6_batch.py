"""[P2-AUDIT-V6-BATCH · 2026-07-03] Los 10 P2 de la auditoría v6 (objetivo ampliado).

  P2-A piso sub-servible en finalizador per-meal (updates/expand) + persist boundary (chunks S2+).
  P2-B plausibilidad de tiempos/temperaturas PRESENTES (clamp por técnica, no solo ausencias).
  P2-C contract-lint per-meal en updates + boundary; dish_quality_report en chunk T1.
  P2-D garantía cena-sin-arroz: el autofix compuesto corre en intento final AUNQUE haya hard.
  P2-E KPI transform_ratio (dish_quality_report) + mínimo diario transformado en dish_library.
  P2-F inspiración de biblioteca + variedad cross-day en swap; inspiración en chat-modify.
  P2-G TASTE v2: señales POSITIVAS (swap-to / chat-request) + línea PREFIERE en el contexto.
  P2-H sugerencias de ahorro brand-aware (respetan user_brand_preferences).
  P2-I transparencia de la referencia del tier (endpoint budget-floor + form + banner + PDF).
  P2-J chips frontend para _name_honesty_degraded y _recipe_contract_advisory.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PL = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TO = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_AG = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_CT = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_DL = (_BACKEND / "dish_library.py").read_text(encoding="utf-8")
_TM = (_BACKEND / "taste_model.py").read_text(encoding="utf-8")
_SC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
_NC = (_BACKEND / "nutrition_calculator.py").read_text(encoding="utf-8")
_FRONT = _BACKEND.parent / "frontend" / "src"


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P2-AUDIT-V6-BATCH" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-03"


# ════════════════════════════════════════════════════════════════════════════
# P2-A — piso sub-servible en updates/expand + persist boundary
# ════════════════════════════════════════════════════════════════════════════
def test_p2a_floor_in_single_meal_finalizer():
    fin = _GO.index("def finalize_single_meal_recipe_coherence")
    body = _GO[fin:fin + 20000]
    assert "_floor_subservible_portions(_wrap" in body, \
        "el finalizador per-meal debe correr el PISO sub-servible (antes solo el techo)"
    # orden espejo de assemble: quantize (creador del residuo) ANTES del floor
    assert body.index("_apply_portion_quantization({\"days\": _wrap}") < body.index("_floor_subservible_portions(_wrap")


def test_p2a_floor_in_persist_boundary():
    fpc = _GO.index("def finalize_plan_data_coherence")
    body = _GO[fpc:fpc + 16000]
    assert "shrink_floor=" in body, "el persist boundary (chunks S2+/degradado) debe correr el piso"


# ════════════════════════════════════════════════════════════════════════════
# P2-B — plausibilidad tiempo/temp (clamp de outliers PRESENTES)
# ════════════════════════════════════════════════════════════════════════════
def test_p2b_knob_default_on():
    assert re.search(
        r'RECIPE_TIMETEMP_PLAUSIBILITY_ENABLED\s*=\s*_env_bool\("MEALFIT_RECIPE_TIMETEMP_PLAUSIBILITY",\s*True\)', _GO
    )


def test_p2b_clamps_absurd_time_and_temp(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "RECIPE_TIMETEMP_PLAUSIBILITY_ENABLED", True)
    meal = {"name": "Pollo al horno", "recipe": [
        "Mise en place: sazona.",
        "El Toque de Fuego: hornea 200 min a 450 °C.",
        "Montaje: sirve.",
    ]}
    assert go._clamp_recipe_time_temp_outliers(meal) is True
    step = meal["recipe"][1]
    assert "200 min" not in step, f"tiempo absurdo debe clamparse: {step}"
    assert "450 °C" not in step, f"temperatura industrial debe clamparse: {step}"
    assert "220 °C" in step
    assert meal.get("_recipe_timetemp_clamped") is True


def test_p2b_idempotent_on_valid_times(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "RECIPE_TIMETEMP_PLAUSIBILITY_ENABLED", True)
    meal = {"name": "Pollo al horno", "recipe": [
        "El Toque de Fuego: hornea 18-20 min a 180 °C.",
    ]}
    assert go._clamp_recipe_time_temp_outliers(meal) is False
    assert meal["recipe"][0] == "El Toque de Fuego: hornea 18-20 min a 180 °C."
    assert "_recipe_timetemp_clamped" not in meal


def test_p2b_sancocho_long_boil_not_clamped(monkeypatch):
    # hervores largos legítimos (sancocho 60-90 min) NO se tocan (techo hervir = 90)
    import graph_orchestrator as go
    monkeypatch.setattr(go, "RECIPE_TIMETEMP_PLAUSIBILITY_ENABLED", True)
    meal = {"name": "Sancocho", "recipe": ["El Toque de Fuego: hierve 60-90 min a fuego medio."]}
    assert go._clamp_recipe_time_temp_outliers(meal) is False


def test_p2b_called_from_backstop_seam():
    inj = _GO.index("def _inject_recipe_time_temp_defaults")
    body = _GO[inj:inj + 2500]
    assert "_clamp_recipe_time_temp_outliers(meal)" in body, \
        "el clamp debe correr en el mismo seam del backstop (cubre las 4 superficies)"


# ════════════════════════════════════════════════════════════════════════════
# P2-C — contract-lint en updates/boundary + dish-quality en chunk T1
# ════════════════════════════════════════════════════════════════════════════
def test_p2c_contract_advisory_in_finalizer_and_boundary():
    fin = _GO.index("def finalize_single_meal_recipe_coherence")
    assert '_recipe_contract_advisory"] = _rc_issues[:4]' in _GO[fin:fin + 22000]
    fpc = _GO.index("def finalize_plan_data_coherence")
    # [2026-07-06] ventana 17000→26000: el boundary creció con los seams de la
    # madrugada 07-05/06 (cured-ghost, mise-split, note-align, tracer) y el
    # callsite quedó a offset ~18.9k del def — drift de ventana, no ausencia.
    assert "contract_advisory=" in _GO[fpc:fpc + 26000]


def test_p2c_dish_quality_report_in_chunk_t1():
    assert "dish-quality chunk T1" in _CT or "(P2-C) dish_quality_report" in _CT
    idx = _CT.index("[P2-AUDIT-V6-BATCH · 2026-07-03] (P2-C)")
    blk = _CT[idx:idx + 900]
    assert "compute_dish_quality_report" in blk and "DISH_QUALITY_TELEMETRY_ENABLED" in blk


# ════════════════════════════════════════════════════════════════════════════
# P2-D — garantía cena-sin-arroz en intento final (aunque haya hard)
# ════════════════════════════════════════════════════════════════════════════
def test_p2d_autofix_runs_before_hard_split():
    gate = _GO.index("_sa_is_final = _sa_attempt >= MAX_ATTEMPTS")
    blk = _GO[gate:gate + 4000]
    idx_autofix = blk.index("_night_rice_autofix(plan.get(\"days\", []), compound=True)")
    idx_hard = blk.index("_sa_has_hard = any(")
    assert idx_autofix < idx_hard, \
        "el autofix debe correr ANTES del split hard/soft — la mezcla hard+soft en intento final " \
        "entregaba el plan degradado CON el arroz nocturno limpiable"
    assert "_sa_is_final and NIGHT_RICE_COMPOUND_FINAL" in blk


# ════════════════════════════════════════════════════════════════════════════
# P2-E — KPI transform_ratio + mínimo transformado en la biblioteca
# ════════════════════════════════════════════════════════════════════════════
def test_p2e_transform_ratio_in_dish_quality_report():
    import graph_orchestrator as go
    plan = {"days": [{"day": 1, "meals": [
        {"name": "Panqueques de avena y guineo", "ingredients": ["100g de avena"],
         "recipe": ["Mise en place: mezcla.", "El Toque de Fuego: 2-3 min por lado a fuego medio.", "Montaje: sirve."]},
        {"name": "Pollo a la plancha con arroz", "ingredients": ["150g de pollo"],
         "recipe": ["Mise en place: sazona.", "El Toque de Fuego: plancha 3-4 min por lado.", "Montaje: sirve."]},
    ]}]}
    rep = go.compute_dish_quality_report(plan)
    assert rep.get("transform_meals") == 1
    assert rep.get("transform_ratio") == 0.5


def test_p2e_library_transform_min_knob_and_nudge():
    assert re.search(
        r'DISH_LIBRARY_TRANSFORM_MIN\s*=\s*_env_int\("MEALFIT_DISH_LIBRARY_TRANSFORM_MIN",\s*1', _DL
    )
    assert "plato(s) TRANSFORMADO(s)" in _DL


def test_p2e_nudge_in_context(monkeypatch):
    import dish_library as dl
    monkeypatch.setattr(dl, "DISH_LIBRARY_ENABLED", True)
    monkeypatch.setattr(dl, "DISH_LIBRARY_TRANSFORM_MIN", 1)
    ctx = dl.build_dish_library_context({"protein_pool": ["Pollo"], "meal_types": ["Cena"]}, 1)
    assert "TRANSFORMADO" in ctx and "al menos 1" in ctx


# ════════════════════════════════════════════════════════════════════════════
# P2-F — inspiración + cross-day en superficies de update
# ════════════════════════════════════════════════════════════════════════════
def test_p2f_swap_inspiration_helper(monkeypatch):
    import dish_library as dl
    monkeypatch.setattr(dl, "DISH_LIBRARY_ENABLED", True)
    ctx = dl.build_swap_inspiration_context("Cena", seed=3)
    assert "INSPIRACIÓN" in ctx and "ELIGE Y ADAPTA" in ctx
    assert dl.build_swap_inspiration_context("SlotInventado") == ""
    monkeypatch.setattr(dl, "DISH_LIBRARY_ENABLED", False)
    assert dl.build_swap_inspiration_context("Cena") == ""


def test_p2f_wired_in_swap_and_chat():
    # swap: cross-day + inspiración
    assert "def _cross_day_meal_names_for_swap" in _PL
    assert 'data["cross_day_meal_names"] = _cross_day' in _PL
    assert "cross_day_meal_names" in _AG and "VARIEDAD ENTRE DÍAS" in _AG
    assert "build_swap_inspiration_context as _bsi_swap" in _AG
    # chat-modify: inspiración (el cross-day ya existía vía P2-CHATMODIFY-CROSS-DAY-VARIETY)
    assert "build_swap_inspiration_context as _bsi_cm" in _TO


# ════════════════════════════════════════════════════════════════════════════
# P2-G — TASTE v2: positivas
# ════════════════════════════════════════════════════════════════════════════
def test_p2g_knobs_default_on():
    assert re.search(r'TASTE_POSITIVE_ENABLED\s*=\s*_env_bool\("MEALFIT_TASTE_POSITIVE",\s*True\)', _TM)


def test_p2g_record_swap_to(monkeypatch):
    import taste_model as tm
    monkeypatch.setattr(tm, "TASTE_MODEL_ENABLED", True)
    monkeypatch.setattr(tm, "TASTE_POSITIVE_ENABLED", True)
    recorded = []
    monkeypatch.setattr(tm, "_record", lambda uid, tok, sig, w, src: recorded.append((tok, sig, w)) or True)
    uid = "11111111-2222-3333-4444-555555555555"
    assert tm.record_swap_to(uid, "Pollo guisado", "Camarones al ajillo") is True
    assert recorded and recorded[0][1] == "swap_to" and recorded[0][2] == -1.0
    # misma proteína → sin señal (mantenerla por inercia no es elegirla)
    recorded.clear()
    assert tm.record_swap_to(uid, "Pollo guisado", "Pollo a la plancha") is False
    assert not recorded


def test_p2g_chat_request_positive_with_negation_guard(monkeypatch):
    import taste_model as tm
    monkeypatch.setattr(tm, "TASTE_MODEL_ENABLED", True)
    monkeypatch.setattr(tm, "TASTE_POSITIVE_ENABLED", True)
    recorded = []
    monkeypatch.setattr(tm, "_record", lambda uid, tok, sig, w, src: recorded.append((tok, sig, w)) or True)
    uid = "11111111-2222-3333-4444-555555555555"
    assert tm.record_chat_request_positive(uid, "Camarones al ajillo", "ponme camarones por favor") is True
    assert recorded and recorded[0][1] == "chat_positive" and recorded[0][2] == -2.0
    # negación que precede al token → jamás positivo
    recorded.clear()
    assert tm.record_chat_request_positive(uid, "Pollo guisado", "no quiero pollo") is False
    assert not recorded


def test_p2g_context_includes_prefiere_line(monkeypatch):
    import taste_model as tm
    monkeypatch.setattr(tm, "negative_tokens_for_user", lambda uid: ["res"])
    monkeypatch.setattr(tm, "positive_tokens_for_user", lambda uid: ["camarones"])
    ctx = tm.build_taste_context("x")
    assert "res" in ctx and "camarones" in ctx and "PREFIÉRELAS" in ctx
    # sin señal alguna → '' (prompt-cache)
    monkeypatch.setattr(tm, "negative_tokens_for_user", lambda uid: [])
    monkeypatch.setattr(tm, "positive_tokens_for_user", lambda uid: [])
    assert tm.build_taste_context("x") == ""


def test_p2g_wired_in_persist_sites():
    assert "record_swap_to" in _PL, "swap-persist debe registrar la señal positiva"
    assert "record_chat_request_positive" in _TO, "chat-modify debe registrar el pedido explícito"


# ════════════════════════════════════════════════════════════════════════════
# P2-H — sugerencias brand-aware
# ════════════════════════════════════════════════════════════════════════════
def test_p2h_suggestions_skip_user_branded_items(monkeypatch):
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "fetch_brand_pref_packages",
                        lambda uid: {"salmon": {"grams": 454, "price": 500, "label": "x", "unit": "paquete"}})
    monkeypatch.setattr(sc, "cheapest_supermarket_variant",
                        lambda name: {"brand": "MarcaX", "presentation": "1 lb", "price_rd": 99.0})
    weekly = [{"name": "Salmón", "estimated_cost_rd": 900}, {"name": "Pollo", "estimated_cost_rd": 300}]
    sugs = sc.build_budget_suggestions(weekly, user_id="11111111-2222-3333-4444-555555555555")
    # [P1-BUDGET-BRAND-PREMIUM · 2026-07-07] build_budget_suggestions puede prepender un resumen
    # {"type":"marca_premium_total", ...} SIN clave "item" → filtrar a las sugerencias per-ítem.
    items = [s["item"] for s in sugs if "item" in s]
    assert "Salmón" not in items, "el usuario YA eligió marca de salmón — no sugerir contra su decisión"
    assert "Pollo" in items
    # sin user_id → comportamiento previo (ambos)
    sugs_anon = sc.build_budget_suggestions(weekly)
    assert {"Salmón", "Pollo"} == {s["item"] for s in sugs_anon if "item" in s}


def test_p2h_user_id_wired_through_refresh_and_callsites():
    assert "user_id=None) -> None" in _NC.replace("\n", " ") or "user_id=None" in _NC[_NC.index("def refresh_budget_reconciliation"):_NC.index("def refresh_budget_reconciliation") + 300]
    assert "_p1b_rbr(plan_data_fresh, active_household=household_size, user_id=verified_user_id)" in _PL
    assert "_rbr_il(plan_data, user_id=user_id)" in _PL
    assert "_p1b_rbr_cm(plan_data_fresh, user_id=user_id)" in _TO
    # assemble (1ª pasada refactorizada al helper) + convergencia pasan _uid
    assert "_p1b_bbs(" in _GO and "user_id=_uid" in _GO


# ════════════════════════════════════════════════════════════════════════════
# P2-I — transparencia de la referencia del tier
# ════════════════════════════════════════════════════════════════════════════
def test_p2i_endpoint_exposes_tier_references():
    idx = _PL.index("def api_budget_floor")
    blk = _PL[idx:idx + 3500]
    assert '"tier_references"' in blk and "_budget_tier_band_factor" in blk


def test_p2i_frontend_consumes_reference():
    hook = (_FRONT / "hooks" / "useBudgetFloor.js").read_text(encoding="utf-8")
    assert "tier_references" in hook and "tierReferences" in hook
    form = (_FRONT / "components" / "assessment" / "questions" / "QBudget.jsx").read_text(encoding="utf-8")
    assert "referencia estimada" in form
    dash = (_FRONT / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
    assert "referencia estimada" in dash


# ════════════════════════════════════════════════════════════════════════════
# P2-J — chips frontend
# ════════════════════════════════════════════════════════════════════════════
def test_p2j_meal_advisories_chips():
    adv = (_FRONT / "utils" / "mealAdvisories.js").read_text(encoding="utf-8")
    assert "_name_honesty_degraded" in adv
    assert "_recipe_contract_advisory" in adv
