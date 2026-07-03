"""[P2-AUDIT-V2-BATCH · 2026-07-01] Lote de 18 P2 del audit objetivo v2 (grupos A-G).

Cada item tiene su tooltip-anchor propio en producción; este archivo ancla el lote completo
(1 sección por item, parser-based + funcionales puros sin DB). Item descartado tras verificación:
"regen-day hardcodea pantry_strict=True" (audit micros GAP-2) — /regenerate-day es pantry-constrained
POR DISEÑO (docstring del endpoint + gate de suficiencia): el closer NO debe "comprar más" cocinando
desde la nevera. Se documenta aquí para que un futuro audit no lo re-flagee.

A: expand (uncounted-addition, offcatalog-strip) · B: micros (carb-retrim updates, vitA/Se closer) ·
C: slots (arroz en ingredients del desayuno) · D: creatividad (rotación fallback, cron KPI, drop
telemetry, contrato pool↔precio) · E: recetas (undercook-note, household-sync, template-time,
qty-presence persist, english-lint) · F: paridad (chat modified_at, chat band-warn, swap-persist
finalize) · G: delivered_macros.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parent.parent
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_SHOP = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


# ───────────────────────────── Grupo A · expand ─────────────────────────────
def test_a1_expand_uncounted_guard_wired():
    assert "P2-EXPAND-UNCOUNTED-ADDITION" in _PLANS
    assert 'os.environ.get("MEALFIT_EXPAND_UNCOUNTED_GUARD", "true")' in _PLANS
    i_guard = _PLANS.find("P2-EXPAND-UNCOUNTED-ADDITION")
    i_charge = _PLANS.find('log_api_usage(user_id, "llm_recipe_expand")')
    assert -1 < i_guard < i_charge, "el detector de calorías no contadas debe correr ANTES de cobrar"


def test_a2_expand_offcatalog_strip_wired():
    assert "P2-EXPAND-OFFCATALOG-STRIP" in _PLANS
    assert "_strip_offcatalog_condiments_from_recipe as _oc_strip_exp" in _PLANS


# ───────────────────────────── Grupo B · micros ─────────────────────────────
def test_b1_regen_day_pantry_strict_is_by_design():
    """Descartado tras verificación: regen-day es pantry-constrained POR DISEÑO (no gap)."""
    assert "pantry-constrained" in _PLANS and "pantry_strict=True" in _PLANS


def test_b2_carb_retrim_in_both_update_surfaces():
    assert _PLANS.count("P2-UPDATES-CARB-RETRIM") >= 1, "falta el re-trim en swap-persist"
    assert _TOOLS.count("P2-UPDATES-CARB-RETRIM") >= 1, "falta el re-trim en chat-modify"
    # el trim corre ANTES del quantize (mismo orden que assemble)
    for src, qz in ((_PLANS, "_qz_swap(plan_data, _qdb_swap)"), (_TOOLS, "_qz_cm(plan_data_fresh, _qdb_cm)")):
        i_trim = src.find("P2-UPDATES-CARB-RETRIM")
        i_qz = src.find(qz)
        assert -1 < i_trim < i_qz, "el re-trim debe correr antes del quantize"


def test_b3_vita_selenium_closer_keys_with_ul():
    assert "vit_a_mcg" in go._MICRO_CLOSER_KEYS and "selenium_mcg" in go._MICRO_CLOSER_KEYS
    assert go._MICRO_CLOSER_UL.get("vit_a_mcg") == 3000.0, "vit A SIN techo UL sería hepatotóxica/teratogénica"
    assert go._MICRO_CLOSER_UL.get("selenium_mcg") == 400.0
    assert go._MICRO_CLOSER_INGREDIENT_KEY.get("vit_a_mcg") == "vit_a_mcg"
    assert go._MICRO_CLOSER_INGREDIENT_KEY.get("selenium_mcg") == "selenium_mcg"


def test_b3_vita_selenium_renal_skip():
    # [P2-AUDIT-V5-BATCH · 2026-07-02] (GAP-M2) el tuple inline del skip renal se extrajo al
    # SSOT `_MICRO_CLOSER_RENAL_EXCLUDED` (compartido con el targeting per-día del closer);
    # el anchor verifica la misma invariante sobre el frozenset + su uso en el loop de floors.
    assert "vit_a_mcg" in go._MICRO_CLOSER_RENAL_EXCLUDED and \
        "selenium_mcg" in go._MICRO_CLOSER_RENAL_EXCLUDED, \
        "vit A se acumula en ERC — debe estar en el skip renal del closer"
    assert "if _renal and k in _MICRO_CLOSER_RENAL_EXCLUDED" in _GRAPH, \
        "el loop de floors del closer debe usar el SSOT del skip renal"


# ───────────────────────────── Grupo C · slots ─────────────────────────────
def test_c1_breakfast_rice_in_ingredients_detected():
    from constants import slot_ingredient_violations
    v = slot_ingredient_violations(["150g arroz blanco", "1 huevo"], "Desayuno")
    assert v and v[0]["hard"] is True, "arroz en ingredients del desayuno debe ser violación hard"
    assert slot_ingredient_violations(["100g harina de arroz"], "Desayuno") == [], \
        "exclusión SSOT (harina de arroz) ignorada — rompería panqueques de harina de arroz"
    assert slot_ingredient_violations(["150g arroz blanco"], "Cena") == [], \
        "cena NO va por este detector (tiene su autofix _night_rice_autofix)"
    assert slot_ingredient_violations([], "Desayuno") == []


def test_c1_wired_in_s1_and_update_backstop():
    assert _GRAPH.count("slot_ingredient_violations") >= 2, \
        "el pase ingredient-level debe correr en _detect_slot_appropriateness (S1) Y en el backstop de updates"


# ──────────────────────────── Grupo D · creatividad ────────────────────────────
def test_d1_fallback_rotates_by_day():
    pool = [("Plato A", frozenset(), "a", ["x"]),
            ("Plato B", frozenset(), "b", ["y"]),
            ("Plato C", frozenset({"pollo"}), "c", ["z"]),
            ("Neutral", frozenset(), "n", ["w"])]
    assert go._select_safe_fallback_meal(pool, frozenset(), day_number=1)[0] == "Plato A"
    assert go._select_safe_fallback_meal(pool, frozenset(), day_number=2)[0] == "Plato B"
    # día 3 con 'pollo' restringido → el pool seguro tiene 3 → rota al tercero seguro (Neutral)
    assert go._select_safe_fallback_meal(pool, frozenset({"pollo"}), day_number=3)[0] == "Neutral"
    # determinista: mismo día → misma plantilla
    assert go._select_safe_fallback_meal(pool, frozenset(), day_number=2)[0] == "Plato B"


def test_d2_creativity_kpi_cron_registered():
    assert "def _creativity_kpi_job" in _CRON
    assert 'id="creativity_kpi_job"' in _CRON, "el cron no está registrado en register_plan_chunk_scheduler"
    assert "MEALFIT_CREATIVITY_KPI_INTERVAL_MIN" in _CRON
    assert "snapshot_and_reset_verified_only_drops" in _CRON


def test_d3_verified_drop_telemetry_sink():
    from shopping_calculator import record_verified_only_drop, snapshot_and_reset_verified_only_drops
    snapshot_and_reset_verified_only_drops()  # limpiar estado previo
    record_verified_only_drop("Salsa Inventada")
    record_verified_only_drop("salsa inventada ")
    record_verified_only_drop(None)
    snap = snapshot_and_reset_verified_only_drops()
    assert snap.get("salsa inventada") == 2
    assert snapshot_and_reset_verified_only_drops() == {}, "el snapshot debe resetear"
    assert "record_verified_only_drop(name)" in _SHOP, "el drop site no alimenta el sink"


def test_d4_pool_price_contract_script_exists():
    script = (_BACKEND / "scripts" / "check_pool_prices.py").read_text(encoding="utf-8")
    for pool in ("DOMINICAN_PROTEINS", "DOMINICAN_CARBS", "DOMINICAN_VEGGIES_FATS", "DOMINICAN_FRUITS"):
        assert pool in script, f"el contrato pool↔precio no cubre {pool}"
    assert "sys.exit(2)" in script
    # el fix de datos que el contrato encontró (alias 'pescado') vive en migración SSOT en ambos dirs
    mig = _BACKEND / "migrations" / "p2_pool_pescado_alias_2026_07_01.sql"
    assert mig.exists()
    root_mig = _BACKEND.parent / "migrations" / "p2_pool_pescado_alias_2026_07_01.sql"
    if root_mig.exists():
        assert mig.read_text(encoding="utf-8") == root_mig.read_text(encoding="utf-8"), "drift SSOT migrations"


# ───────────────────────────── Grupo E · recetas ─────────────────────────────
def test_e1_undercook_time_note():
    assert go.UNDERCOOK_TIME_NOTE_ENABLED is True
    plan = {"days": [{"meals": [{
        "name": "Pollo salteado",
        "ingredients": ["150g de pollo"],
        "recipe": ["Mise en place: corta el pollo.",
                   "El Toque de Fuego: cocina el pollo 2 minutos a fuego alto.",
                   "Montaje: sirve."],
    }]}]}
    n = go._apply_food_safety_fixes(plan)
    meal = plan["days"][0]["meals"][0]
    assert meal.get("_food_safety_undercook_time") is True, f"subcocción de pollo (2 min) sin nota (fixed={n})"
    assert any("debe cocinarse por completo" in str(s) for s in meal["recipe"])
    # idempotente
    before = list(meal["recipe"])
    go._apply_food_safety_fixes(plan)
    assert meal["recipe"] == before


def test_e1_no_false_positive_on_searing_or_long_times():
    plan = {"days": [{"meals": [{
        "name": "Pollo guisado", "ingredients": ["150g de pollo"],
        "recipe": ["El Toque de Fuego: sella el pollo 2 minutos y luego guisa 25 minutos."],
    }, {
        "name": "Pollo al horno", "ingredients": ["150g de pollo"],
        "recipe": ["El Toque de Fuego: hornea el pollo 30-35 minutos a 200°C."],
    }]}]}
    go._apply_food_safety_fixes(plan)
    for m in plan["days"][0]["meals"]:
        assert not m.get("_food_safety_undercook_time"), f"falso positivo en: {m['recipe']}"


def test_e2_step_household_sync():
    from humanize_ingredients import sync_recipe_steps_to_household
    meal = {
        "ingredients": ["¾ taza de arroz", "1 huevo"],
        "ingredients_raw": ["150 g de arroz", "1 huevo"],
        "recipe": ["Mise en place: pesa 150 g de arroz y lávalo.",
                   "⚠️ Seguridad alimentaria: nota con 150 g de arroz que no se toca."],
    }
    n = sync_recipe_steps_to_household(meal)
    assert n == 1
    assert "¾ taza de arroz (150 g)" in meal["recipe"][0]
    assert "y lávalo" in meal["recipe"][0], "el texto posterior a la mención debe preservarse"
    assert "¾ taza" not in meal["recipe"][1], "las notas ⚠ no se tocan"
    # idempotente (la mención reescrita ya no matchea "g de")
    assert sync_recipe_steps_to_household(meal) == 0


def test_e3_nonempty_template_has_concrete_time():
    meal = {"name": "Plato X", "ingredients": ["100g de pollo"], "recipe": []}
    assert go._ensure_nonempty_recipe(meal) is True
    fuego = next(s for s in meal["recipe"] if s.lower().startswith("el toque de fuego"))
    assert go._CONTRACT_TIME_RE.search(fuego), \
        "la plantilla del nonempty debe cumplir el contrato de tiempo que el sistema mide"


def test_e4_qty_presence_in_persist_boundary():
    assert "P2-QTY-PRESENCE-PERSIST" in _GRAPH
    i_fn = _GRAPH.find("def finalize_plan_data_coherence")
    i_qty = _GRAPH.find("_ensure_ingredient_quantities", i_fn)
    i_next = _GRAPH.find("def finalize_single_meal_recipe_coherence", i_fn)
    assert i_fn < i_qty < i_next, "el persist boundary no corre el qty-presence guard"


def test_e5_english_lint_in_contract():
    issues = go._recipe_step_contract_issues({"recipe": [
        "Mise en place: corta todo.",
        "El Toque de Fuego: add the chicken and cook until golden, 10 minutos.",
        "Montaje: sirve.",
    ]})
    assert any("inglés residual" in i for i in issues)
    clean = go._recipe_step_contract_issues({"recipe": [
        "Mise en place: corta todo.",
        "El Toque de Fuego: cocina el pollo 10 minutos a fuego medio.",
        "Montaje: sirve caliente.",
    ]})
    assert not any("inglés" in i for i in clean)


# ──────────────────────────── Grupo F · paridad updates ────────────────────────────
def test_f1_chat_modify_bumps_plan_modified_at():
    assert "P2-CHATMODIFY-MODIFIED-AT" in _TOOLS
    i_cb = _TOOLS.find("def _apply_meal_modification")
    i_bump = _TOOLS.find('plan_data_fresh["_plan_modified_at"]', i_cb)
    i_atomic = _TOOLS.find("update_plan_data_atomic", i_cb)
    assert i_cb < i_bump < i_atomic, "el sello debe setearse dentro del callback atómico"


def test_f2_chat_modify_surfaces_band_and_slot_warnings():
    assert "P2-CHATMODIFY-BAND-WARN" in _AGENT
    assert '_mod_meal_flags.get("_macro_band_low")' in _AGENT
    assert '_mod_meal_flags.get("_slot_advisory")' in _AGENT
    assert "coherence_warnings.extend(_band_warn_bits)" in _AGENT, \
        "los warnings deben viajar por el canal SSE que el frontend ya consume (toast)"


def test_f3_swap_persist_runs_finalizer_truthup_slot():
    assert "P2-SWAP-PERSIST-FINALIZE" in _PLANS
    i_mut = _PLANS.find("def _swap_mutator")
    seg = _PLANS[i_mut:i_mut + 4000]
    assert "finalize_single_meal_recipe_coherence as _fin_sp" in seg
    assert "_truth_up_meal_macros_from_strings as _tu_sp" in seg
    assert "slot_coherence_backstop_for_meal as _slot_sp" in seg
    assert '_persist_allergies' in seg, "el finalizer debe recibir las allergies hidratadas del perfil"


# ───────────────────────────── Grupo G · macros ─────────────────────────────
def test_g1_delivered_macros_computed():
    plan = {"days": [
        {"meals": [{"protein": 150, "carbs": 200, "fats": 60, "cals": 2000}]},
        {"meals": [{"protein": 130, "carbs": 180, "fats": 50, "cals": 1800}]},
    ]}
    assert go.refresh_delivered_macros(plan) is True
    assert plan["delivered_macros"] == {"protein": 140, "carbs": 190, "fats": 55}
    assert plan["delivered_calories"] == 1900
    assert go.refresh_delivered_macros({"days": []}) is False


def test_g1_wired_in_assemble_and_update_hook():
    assert _GRAPH.count("refresh_delivered_macros(") >= 3, \
        "delivered_macros debe refrescarse en assemble Y en el hook post-update (recompute micros)"


def test_marker_batch_anchor():
    assert "P2-AUDIT-V2-BATCH" in _GRAPH and "P2-AUDIT-V2-BATCH" in _PLANS and "P2-AUDIT-V2-BATCH" in _TOOLS
