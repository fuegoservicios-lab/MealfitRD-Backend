"""[P2-AUDIT-V3-BATCH · 2026-07-02] Los ~20 P2 del audit objetivo v3 (cierre del plan 1 P0 / 8 P1 / ~22 P2).

Grupos: A macros/micros core (closer raw-basis, vitK lever, fallback macro-engine, postquantize recheck,
delivered-refresh-always, band thresholds 0.45→0.60), B slots (desayuno guisados/legumbres, avena+fritura
en cena, desayuno-arroz autofix), C creatividad (transform clause planner, §15 SSOT, transform-base boost,
cross-day gate, sweet-fish/light-name advisories, pantry variety advisory), D expand (micro-recompute,
finalize-parity, quota-abort, guest-safety), E ops (alerta micro_estimado_bajo_chronic, NOT NULL micros
extendidos en Neon), F chat-modify target anchor.

Cerrados SIN código (verificados): P2-SWAP-PERSIST-TRUTHUP (ya existía vía P2-SWAP-PERSIST-FINALIZE) y
P2-REGEN-DAY-MICRO-CLOSER (pantry-strict es diseño documentado — la Nevera es el bound).
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(g.__file__).resolve().parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_AIH = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.search(r"(\d+)\s*g", str(s))
        return float(m.group(1)) if m else 150.0


def test_all_batch_markers_present():
    src_all = _GO + _PLANS + _TOOLS + _AGENT + _CRON + _AIH
    src_all += (_BACKEND / "constants.py").read_text(encoding="utf-8")
    src_all += (_BACKEND / "prompts" / "preferences.py").read_text(encoding="utf-8")
    src_all += (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
    for marker in ("P2-MICRO-CLOSER-RAW-BASIS", "P2-VITK-LEVER", "P2-FALLBACK-MACRO-ENGINE",
                   "P2-POSTQUANTIZE-RECHECK", "P2-DELIVERED-REFRESH-ALWAYS", "P2-DESAYUNO-ARROZ-AUTOFIX",
                   "P2-CROSSDAY-VARIETY-GATE", "P2-SWEETFISH-ADVISORY", "P2-LIGHTNAME-ADVISORY",
                   "P2-SLOT-DESAYUNO-GUISADOS", "P2-SLOT-CENA-AVENA", "P2-SLOT-CENA-FRITURA",
                   "P2-SLOT-SSOT-PROMPT", "P2-TRANSFORM-BASE-BOOST", "P2-CHATMOD-TARGET-ANCHOR",
                   "P2-PANTRY-VARIETY-ADVISORY", "P2-EXPAND-MICRO-RECOMPUTE", "P2-EXPAND-FINALIZE-PARITY",
                   "P2-EXPAND-QUOTA-ABORT", "P2-EXPAND-GUEST-SAFETY", "P2-MICRO-KPI-ALERT",
                   "P2-BAND-THRESHOLDS"):
        assert marker in src_all, f"marker {marker} ausente"


# ── A: macros/micros core ────────────────────────────────────────────────────

def test_closer_raw_basis_wired():
    i = _GO.index("def _close_micro_gaps_for_plan")
    body = _GO[i:i + 30000]
    assert "P2-MICRO-CLOSER-RAW-BASIS" in body
    assert "_raw_ok" in body   # medición sobre ingredients_raw alineado


def test_vitk_lever_with_guards():
    assert "vit_k_mcg" in g._MICRO_CLOSER_KEYS
    assert g._MICRO_CLOSER_INGREDIENT_KEY.get("vit_k_mcg") == "vit_k_mcg"
    assert "vit_k_mcg" not in g._MICRO_CLOSER_UL   # IOM no establece UL para vit K
    body = _GO[_GO.index("def _close_micro_gaps_for_plan"):][:30000]
    assert "detect_anticoagulant" in body, "skip warfarina obligatorio (consistencia INR)"
    assert '"vit_k_mcg" and _anticoag' in body


def test_fallback_macro_rebalance_knob_and_bariatric_exempt():
    assert g.FALLBACK_MACRO_REBALANCE_ENABLED is True
    i = _GO.index("FALLBACK_MACRO_REBALANCE_ENABLED and not _is_bar_fb")
    assert i > 0   # bariátrico exento (contrato de porciones)


def test_postquantize_recheck_between_quantize_and_qtysync():
    assert g.POSTQUANTIZE_RECHECK_ENABLED is True
    assert 0.01 <= g.POSTQUANTIZE_RECHECK_TOL <= 0.2
    i_q = _GO.index("quantize final: porciones humanas")
    i_rq = _GO.index("P2-POSTQUANTIZE-RECHECK] {_rq_fixed}", i_q - 2000)
    i_sync = _GO.index("re-sincronizadas post-quantize")
    assert i_q < i_rq < i_sync, "recheck DEBE correr tras el quantize y antes del qty-sync"


def test_delivered_refresh_always_both_sites():
    # sitio 1: entrega post-review S1; sitio 2: apply_update_band_parity (3 superficies de update)
    assert _GO.count("P2-DELIVERED-REFRESH-ALWAYS") >= 2


def test_band_thresholds_bumped():
    assert g.BAND_RETRY_THRESHOLD_MACROS_ONLY == 0.60
    assert g.BAND_SCORE_GATE_THRESHOLD_MACROS_ONLY == 0.60


# ── B: slots ─────────────────────────────────────────────────────────────────

def test_slot_desayuno_guisados_and_legumbres():
    from constants import slot_violations_for_meal_name as v
    assert v("Habichuelas Guisadas con Yuca", "desayuno")
    assert v("Pollo Guisado con Víveres", "desayuno")
    assert not v("Habichuelas con Dulce", "desayuno")     # postre criollo → exclude
    assert not v("Salami Guisado con Mangú", "desayuno")  # desayuno RD legítimo (token compuesto no matchea)
    assert not v("Habichuelas Guisadas", "almuerzo")      # regla solo desayuno


def test_slot_cena_avena_and_fritura():
    from constants import slot_violations_for_meal_name as v
    assert v("Avena con Frutas", "cena")
    assert not v("Empanizado de Avena con Pollo", "cena")   # exclude ingrediente-de-costra
    assert v("Pollo Frito con Tostones", "cena")
    assert v("Chicharrón de Cerdo con Yuca", "cena")
    assert not v("Tilapia a la Plancha con Tostones", "cena")  # tostones acompañante = legítimo
    assert not v("Chicharrón de Pollo", "merienda")            # merienda RD legítima (regla solo cena)


def test_breakfast_rice_autofix_functional(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Arroz con Huevo Frito",
                                  "ingredients": ["arroz blanco 100g", "2 unidades huevo"],
                                  "recipe": ["Cocina el arroz blanco."]}]}]
    assert g._breakfast_rice_autofix(days, db=_DB()) == 1
    meal = days[0]["meals"][0]
    assert "arroz" not in meal["name"].lower()
    assert not any("arroz" in str(i).lower() for i in meal["ingredients"])
    assert g._breakfast_rice_autofix(days, db=_DB()) == 0   # idempotente


def test_breakfast_rice_autofix_skips_compound_and_knob(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Moro de Gandules",
                                  "ingredients": ["moro 200g"]}]}]
    assert g._breakfast_rice_autofix(days, db=_DB()) == 0   # compuesto → al gate hard
    monkeypatch.setattr(g, "BREAKFAST_RICE_AUTOFIX_ENABLED", False)
    days2 = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Arroz Blanco",
                                   "ingredients": ["arroz 100g"]}]}]
    assert g._breakfast_rice_autofix(days2, db=_DB()) == 0  # rollback por knob


def test_breakfast_autofix_wired_in_assemble_and_clinical_layer():
    assert _GO.count("_breakfast_rice_autofix(") >= 3   # def + assemble + capa clínica


# ── C: creatividad ───────────────────────────────────────────────────────────

def test_crossday_dish_gate():
    plan = {"days": [{"day": d, "meals": [{"meal": "Desayuno", "name": "Revoltillo criollo",
                                           "ingredients": []}]} for d in (1, 2, 3)]}
    rep = g.build_variety_report(plan)
    assert rep.get("cross_day_dishes"), "mismo plato-base en 3 días debe reportarse"
    issues = g._variety_repeat_gate_issues(rep)
    assert any("REPETIDO ENTRE DÍAS" in s for s in issues)


def test_crossday_dish_gate_knob_off(monkeypatch):
    monkeypatch.setattr(g, "VARIETY_GATE_CROSS_DAY_DISH", False)
    rep = {"cross_day_dishes": {"revoltillo": 3}, "meals_per_day": 4}
    assert not any("REPETIDO ENTRE DÍAS" in s for s in g._variety_repeat_gate_issues(rep))


def test_sweet_fish_and_light_name_advisories():
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Cena", "name": "Tilapia con Mango", "cals": 400, "ingredients": []},
        {"meal": "Cena", "name": "Cena Ligera de Cerdo", "cals": 900, "ingredients": []},
    ]}]}
    rep = g.build_variety_report(plan)
    assert rep.get("sweet_fish_pairings") == 1
    assert rep.get("light_name_heavy_meals") == 1
    # advisory-only: jamás alimentan el gate de retry
    assert not any("Tilapia" in s for s in g._variety_repeat_gate_issues(rep))


def test_planner_transform_clause():
    from prompts.preferences import DETERMINISTIC_VARIETY_PROMPT as p
    assert "BASES TRANSFORMABLES" in p
    assert "panqueques" in p.lower()


def test_day_generator_ssot_block_appended():
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT as dp
    assert "CONTRATO EXACTO DEL VALIDADOR DE HORARIO" in dp
    # derivado del SSOT: los labels del validador viajan al prompt (drift-proof)
    assert "arroz de noche" in dp


def test_transform_base_boost_wired():
    assert "MEALFIT_TRANSFORM_BASE_BOOST" in _AIH
    assert '"harina"' in _AIH


def test_pantry_variety_advisory_wired():
    assert '_out["_same_day_protein_advisory"] = True' in _AGENT


# ── D: expand ────────────────────────────────────────────────────────────────

def test_expand_charge_deferred_after_persist():
    i_atomic = _PLANS.index("target_plan_id, _apply_recipe_expansion, user_id=user_id")
    i_charge = _PLANS.index('log_api_usage(user_id, "llm_recipe_expand")')
    assert i_atomic < i_charge, "el cobro debe correr DESPUÉS del persist (abort → sin cobro)"
    assert '"stale_target": True' in _PLANS


def test_expand_micro_recompute_in_callback():
    i_cb = _PLANS.index("def _apply_recipe_expansion")
    # [P2-AUDIT-V7-BATCH · 2026-07-04] ventana 9000→14000: P2-4 insertó el motor de macros +
    # band-parity ANTES/DESPUÉS del recompute (~35 líneas) — el contrato real es el ORDEN
    # (anclado en test_p2_audit_v7_batch.py::test_p2_4_expand_engine_and_band_parity_order).
    body = _PLANS[i_cb:i_cb + 14000]
    assert "P2-EXPAND-MICRO-RECOMPUTE" in body
    assert "recompute_micronutrient_report_for_plan" in body


def test_expand_finalize_parity_and_guest_safety():
    assert "_ensure_ingredients_used_in_recipe as _eiu_exp" in _PLANS
    assert "_collapse_double_fraction as _cdf_exp" in _PLANS
    # guest-safety corre sobre la RESPUESTA (post _resp_steps)
    i_resp = _PLANS.index("_resp_steps = list(expanded_steps)")
    assert "P2-EXPAND-GUEST-SAFETY" in _PLANS[i_resp:i_resp + 3000]


# ── E: ops (alerta + migración) ──────────────────────────────────────────────

def test_micro_kpi_alert_emitted_and_documented():
    assert "micro_estimado_bajo_chronic" in _CRON
    doc = (_BACKEND / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")
    assert "micro_estimado_bajo_chronic" in doc


def test_extended_micros_notnull_migration_ssot():
    name = "p2_extended_micros_notnull_2026_07_02.sql"
    b = _BACKEND / "migrations" / name
    r = _BACKEND.parent / "migrations" / name
    assert b.exists() and r.exists(), "migración debe vivir en AMBOS dirs (P3-MIGRATIONS-SSOT)"
    assert b.read_text(encoding="utf-8") == r.read_text(encoding="utf-8")
    src = b.read_text(encoding="utf-8")
    assert src.count("ALTER COLUMN") == 8   # las 8 columnas extendidas (el comment también dice SET NOT NULL)
    assert re.search(r"DO \$\$.*RAISE EXCEPTION", src, re.DOTALL)


# ── F: chat-modify target anchor ─────────────────────────────────────────────

def test_chatmod_target_anchor_wired():
    i = _TOOLS.index("P2-CHATMOD-TARGET-ANCHOR")
    body = _TOOLS[i:i + 4000]
    assert "_canonical_slot_fractions" in body
    assert "target_protein=float(_anchor_p or 0)" in _TOOLS
