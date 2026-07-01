"""[P2-AUDIT-BATCH-2 · 2026-06-16] Tests del 2º lote de P2 del audit fresco (gaps de precisión/arquitectura/
subs). Cubre P2-5 (truth-up), P2-6 (low-coverage meals), P2-9 (band score macros-only), P2-10 (chunk
propaga _quality_degraded), P2-11 (gates incondicionales), P2-12 (re-pass allergen), P2-13 (subs diet-aware),
P2-15 (fold restricciones). P2-7 → test_p2_critical_config_alert.py; P2-8 → test_p2_band_score_gate.py.

Validación determinista (helpers puros + parser-anchors); sin LLM/DB/créditos.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BE = Path(__file__).resolve().parent.parent
_GO_SRC = (_BE / "graph_orchestrator.py").read_text(encoding="utf-8")
_CRON_SRC = (_BE / "cron_tasks.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


@pytest.fixture(scope="module")
def cr():
    import condition_rules as _cr
    return _cr


# ───────────────────────── P2-9: band score macros-only (excluye kcal) ─────────────────────────
def test_p2_9_score_macros_only_excludes_kcal(go):
    plan = {
        "macros": {"protein": "100g", "carbs": "200g", "fats": "50g"},
        "calories": 1800,
        "days": [{"day": 1, "meals": [{"protein": 50, "carbs": 100, "fats": 20, "cals": 1800}]}],
    }
    nutrition = {"total_daily_macros": {"protein_str": "100g", "carbs_str": "200g", "fats_str": "50g"},
                 "total_daily_calories": 1800}
    band = go.compute_clinical_band_score(plan, nutrition)
    # macros TODOS fuera de banda, kcal en banda → mixto 0.25 (1/4), macros-only 0.0 (0/3)
    assert band["score"] == 0.25
    assert band["score_macros_only"] == 0.0
    assert band["cells_total_macros"] == 3 and band["cells_in_band_macros"] == 0


def test_p2_9_knob_default_false(go):
    # [P1-BAND-GATE-ALL4 · 2026-07-01] SUPERSEDED: el default era False "hasta re-tunear el umbral vs
    # macros-only" — el re-tuning ocurrió (umbrales dedicados *_MACROS_ONLY=0.45) y el flip es ON.
    # El ancla ahora protege el par flip+umbral: si alguien revierte el flip o borra los umbrales
    # re-tuneados sin pasar por una decisión explícita, esto falla.
    assert go.BAND_GATE_USE_MACROS_ONLY is True
    assert go.BAND_RETRY_THRESHOLD_MACROS_ONLY == 0.45
    assert go.BAND_SCORE_GATE_THRESHOLD_MACROS_ONLY == 0.45


# ───────────────────────── P2-5: truth-up de macros desde strings ─────────────────────────
class _StubDB:
    def macros_from_ingredient_string(self, s):
        s = str(s).lower()
        if "al gusto" in s:
            return None  # cantidad no convertible
        if "pollo" in s:
            return {"protein": 30.0, "carbs": 0.0, "fats": 3.0, "kcal": 150.0}
        if "arroz" in s:
            return {"protein": 4.0, "carbs": 45.0, "fats": 0.5, "kcal": 200.0}
        return None

    def lookup(self, s):
        s = str(s).lower()
        return object() if ("pollo" in s or "arroz" in s or "sal" in s) else None


def test_p2_5_truthup_recomputes_from_strings(go):
    meal = {"name": "Almuerzo", "ingredients": ["180g pechuga de pollo", "1 taza arroz"],
            "protein": 99, "carbs": 12, "fats": 1, "cals": 500}  # números drifteados
    changed = go._truth_up_meal_macros_from_strings(meal, _StubDB())
    assert changed is True
    assert meal["protein"] == 34   # 30 + 4
    assert meal["carbs"] == 45     # 0 + 45
    assert meal["cals"] == 350     # 150 + 200


def test_p2_5_gate_no_override_when_unconvertible_qty(go):
    # nombre resuelve (lookup!=None) pero cantidad 'al gusto' (macros None) → NO override (conservador)
    meal = {"name": "X", "ingredients": ["sal al gusto", "180g pechuga de pollo"],
            "protein": 99, "carbs": 99, "fats": 99}
    assert go._truth_up_meal_macros_from_strings(meal, _StubDB()) is False
    assert meal["protein"] == 99  # intacto


def test_p2_5_gate_no_resolution_keeps_intact(go):
    meal = {"name": "Sancocho", "ingredients": ["1 plato de sancocho compuesto"],
            "protein": 40, "carbs": 60, "fats": 20}
    assert go._truth_up_meal_macros_from_strings(meal, _StubDB()) is False
    assert meal["protein"] == 40


def test_p2_5_knob_default_on(go):
    assert go.MACRO_TRUTHUP_ENABLED is True


# ───────────────────────── P2-6: telemetría low-coverage por-meal ─────────────────────────
def test_p2_6_knobs(go):
    assert go.LOW_COVERAGE_MEAL_SIGNAL is True
    assert go.LOW_COVERAGE_MEAL_FLOOR == 0.6


def test_p2_6_anchor_and_flag_persist():
    assert "P2-LOW-COVERAGE-MEALS" in _GO_SRC
    assert 'plan["low_coverage_meals"]' in _GO_SRC
    # telemetría, NO hard-gate: no debe llamar a un _maybe_mark_*_degraded para low-coverage
    assert "_m_res" in _GO_SRC and "_m_tot" in _GO_SRC


# ───────────────────────── P2-11: los 3 gates corren incondicionales ─────────────────────────
def test_p2_11_gates_dedented_below_band_val_check():
    assert "P2-11-DEGRADED-GATES-UNCONDITIONAL" in _GO_SRC
    # el band-gate está dentro del `if _band_val is not None` (más indentado); los 3 gates fuera (menos).
    band_gate = re.search(r"^(\s*)if _maybe_mark_low_band_degraded\(", _GO_SRC, re.MULTILINE)
    panel_gate = re.search(r"^(\s*)if _maybe_mark_panel_degraded\(", _GO_SRC, re.MULTILINE)
    clin_gate = re.search(r"^(\s*)if _maybe_mark_clinical_layer_incomplete_degraded\(", _GO_SRC, re.MULTILINE)
    res_gate = re.search(r"^(\s*)if _maybe_mark_low_resolution_degraded\(", _GO_SRC, re.MULTILINE)
    assert band_gate and panel_gate and clin_gate and res_gate
    assert len(panel_gate.group(1)) < len(band_gate.group(1)), "panel gate debe estar menos indentado (fuera del if)"
    assert len(clin_gate.group(1)) == len(panel_gate.group(1)) == len(res_gate.group(1))


# ───────────────────────── P2-10: chunk worker propaga _quality_degraded ─────────────────────────
def test_p2_10_chunk_propagates_quality_degraded():
    assert "P2-10-CHUNK-QUALITY-DEGRADED" in _CRON_SRC
    assert "result.get('_quality_degraded')" in _CRON_SRC
    # las 8 keys deben estar en P0_4_T2_INCREMENTAL_KEYS
    start = _CRON_SRC.find("P0_4_T2_INCREMENTAL_KEYS = (")
    block = _CRON_SRC[start: start + 2000]
    for k in ("_quality_degraded", "_quality_degraded_reason", "_quality_degraded_severity",
              "_quality_degraded_attempts", "_quality_degraded_band_score", "_quality_degraded_panel_detail",
              "_quality_degraded_clinical_detail", "_quality_degraded_resolution_pct"):
        assert f"'{k}'" in block, f"falta {k} en P0_4_T2_INCREMENTAL_KEYS"


# ───────────────────────── P2-12: re-pass allergen tras Guard 3 ─────────────────────────
def test_p2_12_allergen_repass_after_condition_subs():
    start = _GO_SRC.find("def _apply_deterministic_clinical_layer")
    body = _GO_SRC[start: start + 60000]
    assert "P2-12" in body
    i_cond = body.find("_apply_condition_substitutions(plan, form_data)")
    i_repass = body.find("_apply_allergen_substitutions(plan, form_data)", i_cond)
    assert i_cond != -1 and i_repass != -1 and i_repass > i_cond, \
        "el re-pass de allergen debe aparecer DESPUÉS de las subs por condición (Guard 3)"


def test_p2_12_arenque_token_in_fish_subs(cr):
    fish_tokens = cr._ALLERGEN_FISH_SUBS[0][0]
    assert "arenque" in fish_tokens and "arenque salado" in fish_tokens


# ───────────────────────── P2-13: subs diet-aware ─────────────────────────
def test_p2_13_redirect_logic(cr):
    assert cr._redirect_replacement_for_diet("Pechuga de pollo", "vegan") == "Lentejas"
    assert cr._redirect_replacement_for_diet("Pechuga de pollo", "vegetarian") == "Lentejas"
    assert cr._redirect_replacement_for_diet("Filete de pescado blanco", "vegan") == "Lentejas"
    assert cr._redirect_replacement_for_diet("Pechuga de pollo", "pescatarian") == "Filete de pescado blanco"
    assert cr._redirect_replacement_for_diet("Filete de pescado blanco", "pescatarian") == "Filete de pescado blanco"
    assert cr._redirect_replacement_for_diet("Pechuga de pollo", "balanced") == "Pechuga de pollo"


def test_p2_13_canon_diet_covers_legacy_fem(cr):
    assert cr._canon_diet("vegana") == "vegan"
    assert cr._canon_diet("vegetariana") == "vegetarian"
    assert cr._canon_diet("balanced") == "balanced"


def test_p2_13_allergen_collector_diet_aware(cr):
    vegan = cr.collect_allergen_substitutions({"allergies": ["pescado"]}, diet_type="vegan")
    balanced = cr.collect_allergen_substitutions({"allergies": ["pescado"]})
    assert vegan, "alergia a pescado debe producir subs"
    assert all(s["replacement"] != "Pechuga de pollo" for s in vegan), "vegano NO debe reemplazar a pollo"
    assert any(s["replacement"] == "Pechuga de pollo" for s in balanced), "balanced reemplaza a pollo (control)"


def test_p2_13_pescatarian_fish_allergy_not_fish(cr):
    """[review-fix] pescetariano + alergia a pescado: el reemplazo del pescado NO puede ser pescado
    (reintroduciría el alérgeno). Debe caer a vegetal."""
    subs = cr.collect_allergen_substitutions({"allergies": ["pescado"]}, diet_type="pescetariano")
    fish_subs = [s for s in subs if s["condition"] == "allergen:fish"]
    assert fish_subs, "debe haber sub para el alérgeno pescado"
    for s in fish_subs:
        r = s["replacement"].lower()
        assert "pescado" not in r and "filete de pescado" not in r, \
            f"pescetariano+alergia-pescado NO debe reemplazar a pescado: {s['replacement']}"
    # shellfish allergy + pescatarian SÍ puede ir a pescado (pescado != mariscos)
    shell = cr.collect_allergen_substitutions({"allergies": ["camarones"]}, diet_type="pescetariano")
    assert any(s["condition"] == "allergen:shellfish" for s in shell)


# ───────────────────────── P2-15: fold de restricciones a allergies ─────────────────────────
def test_p2_15_fold_intolerances_into_allergies(go):
    fd = {"allergies": ["Ninguna"], "intolerances": ["mariscos"], "dislikes": ["Ninguno"]}
    go._merge_other_text_fields(fd)
    al = [str(x).lower() for x in fd.get("allergies", [])]
    assert "mariscos" in al, "la intolerancia real debe plegarse a allergies"
    assert "ninguna" not in al, "el sentinel 'Ninguna' debe ser desplazado por la restricción real"


def test_p2_15_no_fold_when_no_restrictions(go):
    fd = {"allergies": ["maní"], "dislikes": ["Ninguno"]}
    go._merge_other_text_fields(fd)
    assert [str(x).lower() for x in fd["allergies"]] == ["maní"]


def test_p2_15_knob_default_on(go):
    assert go.FOLD_RESTRICTION_ALIASES is True
