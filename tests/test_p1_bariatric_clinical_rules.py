"""[P1-BARIATRIC-CLINICAL-RULES · 2026-06-27] Reglas clínicas de generación para pacientes post-cirugía
bariátrica. Origen: un plan bariátrico real (corr=558af493) fue RECHAZADO CRÍTICO por el revisor médico
('5¼ lonjas de queso' → dumping/volumen, miel → azúcar simple, pescado en merienda nocturna, '2.5 fresas
solo la cáscara') → entrega degradada. El chip 'Cirugía Bariátrica' (P2-FORM-FREETEXT-SATISFIES) ya enrutaba
6 comidas (P1-CLINICAL-MEAL-COUNT) pero NO había reglas clínicas. Ruleset diseñado + verificado
adversarialmente (workflow 3 lentes clínicas + crítico).

Cubre los 3 fixes desplegados:
  #1 ConditionRule bariátrica (prompt_block anti-dumping + subs azúcar→estevia) + cap determinista de queso/yogurt.
  #2 Relajación de los gates de variedad same-day-protein/fruit en planes de ≥5 comidas.
  #3 dish-quality detecta porción imposible ('solo la cáscara' en fruta sin cáscara comestible).
"""
from __future__ import annotations

import re
from pathlib import Path

import nutrition_calculator as nc
import condition_rules as cr

_BACKEND = Path(nc.__file__).resolve().parent


# ──────────────────────────── #1 meal-count + ConditionRule ────────────────────────────

def test_bariatric_routes_to_6_meals_chip_and_freetext():
    assert nc.decide_meals_per_day({"medicalConditions": ["Cirugía Bariátrica"]})["num_meals"] == 6
    assert nc.decide_meals_per_day({"otherConditions": "me hice un bypass gastrico"})["num_meals"] == 6
    assert nc.decide_meals_per_day({"medicalConditions": ["manga gastrica"]})["num_meals"] == 6


def test_bariatric_condition_rule_active_and_prompt_covers_key_risks():
    active = cr.detect_active_rules({"medicalConditions": ["Cirugía Bariátrica"]})
    assert "bariatric" in [r.id for r in active]
    blk = cr.build_condition_prompt({"medicalConditions": ["Cirugía Bariátrica"]}).upper()
    # dumping + azúcar simple prohibido + cap de queso + merienda nocturna + porciones realistas
    assert "DUMPING" in blk
    assert "AZÚCAR SIMPLE" in blk or "AZUCAR SIMPLE" in blk
    assert "QUESO" in blk and "30 G" in blk.replace("≤", "")
    assert "MERIENDA NOCTURNA" in blk
    assert "PROTEÍNA PRIMERO" in blk or "PROTEINA PRIMERO" in blk


def test_bariatric_substitutes_simple_sugar():
    subs = cr.collect_substitutions({"medicalConditions": ["Cirugía Bariátrica"]})
    blob = " ".join(str(s) for s in subs).lower()
    assert "miel" in blob and "stevia" in blob


# ──────────────────────────── #1 cap determinista (stub DB, sin red) ────────────────────────────

class _StubDB:
    """macros_from_ingredient_string que parsea los gramos del prefijo — evita la DB real."""
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", str(s))
        g = float(m.group(1)) if m else None
        return {"grams": g, "protein": (g or 0) * 0.2, "carbs": (g or 0) * 0.04,
                "fats": (g or 0) * 0.25, "kcal": (g or 0) * 3.0}


def test_cap_bariatric_portions_trims_excess_cheese():
    import graph_orchestrator as g
    days = [{"day": 1, "meals": [{"name": "Merienda de queso",
                                  "ingredients": ["150g de Queso blanco", "60g de Tomate"]}]}]
    n = g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, db=_StubDB())
    assert n >= 1
    cheese = days[0]["meals"][0]["ingredients"][0]
    grams = float(re.match(r"\s*(\d+(?:\.\d+)?)\s*g", cheese).group(1))
    assert grams <= g.BARIATRIC_CHEESE_CAP_G, f"queso no capeado: {cheese}"


def test_cap_bariatric_noop_when_not_bariatric():
    import graph_orchestrator as g
    days = [{"day": 1, "meals": [{"name": "x", "ingredients": ["150g de Queso blanco"]}]}]
    assert g.cap_bariatric_portions(days, {"medicalConditions": ["Ninguna"]}, db=_StubDB()) == 0


# ──────────────────────────── #2 variety gates relajados en ≥5 comidas ────────────────────────────

def test_variety_gates_relaxed_for_high_mealcount():
    import graph_orchestrator as g
    rep = {"fruit_repeats": 1, "same_day_protein_repeats": 1, "sweet_savory_clash": 0,
           "same_day_repeats": 0, "meals_per_day": 6}
    out = g._variety_repeat_gate_issues(rep)
    blob = " ".join(out).upper()
    assert "MISMA PROTEÍNA REPETIDA" not in blob and "MISMA PROTEINA REPETIDA" not in blob
    assert "FRUTA REPETIDA" not in blob


def test_variety_gates_strict_for_normal_mealcount():
    import graph_orchestrator as g
    rep = {"fruit_repeats": 1, "same_day_protein_repeats": 1, "sweet_savory_clash": 0,
           "same_day_repeats": 0, "meals_per_day": 4}
    out = g._variety_repeat_gate_issues(rep)
    blob = " ".join(out).upper()
    assert "MISMA PROTE" in blob  # gate activo en 4 comidas
    assert "FRUTA REPETIDA" in blob


def test_variety_clash_gate_not_relaxed_by_mealcount():
    # el pareo fruta+salado NO se relaja por conteo (es coherencia, no conteo)
    import graph_orchestrator as g
    rep = {"fruit_repeats": 0, "same_day_protein_repeats": 0, "sweet_savory_clash": 1,
           "same_day_repeats": 0, "meals_per_day": 6}
    out = g._variety_repeat_gate_issues(rep)
    assert any("CHOCANTE" in o.upper() for o in out)


# ──────────────────────────── #3 dish-quality 'solo la cáscara' ────────────────────────────

def test_dish_quality_flags_impossible_peel_instruction():
    import graph_orchestrator as g
    low, reason = g._meal_dish_quality_issue(
        {"name": "Fresas raras", "ingredients": ["2.5 Fresas (solo la cáscara)"],
         "recipe": ["Mise en place: lavar", "El toque de fuego: servir frío en copa"]})
    assert low is True and reason


def test_dish_quality_allows_legit_zest():
    import graph_orchestrator as g
    # ralladura de limón (zest) usa la cáscara LEGÍTIMAMENTE → no debe marcarse por esa razón
    low, reason = g._meal_dish_quality_issue(
        {"name": "Pollo al limón", "ingredients": ["120g de Pollo", "Ralladura de limón (solo la cáscara)"],
         "recipe": ["Mise en place: sazonar el pollo con la ralladura",
                    "El toque de fuego: hornear 25 min a 180C"]})
    assert not (low and reason and "cáscara" in str(reason))


# ──────────────────────────── anchors ────────────────────────────

def test_anchors_present():
    cr_src = (_BACKEND / "condition_rules.py").read_text(encoding="utf-8")
    assert "P1-BARIATRIC-CLINICAL-RULES" in cr_src and 'id="bariatric"' in cr_src
    go_src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "def cap_bariatric_portions" in go_src
    assert "P2-VARIETY-HIGH-MEALCOUNT-RELAX" in go_src
    assert "P2-DISH-QUALITY-PEEL" in go_src
    co_src = (_BACKEND / "constants.py").read_text(encoding="utf-8")
    assert "BARIATRIC_CONDITION_TERMS" in co_src
