"""[P3-CONDITION-ENGINE · 2026-06-14] Motor de constraints clínicos declarativo (condition_rules.py).
Generaliza el patrón ad-hoc DM2/ERC a un REGISTRO; añade HTA (sodio enforced) + dislipidemia + anemia
como FILAS de datos. Verifica detección+precedencia, prompts citables, comorbilidad, y la sustitución
determinista por condición (DM2 azúcar + HTA sodio) vía el guard generalizado de graph_orchestrator.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import condition_rules as cr
import graph_orchestrator as go


# ── Registro + detección + precedencia ──
def test_registry_covers_pareto_set():
    ids = {r.id for r in cr.CONDITION_RULES}
    assert {"renal", "dm2", "hta", "dyslipidemia", "anemia"} <= ids


@pytest.mark.parametrize("conds,expected_ids", [
    (["Hipertensión"], ["hta"]),
    (["Colesterol alto"], ["dyslipidemia"]),
    (["Anemia ferropénica"], ["anemia"]),
    (["presion alta"], ["hta"]),
    (["Ninguna"], []),
    ([], []),
])
def test_detect_active_rules(conds, expected_ids):
    active = cr.detect_active_rules({"medicalConditions": conds})
    assert [r.id for r in active] == expected_ids


def test_precedence_renal_first_in_comorbidity():
    """ERC (precedence 10) debe ir antes que DM2 (30) y HTA (40) — seguridad primero."""
    active = cr.detect_active_rules(
        {"medicalConditions": ["Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica"]})
    assert [r.id for r in active] == ["renal", "dm2", "hta"]


# ── Prompt registry-driven ──
def test_prompt_hta_dash():
    txt = cr.build_condition_prompt({"medicalConditions": ["Hipertensión"]})
    assert "DASH" in txt and "1500" in txt and "embutidos" in txt.lower()


def test_prompt_dyslipidemia_and_anemia():
    txt = cr.build_condition_prompt({"medicalConditions": ["Colesterol alto", "Anemia"]})
    assert "saturada" in txt.lower() and "<7%" in txt   # dislipidemia
    assert "hierro" in txt.lower() and "vitamina c" in txt.lower()  # anemia
    assert "MÁS RESTRICTIVA" in txt   # nota de comorbilidad genérica


def test_prompt_multi_condition_dm2_renal_precedence():
    txt = cr.build_condition_prompt({"medicalConditions": ["Diabetes T2", "ERC"]})
    assert "PRECEDENCIA" in txt and "MANDA" in txt


# ── Sustituciones declarativas ──
def test_collect_substitutions_hta_sodium():
    subs = cr.collect_substitutions({"medicalConditions": ["Hipertensión"]})
    labels = {s["label"] for s in subs}
    assert "embutidos" in labels and "cubitos/sazón en polvo" in labels
    assert all(s["condition"] == "hta" for s in subs)


def test_collect_substitutions_merges_dm2_and_hta():
    subs = cr.collect_substitutions({"medicalConditions": ["Diabetes T2", "Hipertensión"]})
    conds = {s["condition"] for s in subs}
    assert conds == {"dm2", "hta"}


# ── Guard generalizado en graph_orchestrator ──
def test_apply_substitutions_hta_replaces_sodium():
    plan = {"days": [{"meals": [
        {"name": "Almuerzo", "ingredients": ["100g de longaniza", "1 cubito de pollo", "Arroz"],
         "ingredients_raw": ["100g de longaniza"], "recipe": ["Guisar"]},
    ]}]}
    n = go._apply_condition_substitutions(plan, {"medicalConditions": ["Hipertensión"]})
    assert n == 1
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert not any("longaniza" in str(i).lower() for i in ings)
    assert not any("cubito" in str(i).lower() for i in ings)
    assert "Arroz" in ings
    assert plan["days"][0]["meals"][0].get("_condition_subs_fixed")


def test_apply_substitutions_dm2_plus_hta_together():
    plan = {"days": [{"meals": [
        {"name": "Desayuno", "ingredients": ["Avena", "1 cda de miel", "2 lonjas de salami"]},
    ]}]}
    go._apply_condition_substitutions(plan, {"medicalConditions": ["Diabetes T2", "Hipertensión"]})
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert not any("miel" in str(i).lower() for i in ings)     # DM2
    assert not any("salami" in str(i).lower() for i in ings)   # HTA
    assert plan["days"][0]["meals"][0].get("_dm2_sugar_fixed")  # flag compat DM2 preservado


def test_apply_substitutions_noop_without_condition():
    plan = {"days": [{"meals": [{"ingredients": ["100g de longaniza", "1 cda de miel"]}]}]}
    assert go._apply_condition_substitutions(plan, {"medicalConditions": ["Ninguna"]}) == 0


def test_apply_substitutions_respects_low_sodium_negative():
    plan = {"days": [{"meals": [{"ingredients": ["1 cda de salsa de soya baja en sodio"]}]}]}
    go._apply_condition_substitutions(plan, {"medicalConditions": ["Hipertensión"]})
    # "baja en sodio" veta la sustitución de la salsa de soya
    assert any("soya" in str(i).lower() for i in plan["days"][0]["meals"][0]["ingredients"])
