"""[P1-CLINICAL-MEAL-COUNT · 2026-06-27] La cantidad de comidas/día del plan la decide la IA desde los
datos clínicos del formulario (tabla del owner): 2-3 'pocas e intensas' (sensibilidad a la insulina) ↔
4-6 'pequeñas y frecuentes' (hipoglucemia/bariátrica/alto gasto). Override opcional del usuario
(num_meals/mealsPerDay). Enforcement determinista: el skeleton fuerza `meal_types` al conteo decidido y
el day_generator genera exactamente esos slots; el pipeline se adapta a len(meals).

SEGURIDAD: el riesgo de hipoglucemia (condición o insulina/sulfonilurea) SIEMPRE gana sobre el reduce de
DM2 — un DM2 en insulina NO reduce comidas (sería iatrogénico).

Cubre: nutrition_calculator.decide_meals_per_day / meal_types_for_count / MEAL_SLOT_SPLITS;
graph_orchestrator._canonical_slot_fractions (3/5/6); knob + override del skeleton (parser-anchored).
"""
from __future__ import annotations

from pathlib import Path

import nutrition_calculator as nc


def _n(fd, kcal=None):
    return nc.decide_meals_per_day(fd, kcal)


def test_default_healthy_user_4_meals():
    r = _n({})
    assert r["num_meals"] == 4 and r["source"] == "default"


def test_dm2_reduces_to_3():
    assert _n({"medicalConditions": ["Diabetes tipo 2"]})["num_meals"] == 3
    assert _n({"medicalConditions": ["Resistencia a la insulina"]})["num_meals"] == 3
    assert _n({"medicalConditions": ["Prediabetes"]})["num_meals"] == 3


def test_hypoglycemia_risk_overrides_dm2_reduce_SAFETY():
    """DM2 + insulina/sulfonilurea o hipoglucemia → 5 (frecuentes); NUNCA reduce (iatrogénico)."""
    assert _n({"medicalConditions": ["Diabetes tipo 2"], "medications": ["Insulina glargina"]})["num_meals"] == 5
    assert _n({"medicalConditions": ["Diabetes tipo 2"], "medications": ["Glibenclamida"]})["num_meals"] == 5
    assert _n({"medicalConditions": ["Hipoglucemia reactiva"]})["num_meals"] == 5


def test_bariatric_6_meals():
    assert _n({"medicalConditions": ["Cirugía bariátrica (bypass gástrico)"]})["num_meals"] == 6
    assert _n({"medicalConditions": ["Manga gástrica"]})["num_meals"] == 6


def test_high_kcal_5_meals():
    assert _n({}, 3000)["num_meals"] == 5
    assert _n({}, 2899)["num_meals"] == 4   # bajo el umbral


def test_user_override_respected_and_clamped():
    assert _n({"num_meals": 3})["source"] == "override"
    assert _n({"mealsPerDay": 6})["num_meals"] == 6
    assert _n({"num_meals": 9})["num_meals"] == 4      # fuera de [2,6] → ignora
    assert _n({"num_meals": 1})["num_meals"] == 4
    # override gana sobre la regla clínica
    assert _n({"medicalConditions": ["Diabetes tipo 2"], "num_meals": 5})["num_meals"] == 5


def test_failsafe_on_garbage():
    assert nc.decide_meals_per_day(None)["num_meals"] == 4


def test_real_form_labels_and_freetext():
    """Payloads EXACTOS del formulario (InteractiveQuestions.jsx): el chip dice 'Diabetes T2' (no
    'Diabetes tipo 2') y hipoglucemia/bariátrica solo entran por texto libre (otherConditions)."""
    # chip "Diabetes T2" → debe matchear (era el bug: el decisor buscaba 'diabetes tipo 2'/'dm2')
    assert nc.decide_meals_per_day({"medicalConditions": ["Diabetes T2"]})["num_meals"] == 3
    # chip Diabetes T2 + chip medicación Insulina → 5 (seguridad hipo)
    assert nc.decide_meals_per_day({"medicalConditions": ["Diabetes T2"], "medications": ["Insulina"]})["num_meals"] == 5
    assert nc.decide_meals_per_day({"medicalConditions": ["Diabetes T2"], "medications": ["Glibenclamida"]})["num_meals"] == 5
    # texto libre del formulario (otherConditions / otherMedications) también se lee
    assert nc.decide_meals_per_day({"medicalConditions": [], "otherConditions": "tengo hipoglucemia reactiva"})["num_meals"] == 5
    assert nc.decide_meals_per_day({"otherConditions": "cirugía bariátrica (bypass gástrico)"})["num_meals"] == 6
    assert nc.decide_meals_per_day({"otherMedications": "uso insulina glargina"})["num_meals"] == 5
    # condiciones del formulario que NO cambian el conteo → 4
    for c in ("Hipertensión", "Colesterol Alto", "Gastritis", "SOP (PCOS)", "Hipotiroidismo"):
        assert nc.decide_meals_per_day({"medicalConditions": [c]})["num_meals"] == 4, c


def test_meal_types_by_count():
    assert nc.meal_types_for_count(3) == ["Desayuno", "Almuerzo", "Cena"]
    assert nc.meal_types_for_count(6)[-1] == "Merienda Nocturna"
    assert len(nc.meal_types_for_count(5)) == 5
    assert nc.meal_types_for_count(99) == nc.meal_types_for_count(4)   # fallback


def test_all_splits_sum_to_one():
    for n, split in nc.MEAL_SLOT_SPLITS.items():
        assert abs(sum(split.values()) - 1.0) < 1e-9, f"split {n} no suma 1.0"


def test_canonical_fractions_adapt_to_count():
    import graph_orchestrator as g
    for n in (3, 5, 6):
        meals = [{"slot": t} for t in nc.meal_types_for_count(n)]
        fr = g._canonical_slot_fractions(meals)
        assert len(fr) == n
        assert all(x is not None and x > 0 for x in fr), (n, fr)
        assert abs(sum(fr) - 1.0) < 0.02, (n, fr)


# ──────────────────────────── parser-anchored ────────────────────────────

def _src(name):
    import graph_orchestrator as g
    return (Path(g.__file__).resolve().parent / name).read_text(encoding="utf-8")


def test_anchors_and_wiring():
    nc_src = _src("nutrition_calculator.py")
    go_src = _src("graph_orchestrator.py")
    assert "P1-CLINICAL-MEAL-COUNT" in nc_src and "P1-CLINICAL-MEAL-COUNT" in go_src
    assert 'CLINICAL_MEAL_COUNT_ENABLED = _env_bool("MEALFIT_CLINICAL_MEAL_COUNT", True)' in go_src
    # override determinista de meal_types en el skeleton + directiva al prompt
    assert '_d["meal_types"] = list(_mc_types)' in go_src
    assert "_meal_count_directive" in go_src
    # merienda nocturna mapeada para el reparto de macros del plan de 6 comidas
    assert '"merienda nocturna": "merienda"' in go_src


def test_knob_default_on():
    import graph_orchestrator as g
    assert g.CLINICAL_MEAL_COUNT_ENABLED is True
