"""[P1-BUDGET-FLOOR-PERSONALIZED · 2026-06-23] El frontend muestra el piso de presupuesto
PERSONALIZADO por las metas (calorías × hogar × ciclo) — el MISMO número que exige el gate de
generación `validate_budget_sufficient` — vía el endpoint `POST /api/plans/budget-floor`.

Antes el form/dashboard mostraban `minBudgetFor` (piso a 2000 kcal sin escalar): un usuario de
calorías altas veía un mínimo bajo y luego el backend lo rechazaba con 422 al generar. Este test
ancla (1) que el piso escala con las calorías/hogar, (2) que el endpoint existe y devuelve el
contrato esperado con fail-open, (3) la conversión de moneda.
"""
import asyncio
from pathlib import Path

import nutrition_calculator as nc


_PLANS_SRC = (Path(__file__).resolve().parent.parent / "routers" / "plans.py").read_text(encoding="utf-8")


def _form(cal_high=False, grocery="biweekly", household=1, currency="DOP"):
    base = {
        "weight": 70, "weightUnit": "kg", "height": 170, "age": 30,
        "gender": "male", "activityLevel": "moderate",
        "groceryDuration": grocery, "householdSize": household, "budgetCurrency": currency,
    }
    if cal_high:
        base.update({"weight": 95, "height": 188, "gender": "male", "activityLevel": "active", "mainGoal": "gain_muscle"})
    else:
        base.update({"weight": 58, "height": 158, "gender": "female", "activityLevel": "sedentary", "mainGoal": "lose_fat"})
    return base


# ---------------------------------------------------------------------------
# 1. El piso escala con calorías y con el hogar (lo que personaliza el mínimo).
# ---------------------------------------------------------------------------
def test_floor_scales_with_calories():
    lo = nc.min_budget_for_goals(_form(cal_high=False))["min_budget_dop"]
    hi = nc.min_budget_for_goals(_form(cal_high=True))["min_budget_dop"]
    assert hi > lo, "Un usuario de más calorías debe tener un piso de presupuesto mayor."


def test_floor_scales_with_household():
    one = nc.min_budget_for_goals(_form(household=1))["min_budget_dop"]
    three = nc.min_budget_for_goals(_form(household=3))["min_budget_dop"]
    assert three > one, "Más personas en el hogar → piso mayor."


def test_high_cal_floor_above_static_cycle():
    # Para un usuario de calorías altas el piso 15d supera el piso estático (7000 a 2000 kcal):
    # justo el caso donde el mínimo estático mentía.
    hi = nc.min_budget_for_goals(_form(cal_high=True, grocery="biweekly"))["min_budget_dop"]
    assert hi > nc._budget_cycle_floor_dop(15), "El personalizado de alta caloría debe exceder el estático."


# ---------------------------------------------------------------------------
# 2. El endpoint existe y devuelve el contrato (parser + funcional).
# ---------------------------------------------------------------------------
def test_endpoint_anchored_in_source():
    assert '@router.post("/budget-floor")' in _PLANS_SRC
    assert "P1-BUDGET-FLOOR-PERSONALIZED" in _PLANS_SRC
    assert "min_budget_for_goals" in _PLANS_SRC
    # Fail-open + campos clave del contrato.
    assert '"ok": False' in _PLANS_SRC
    assert '"min_budget"' in _PLANS_SRC
    assert '"target_calories"' in _PLANS_SRC


def test_endpoint_returns_personalized_floor():
    from routers.plans import api_budget_floor
    res = asyncio.run(api_budget_floor(payload=_form(cal_high=True), _uid=None))
    assert res.get("ok") is True
    assert res["min_budget"] > 0
    assert res["currency"] == "DOP"
    assert res["target_calories"] > 0
    # Coincide con el cálculo directo (cero drift con el gate).
    expected = nc.min_budget_for_goals(_form(cal_high=True))["min_budget_dop"]
    assert res["min_budget_dop"] == int(round(expected))


def test_endpoint_usd_conversion():
    from routers.plans import api_budget_floor
    res = asyncio.run(api_budget_floor(payload=_form(cal_high=True, currency="USD"), _uid=None))
    assert res.get("ok") is True
    assert res["currency"] == "USD"
    # USD = DOP / tasa (mismo divisor que el gate al convertir el monto declarado).
    usd_dop = nc._budget_usd_to_dop()
    assert res["min_budget"] == int(round(res["min_budget_dop"] / usd_dop))


def test_endpoint_fail_open_on_garbage():
    from routers.plans import api_budget_floor
    # payload None / vacío no debe romper: min_budget_for_goals usa defaults seguros.
    res = asyncio.run(api_budget_floor(payload=None, _uid=None))
    assert res.get("ok") in (True, False)  # no lanza; ok=True (defaults) o False (fail-open)
