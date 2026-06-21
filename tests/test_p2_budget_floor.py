"""[P2-BUDGET-FLOOR · 2026-06-21] Piso de presupuesto escalado por metas + bloqueo pre-gen.

Decisión del owner (audit presupuesto↔calidad): BLOQUEAR si el presupuesto 'custom' declarado
es físicamente insuficiente para alimentar al usuario según SUS metas (calorías × días × hogar),
y pedir ajuste — NUNCA bajar la calidad nutricional para encajar en un precio imposible.

El piso se calibra al número del propio frontend (200 DOP/día/persona a 2000 kcal = "comer bien")
y escala linealmente con las calorías objetivo. Conservador (lower bound) para no sobre-bloquear.
"""
import os

import nutrition_calculator as nc


def _form(budget=None, amount=None, currency="DOP", grocery="weekly", household=1, goal="maintenance"):
    f = {
        "weight": 70, "weightUnit": "kg", "height": 170, "age": 30,
        "gender": "male", "activityLevel": "moderate", "mainGoal": goal,
        "groceryDuration": grocery, "householdSize": household,
        "budgetCurrency": currency,
    }
    if budget is not None:
        f["budget"] = budget
    if amount is not None:
        f["budgetAmount"] = amount
    return f


# ---------------------------------------------------------------------------
# min_budget_for_goals
# ---------------------------------------------------------------------------
def test_min_budget_positivo_y_keys():
    info = nc.min_budget_for_goals(_form(grocery="weekly", household=1))
    assert info["min_budget_dop"] > 0
    assert info["days"] == 7
    assert info["household"] == 1
    assert info["target_calories"] > 0


def test_min_budget_escala_con_dias_y_hogar():
    semanal = nc.min_budget_for_goals(_form(grocery="weekly", household=1))["min_budget_dop"]
    mensual = nc.min_budget_for_goals(_form(grocery="monthly", household=1))["min_budget_dop"]
    hogar2 = nc.min_budget_for_goals(_form(grocery="weekly", household=2))["min_budget_dop"]
    assert mensual > semanal, "30 días debe costar más que 7."
    assert hogar2 > semanal, "2 personas deben costar más que 1."
    # 30 días ≈ ~4.3× los 7 días (lineal en días).
    assert mensual >= semanal * 3.5


def test_min_budget_escala_con_calorias():
    # gain_muscle (superávit) tiene más calorías que lose_fat (déficit) → más presupuesto.
    bajo = nc.min_budget_for_goals(_form(goal="lose_fat"))["min_budget_dop"]
    alto = nc.min_budget_for_goals(_form(goal="gain_muscle"))["min_budget_dop"]
    assert alto >= bajo


# ---------------------------------------------------------------------------
# validate_budget_sufficient
# ---------------------------------------------------------------------------
def test_bloquea_presupuesto_custom_demasiado_bajo():
    ok, detail = nc.validate_budget_sufficient(_form(budget="custom", amount=500, grocery="weekly"))
    assert ok is False
    assert detail and detail["error_code"] == "budget_below_goal_floor"
    assert "message" in detail and detail["min_budget"] > detail["declared"]


def test_acepta_presupuesto_custom_suficiente():
    ok, detail = nc.validate_budget_sufficient(_form(budget="custom", amount=8000, grocery="weekly"))
    assert ok is True and detail is None


def test_categorico_no_se_chequea():
    # 'medium'/'low'/'high'/'unlimited' son cualitativos — no tienen monto que validar.
    for b in ("low", "medium", "high", "unlimited"):
        ok, _ = nc.validate_budget_sufficient(_form(budget=b))
        assert ok is True, f"budget categórico '{b}' no debe bloquearse."


def test_custom_sin_monto_no_bloquea():
    ok, _ = nc.validate_budget_sufficient(_form(budget="custom", amount=None))
    assert ok is True


def test_usd_se_convierte_a_dop():
    # US$200 ≈ RD$12,000 (×60) → suficiente para semanal/1 persona.
    ok, _ = nc.validate_budget_sufficient(_form(budget="custom", amount=200, currency="USD", grocery="weekly"))
    assert ok is True
    # US$5 ≈ RD$300 → insuficiente.
    ok2, detail2 = nc.validate_budget_sufficient(_form(budget="custom", amount=5, currency="USD", grocery="weekly"))
    assert ok2 is False and detail2["currency"] == "USD"


def test_knob_off_no_bloquea(monkeypatch):
    monkeypatch.setenv("MEALFIT_BUDGET_FLOOR_ENABLED", "false")
    ok, _ = nc.validate_budget_sufficient(_form(budget="custom", amount=1, grocery="weekly"))
    assert ok is True, "Con el knob OFF, ningún presupuesto se bloquea (rollback sin redeploy)."


def test_anchor_marker():
    src = open(nc.__file__, encoding="utf-8").read()
    assert "P2-BUDGET-FLOOR" in src
    assert "def validate_budget_sufficient" in src
    assert "def min_budget_for_goals" in src
