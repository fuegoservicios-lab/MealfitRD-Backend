"""[P2-INVENTORY-LEGACY-PLAN-WARN-ONCE · 2026-09-02] El aviso «plan activo sin
calc_household_multiplier» se emite UNA vez por usuario y proceso.

Medido en prod: 16 cuentas `e2e-test-*@test.local` (21-22 ago, 0 comidas) cuyo plan usable
es un esqueleto sin el campo; el cron de refill las recorre en cada tick y escribía 16
WARNING idénticos por vuelta. El fallback (rates hardcodeados por categoría) no cambia.

Tooltip-anchor: P2-INVENTORY-LEGACY-PLAN-WARN-ONCE | _P2_4_NO_MULTIPLIER_WARNED
"""
import logging
from unittest.mock import patch

import db_inventory


def _plan_without_multiplier():
    return {"plan_data": {"days": [{"meals": []}], "aggregated_shopping_list_weekly": [{"name": "Huevo"}]}}


def test_warns_once_per_user_then_debug(caplog):
    db_inventory._P2_4_NO_MULTIPLIER_WARNED.clear()
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_query", return_value=_plan_without_multiplier()):
        with caplog.at_level(logging.DEBUG, logger="db_inventory"):
            r1 = db_inventory._compute_dynamic_consumption_rates("user-a")
            r2 = db_inventory._compute_dynamic_consumption_rates("user-a")
            r3 = db_inventory._compute_dynamic_consumption_rates("user-b")
    assert r1 == {} and r2 == {} and r3 == {}
    warns = [rec for rec in caplog.records if rec.levelno == logging.WARNING and "[P2-4]" in rec.getMessage()]
    debugs = [rec for rec in caplog.records if rec.levelno == logging.DEBUG and "[P2-4]" in rec.getMessage()]
    assert len(warns) == 2, "una vez por usuario (a y b)"
    assert len(debugs) == 1, "la repetición de a va a DEBUG"
    assert "user=user-a" in warns[0].getMessage()


def test_plan_with_multiplier_does_not_touch_the_guard():
    db_inventory._P2_4_NO_MULTIPLIER_WARNED.clear()
    plan = _plan_without_multiplier()
    plan["plan_data"]["calc_household_multiplier"] = 1.0
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_query", return_value=plan):
        db_inventory._compute_dynamic_consumption_rates("user-c", current_household_multiplier=1.0)
    assert "user-c" not in db_inventory._P2_4_NO_MULTIPLIER_WARNED
