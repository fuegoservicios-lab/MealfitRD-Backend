"""[P1-INCREMENTAL-LEARNING-ACTIVE-PLAN · 2026-09-02] El hook de aprendizaje incremental
(`trigger_incremental_learning`, disparado por CADA comida registrada en el diario) pedía
`meal_plans.status = 'active'`, columna que la tabla no tiene (desde 5c5ade2, 2026-04-22).
Moría en su except con un logger.error y el aprendizaje intradía nunca corrió; solo el
nocturno. Ahora resuelve «plan activo» con `get_latest_usable_meal_plan` (SSOT usable: el
placeholder vacío de la cola no cuenta).

Tooltip-anchor: P1-INCREMENTAL-LEARNING-ACTIVE-PLAN | get_latest_usable_meal_plan en trigger_incremental_learning
"""
import re
from pathlib import Path
from unittest.mock import patch

SRC = (Path(__file__).resolve().parents[1] / "cron_tasks.py").read_text(encoding="utf-8")


def _fn_body() -> str:
    i = SRC.index("def trigger_incremental_learning(")
    j = SRC.index("\ndef ", i + 10)
    return SRC[i:j]


def test_no_status_column_on_meal_plans_anywhere_in_cron_tasks():
    assert not re.search(r"FROM meal_plans[^\n]*status = 'active'", SRC), (
        "meal_plans no tiene columna status: esa query falla siempre"
    )


def test_hook_uses_usable_plan_reader_and_keeps_ingredients_contract():
    body = _fn_body()
    assert "from db import get_latest_usable_meal_plan" in body
    assert "get_latest_usable_meal_plan(user_id)" in body
    # contrato NG-1B (test_p2_cron_opt_2): diversity_score vivo
    assert "get_consumed_meals_since(user_id, plan_start_date_str, include_ingredients=True)" in body


def test_hook_persists_signals_for_usable_plan():
    import cron_tasks
    plan = {"days": [{"day": 1, "meals": []}], "_plan_start_date": "2026-09-01T00:00:00+00:00"}
    with patch("db.get_latest_usable_meal_plan", return_value=plan), \
         patch("db_profiles.get_user_profile", return_value={"health_profile": {"goal": "x"}}), \
         patch("db_facts.get_consumed_meals_since", return_value=[{"name": "Huevo"}]) as gc, \
         patch.object(cron_tasks, "_persist_nightly_learning_signals") as persist:
        cron_tasks.trigger_incremental_learning("u1")
    gc.assert_called_once_with("u1", "2026-09-01T00:00:00+00:00", include_ingredients=True)
    persist.assert_called_once()
    assert persist.call_args.args[2] == plan["days"]


def test_hook_is_noop_without_usable_plan():
    import cron_tasks
    with patch("db.get_latest_usable_meal_plan", return_value=None), \
         patch.object(cron_tasks, "_persist_nightly_learning_signals") as persist:
        cron_tasks.trigger_incremental_learning("u1")
    persist.assert_not_called()
