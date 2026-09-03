"""[P1-ARQ25-F1-CLOSE · 2026-09-02] El coach no habla con el placeholder de la cola.

Desde la Fase 1 el plan nace en `meal_plans` antes de generarse (`generating`, `days=[]`).
`get_latest_meal_plan*` devuelve el más reciente por fecha, así que durante la generación
el chat y las tools veían un plan vacío. Los lectores «usables» saltan ese placeholder;
los routers que resuelven «el plan de esta mutación» siguen con los lectores de siempre.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent


def _src(rel):
    return (BACKEND / rel).read_text(encoding="utf-8")


def test_usable_readers_filter_out_empty_placeholders(monkeypatch):
    db_plans = pytest.importorskip("db_plans")
    seen = {}

    def _fake(sql, params=None, fetch_one=False, **kw):
        seen["sql"] = sql
        seen["params"] = params
        return {"id": "p1", "plan_data": {"days": [{"day": 1}]}, "created_at": None}

    monkeypatch.setattr(db_plans, "execute_sql_query", _fake)
    rec = db_plans.get_latest_usable_meal_plan_with_id("u1")
    assert rec["id"] == "p1" and seen["params"] == ("u1",)
    assert db_plans.USABLE_PLAN_SQL_FILTER in seen["sql"], "el filtro va en la query, no en Python"
    assert "generation_status" in db_plans.USABLE_PLAN_SQL_FILTER and "jsonb_array_length" in db_plans.USABLE_PLAN_SQL_FILTER
    assert "ORDER BY created_at DESC LIMIT 1" in seen["sql"] and "WHERE user_id = %s" in seen["sql"], "I2"
    assert db_plans.get_latest_usable_meal_plan("u1") == {"days": [{"day": 1}]}


def test_usable_readers_fail_closed_to_none(monkeypatch):
    db_plans = pytest.importorskip("db_plans")

    def _boom(*a, **k):
        raise RuntimeError("db caída")

    monkeypatch.setattr(db_plans, "execute_sql_query", _boom)
    assert db_plans.get_latest_usable_meal_plan_with_id("u1") is None
    assert db_plans.get_latest_usable_meal_plan("u1") is None


def test_legacy_readers_keep_seeing_the_placeholder():
    """Los routers que adjuntan el run o deduplican creación reciente necesitan VER el placeholder."""
    src = _src("db_plans.py")
    i = src.find("def get_latest_meal_plan_with_id(")
    body = src[i:i + 900]
    assert "USABLE_PLAN_SQL_FILTER" not in body and "jsonb_array_length" not in body
    rp = _src("routers/plans.py")
    assert "get_latest_usable_meal_plan" not in rp, "los routers no cambian de lector"


def test_chat_agent_and_tools_read_the_usable_plan():
    agent = _src("agent.py")
    assert "get_latest_meal_plan_with_id(" not in agent, "el chat no debe leer el placeholder"
    assert agent.count("get_latest_usable_meal_plan_with_id(") >= 4
    tools = _src("tools.py")
    assert not re.search(r"(?<![a-z_])get_latest_meal_plan\(user_id\)", tools)
    assert "get_latest_meal_plan_with_id(user_id)" not in tools
    assert "get_latest_usable_meal_plan(user_id)" in tools and "get_latest_usable_meal_plan_with_id(user_id)" in tools
    sc = _src("shopping_calculator.py")
    assert "plan_record = get_latest_usable_meal_plan_with_id(user_id)" in sc


def test_facade_exports_the_usable_readers():
    db = pytest.importorskip("db")
    assert callable(getattr(db, "get_latest_usable_meal_plan_with_id", None))
    assert callable(getattr(db, "get_latest_usable_meal_plan", None))
