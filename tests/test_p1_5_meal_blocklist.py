"""[P1-5] Permanent meal blocklist + dynamic cap para plan 30d.

Antes, `_chunk_lessons.repeated_meal_names` se capeaba a 10 y `_lifetime_lessons_summary`
solo trackeaba `top_rejection_hits` y `top_repeated_bases` — los meal_names repetidos
cross-chunk se perdían cuando salían del rolling window. En plan 30d (10+ chunks):
si "Pollo al horno" se repetía en chunks 1 y 4, el rolling window (cap 8) eventualmente
lo expulsaba y el LLM podía regenerarlo en chunks 9+.

P1-5 agrega:
  1. `top_repeated_meal_names` en lifetime_summary (cross-chunk).
  2. `permanent_meal_blocklist` para meals con presencia en >=2 chunks distintos
     (señal fuerte de repetición sistémica).
  3. Cap dinámico en `_chunk_lessons.repeated_meal_names`: total_chunks*3, tope 30.
"""
import json
from copy import deepcopy
from unittest.mock import patch
from db_plans import _inherit_lifetime_lessons_from_prior_plan


def _lesson(chunk: int, repeated_meals=None, repeated_bases=None,
            timestamp_iso="2026-05-01T12:00:00+00:00"):
    return {
        "chunk": chunk,
        "timestamp": timestamp_iso,
        "repeated_meal_names": repeated_meals or [],
        "repeated_bases": repeated_bases or [],
        "rejected_meals_that_reappeared": [],
        "allergy_hits": [],
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
    }


def test_inherit_summary_includes_top_repeated_meal_names_when_recompute():
    """P0-1 inheritance: si filtramos por TTL, recomputamos summary y debe incluir
    top_repeated_meal_names ordenados por frecuencia cross-chunk."""
    from datetime import datetime, timezone, timedelta

    # 2 lecciones, ambas dentro del TTL
    fresh_iso = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    stale_iso = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    history = [
        _lesson(1, repeated_meals=["Pollo al horno", "Arroz blanco"], timestamp_iso=fresh_iso),
        _lesson(2, repeated_meals=["Pollo al horno", "Pasta carbonara"], timestamp_iso=fresh_iso),
        _lesson(3, repeated_meals=["Lasaña vieja"], timestamp_iso=stale_iso),  # filtrado
    ]
    plan_data = {
        "_lifetime_lessons_history": history,
        "_lifetime_lessons_summary": {"_lifetime_window_days": 60},
    }

    with patch(
        "db_plans.execute_sql_query",
        return_value={"id": "prior-1", "plan_data": plan_data},
    ):
        result = _inherit_lifetime_lessons_from_prior_plan(None, "user-x")

    assert result is not None
    summary = result["summary"]
    # Pollo al horno aparece en 2 chunks → debe estar en blocklist permanente.
    assert "Pollo al horno" in summary["permanent_meal_blocklist"]
    # Single-chunk repeats van a top_repeated_meal_names pero NO al blocklist.
    assert "Arroz blanco" in summary["top_repeated_meal_names"]
    assert "Arroz blanco" not in summary["permanent_meal_blocklist"]
    # Lecciones expiradas no aparecen.
    assert "Lasaña vieja" not in summary["top_repeated_meal_names"]


def test_inherit_meal_blocklist_orders_by_chunk_count():
    """Meals con más chunks distintos van primero en el blocklist."""
    from datetime import datetime, timezone, timedelta
    fresh = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    history = [
        _lesson(1, repeated_meals=["Plato A", "Plato B"], timestamp_iso=fresh),
        _lesson(2, repeated_meals=["Plato A", "Plato B"], timestamp_iso=fresh),
        _lesson(3, repeated_meals=["Plato A"], timestamp_iso=fresh),  # 3 chunks
        _lesson(4, repeated_meals=["Plato C", "Plato C"], timestamp_iso=fresh),  # solo chunk 4 → 1 chunk
        _lesson(5, repeated_meals=["Lasaña vieja"], timestamp_iso=(
            datetime.now(timezone.utc) - timedelta(days=400)
        ).isoformat()),  # filtrado
    ]
    plan_data = {
        "_lifetime_lessons_history": history,
        "_lifetime_lessons_summary": {},
    }

    with patch(
        "db_plans.execute_sql_query",
        return_value={"id": "prior-2", "plan_data": plan_data},
    ):
        result = _inherit_lifetime_lessons_from_prior_plan(None, "user-x")

    summary = result["summary"]
    # top_repeated_meal_names ordena por presencia descendente: Plato A (3), Plato B (2), Plato C (1).
    assert summary["top_repeated_meal_names"][0] == "Plato A"
    assert summary["top_repeated_meal_names"][1] == "Plato B"
    # Blocklist: solo Plato A y B (>=2 chunks). Plato C tiene solo 1 chunk distinto.
    assert "Plato A" in summary["permanent_meal_blocklist"]
    assert "Plato B" in summary["permanent_meal_blocklist"]
    assert "Plato C" not in summary["permanent_meal_blocklist"]


def test_blocklist_caps_at_50():
    """Blocklist no crece sin límite — cap defensivo de 50."""
    from datetime import datetime, timezone, timedelta
    fresh = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    # 100 meals únicos cada uno en 2 chunks → todos califican.
    history = [
        _lesson(1, repeated_meals=[f"Meal {i}" for i in range(100)], timestamp_iso=fresh),
        _lesson(2, repeated_meals=[f"Meal {i}" for i in range(100)], timestamp_iso=fresh),
    ]
    plan_data = {"_lifetime_lessons_history": history, "_lifetime_lessons_summary": {}}

    with patch(
        "db_plans.execute_sql_query",
        return_value={"id": "prior-3", "plan_data": plan_data},
    ):
        result = _inherit_lifetime_lessons_from_prior_plan(None, "user-x")

    summary = result["summary"]
    assert len(summary["permanent_meal_blocklist"]) <= 50
    assert len(summary["top_repeated_meal_names"]) <= 30


def test_top_repeated_meal_names_caps_at_30():
    """top_repeated_meal_names cap de 30 (vs hardcoded 10 anterior)."""
    from datetime import datetime, timezone, timedelta
    fresh = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    # 50 meals únicos en 1 chunk c/u → no van a blocklist pero sí a top_repeated_meal_names.
    history = [
        _lesson(i, repeated_meals=[f"Single Meal {i}"], timestamp_iso=fresh)
        for i in range(1, 51)
    ]
    plan_data = {"_lifetime_lessons_history": history, "_lifetime_lessons_summary": {}}

    with patch(
        "db_plans.execute_sql_query",
        return_value={"id": "prior-4", "plan_data": plan_data},
    ):
        result = _inherit_lifetime_lessons_from_prior_plan(None, "user-x")

    summary = result["summary"]
    assert len(summary["top_repeated_meal_names"]) == 30
    assert len(summary["permanent_meal_blocklist"]) == 0  # ninguno repitió en 2+ chunks


def test_inheritance_preserves_legacy_fields():
    """Los campos pre-existentes (top_rejection_hits, top_repeated_bases) siguen funcionando."""
    from datetime import datetime, timezone, timedelta
    fresh = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    history = [
        {
            "chunk": 1,
            "timestamp": fresh,
            "rejection_violations": 1,
            "allergy_violations": 0,
            "rejected_meals_that_reappeared": ["Bistec encebollado"],
            "repeated_bases": [{"bases": ["res", "cebolla"]}],
            "repeated_meal_names": [],
            "allergy_hits": [],
        }
    ]
    plan_data = {"_lifetime_lessons_history": history, "_lifetime_lessons_summary": {}}

    with patch(
        "db_plans.execute_sql_query",
        return_value={"id": "prior-5", "plan_data": plan_data},
    ):
        result = _inherit_lifetime_lessons_from_prior_plan(None, "user-x")

    summary = result["summary"]
    assert "Bistec encebollado" in summary["top_rejection_hits"]
    assert "res" in summary["top_repeated_bases"]
    assert summary["total_rejection_violations"] == 1
