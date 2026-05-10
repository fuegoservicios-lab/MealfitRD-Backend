"""
Tests P1-8: Implicit proxy ratio=0 cuando zero-log Y zero mutations.

Antes: si el usuario no logueaba comidas (`explicit_logged=0`), el cálculo
implicit proxy retornaba `ratio = min(0.5 + mutations/max(planned, 6), 0.85)`.
Con `consumption_mutations_count=0`, eso daba `ratio=0.5` — "50% adherencia
asumida" sin ninguna evidencia. El usuario podría haber comido takeout 100%
del chunk y `learning_metrics` registraba 50% adherencia, envenenando el
contexto del LLM en chunks subsiguientes.

Cambio P1-8: si `zero_log_proxy=True` Y `consumption_mutations_count=0`:
  - `ratio = 0.0` (refleja "no hay evidencia").
  - `matched = 0`.
  - Nuevo flag `zero_log_no_mutations=True` en el output, propagado por
    `_check_chunk_learning_ready` para que dashboards y rebuild de lecciones
    distingan "sin señal" vs "ratio bajo medido".

Para sparse-log (algún log + 0 mutations) preservamos la fórmula original
porque al menos hay UNA señal explícita.

Cubre:
  1. zero-log + 0 mutations → ratio=0.0, flag=True.
  2. zero-log + mutations >= MIN → ratio>0 (proxy clásico), flag=False.
  3. sparse-log + 0 mutations → fórmula original (NO override), flag=False.
  4. logged completo + 0 mutations → ratio normal (no proxy), flag=False.
  5. planned_total=0 → ratio=1.0 default (caso edge sin chunk previo).
  6. _check_chunk_learning_ready propaga el flag en su return.
"""
import pytest
from unittest.mock import patch
from cron_tasks import _calculate_chunk_consumption_ratio


def _day_with_meals(day_num, meal_names):
    return {
        "day": day_num,
        "meals": [{"name": n, "type": "Almuerzo"} for n in meal_names],
    }


# ---------------------------------------------------------------------------
# _calculate_chunk_consumption_ratio
# ---------------------------------------------------------------------------
def test_p1_8_zero_log_zero_mutations_returns_zero_ratio():
    """zero-log + 0 mutations → ratio=0.0, flag=True (NO 0.5 inventado)."""
    days = [
        _day_with_meals(1, ["Pollo asado", "Avena"]),
        _day_with_meals(2, ["Salmón", "Yogur"]),
    ]
    result = _calculate_chunk_consumption_ratio(
        previous_chunk_days=days,
        consumed_records=[],  # zero log
        consumption_mutations_count=0,  # zero mutations
    )
    assert result["zero_log_proxy"] is True
    assert result["zero_log_no_mutations"] is True
    assert result["ratio"] == 0.0
    assert result["matched_meals"] == 0
    assert result["planned_meals"] == 4


def test_p1_8_zero_log_with_mutations_uses_proxy():
    """zero-log + mutations > 0 → fórmula proxy clásica, flag=False."""
    days = [_day_with_meals(1, ["P1", "P2", "P3", "P4", "P5", "P6"])]  # planned=6
    result = _calculate_chunk_consumption_ratio(
        previous_chunk_days=days,
        consumed_records=[],
        consumption_mutations_count=3,
    )
    assert result["zero_log_proxy"] is True
    assert result["zero_log_no_mutations"] is False
    # Fórmula: min(0.5 + 3/6, 0.85) = 0.85 (capped)
    assert result["ratio"] == 0.85
    assert result["matched_meals"] == 6  # asume planned cuando proxy, NO 0


def test_p1_8_sparse_log_with_zero_mutations_preserves_legacy():
    """
    sparse-log (algún log <25%) + 0 mutations: NO aplicar override, fórmula clásica.
    El log explícito mismo es señal mínima de actividad.
    """
    days = [_day_with_meals(d, [f"M{d}-{i}" for i in range(4)]) for d in range(1, 4)]
    # planned_total=12, sparse threshold = max(2, 12*0.25) = 3 → log < 3 = sparse
    result = _calculate_chunk_consumption_ratio(
        previous_chunk_days=days,
        consumed_records=[{"meal_name": "M1-0"}],  # 1 log < 3 = sparse
        consumption_mutations_count=0,
    )
    assert result["sparse_logging_proxy"] is True
    assert result["zero_log_proxy"] is False
    assert result["zero_log_no_mutations"] is False  # solo zero-log dispara el flag
    # Fórmula original: min(0.5 + 0/12, 0.85) = 0.5
    assert result["ratio"] == 0.5


def test_p1_8_full_log_no_proxy():
    """Logged >= 25% del planned → no proxy, ratio normal sin override."""
    days = [_day_with_meals(1, ["Pollo asado", "Avena", "Pasta"])]
    result = _calculate_chunk_consumption_ratio(
        previous_chunk_days=days,
        consumed_records=[
            {"meal_name": "Pollo asado"},
            {"meal_name": "Avena"},
        ],
        consumption_mutations_count=0,  # irrelevante: no proxy
    )
    assert result["zero_log_proxy"] is False
    assert result["sparse_logging_proxy"] is False
    assert result["zero_log_no_mutations"] is False
    # ratio = 2/3 = 0.6667
    assert abs(result["ratio"] - 2 / 3) < 0.001


def test_p1_8_no_planned_meals_returns_default_ratio():
    """Edge: planned_total=0 → ratio=1.0 (no aplica proxy)."""
    result = _calculate_chunk_consumption_ratio(
        previous_chunk_days=[],
        consumed_records=[],
        consumption_mutations_count=0,
    )
    assert result["planned_meals"] == 0
    assert result["zero_log_proxy"] is False
    assert result["zero_log_no_mutations"] is False
    assert result["ratio"] == 1.0


def test_p1_8_zero_log_no_mutations_default_field_to_false():
    """Cualquier resultado SIN zero_log_no_mutations debe tener default False."""
    days = [_day_with_meals(1, ["A", "B"])]
    result = _calculate_chunk_consumption_ratio(
        previous_chunk_days=days,
        consumed_records=[{"meal_name": "A"}, {"meal_name": "B"}],
        consumption_mutations_count=0,
    )
    assert result["zero_log_no_mutations"] is False


# ---------------------------------------------------------------------------
# Propagación a _check_chunk_learning_ready
# ---------------------------------------------------------------------------
@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.get_inventory_activity_since")
@patch("cron_tasks.get_consumed_meals_since")
@patch("cron_tasks.execute_sql_query")
def test_p1_8_check_chunk_propagates_flag_in_result(
    mock_query, mock_consumed, mock_activity, mock_now
):
    """`_check_chunk_learning_ready` expone `zero_log_no_mutations` en su return."""
    from cron_tasks import _check_chunk_learning_ready
    from datetime import datetime, timezone

    mock_now.return_value = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    mock_consumed.return_value = []  # zero log
    mock_activity.return_value = {
        "mutations_count": 0,
        "consumption_mutations_count": 0,  # zero mutations
        "manual_mutations_count": 0,
    }

    def query_side(q, params=None, **kw):
        if "health_profile" in (q or "").lower():
            return {"health_profile": {"tz_offset_minutes": 0}}
        return None
    mock_query.side_effect = query_side

    plan_data = {"days": [
        {"day": 1, "meals": [{"name": "M1", "ingredients": ["x"]}]},
        {"day": 2, "meals": [{"name": "M2", "ingredients": ["y"]}]},
        {"day": 3, "meals": [{"name": "M3", "ingredients": ["z"]}]},
    ]}
    snapshot = {"form_data": {"_plan_start_date": "2026-04-30"}}

    result = _check_chunk_learning_ready(
        user_id="u-p18",
        meal_plan_id="m-p18",
        week_number=2,
        days_offset=3,
        plan_data=plan_data,
        snapshot=snapshot,
    )

    # El flag debe propagarse al return
    assert "zero_log_no_mutations" in result
    assert result["zero_log_no_mutations"] is True
    # Y el ratio expuesto debe ser 0 (no 0.5)
    assert result.get("ratio") == 0.0
    # ready=False porque no hay señal real
    assert result["ready"] is False


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.get_inventory_activity_since")
@patch("cron_tasks.get_consumed_meals_since")
@patch("cron_tasks.execute_sql_query")
def test_p1_8_check_chunk_flag_false_when_mutations_present(
    mock_query, mock_consumed, mock_activity, mock_now
):
    """Con mutations >= MIN, flag=False y proxy aplicado."""
    from cron_tasks import _check_chunk_learning_ready
    from datetime import datetime, timezone
    import constants

    constants.CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS = 2

    mock_now.return_value = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    mock_consumed.return_value = []
    mock_activity.return_value = {
        "mutations_count": 3,
        "consumption_mutations_count": 3,
        "manual_mutations_count": 0,
    }

    def query_side(q, params=None, **kw):
        if "health_profile" in (q or "").lower():
            return {"health_profile": {"tz_offset_minutes": 0}}
        return None
    mock_query.side_effect = query_side

    plan_data = {"days": [
        {"day": 1, "meals": [{"name": f"M{i}", "ingredients": ["x"]}]}
        for i in range(1, 4)
    ]}
    snapshot = {"form_data": {"_plan_start_date": "2026-04-30"}}

    result = _check_chunk_learning_ready(
        user_id="u-p18b",
        meal_plan_id="m-p18b",
        week_number=2,
        days_offset=3,
        plan_data=plan_data,
        snapshot=snapshot,
    )

    assert result["zero_log_no_mutations"] is False
    assert result["zero_log_proxy"] is True  # sigue siendo zero-log
    assert result["inventory_proxy_used"] is True  # proxy aprobado
    assert result["ratio"] > 0  # NO 0; fórmula proxy aplicó
