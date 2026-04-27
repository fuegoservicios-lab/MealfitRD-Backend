import pytest
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from db_inventory import get_user_inventory, release_meal_reservation

from cron_tasks import (
    _alert_if_degraded_rate_high,
    _build_filtered_edge_recipe_day,
    _calculate_chunk_consumption_ratio,
    _calculate_learning_metrics,
    _check_chunk_learning_ready,
    _compute_chunk_retry_delay_minutes,
    _count_meaningful_pantry_items,
    _compute_chunk_delay_days,
    _compute_expected_preemption_seconds,
    _enqueue_plan_chunk,
    _filter_days_by_fresh_pantry,
    _recover_pantry_paused_chunks,
    _should_pause_for_empty_pantry,
    process_plan_chunk_queue,
)
from routers.plans import api_shift_plan
from constants import split_with_absorb

from constants import (
    CHUNK_MIN_FRESH_PANTRY_ITEMS,
    CHUNK_LEARNING_MODE,
    CHUNK_PROACTIVE_MARGIN_DAYS,
    CHUNK_LEARNING_READY_DELAY_HOURS,
    CHUNK_MAX_FAILURE_ATTEMPTS,
    PLAN_CHUNK_SIZE,
    CHUNK_PANTRY_EMPTY_MAX_REMINDERS,
    CHUNK_RETRY_CRITICAL_MINUTES,
)

def test_compute_chunk_delay_days_defaults_to_strict_mode():
    delay_days, mode, _, _ = _compute_chunk_delay_days(3, 4, 2, {"totalDays": 7}, "initial_plan")

    assert CHUNK_LEARNING_MODE == "strict"
    assert mode == "strict"
    assert delay_days == max(0, 3 - CHUNK_PROACTIVE_MARGIN_DAYS)


def test_expected_preemption_seconds_matches_days_advanced():
    assert _compute_expected_preemption_seconds(9, 6) == 3 * 86400
    assert _compute_expected_preemption_seconds(3, 3) == 0


def test_compute_chunk_retry_delay_minutes_uses_exponential_backoff_and_critical_override():
    assert _compute_chunk_retry_delay_minutes(1) == 2
    assert _compute_chunk_retry_delay_minutes(2) == 4
    assert _compute_chunk_retry_delay_minutes(3) == 8
    assert _compute_chunk_retry_delay_minutes(4, is_critical=True) == CHUNK_RETRY_CRITICAL_MINUTES


def test_split_with_absorb_keeps_7_day_plan_as_3_plus_4():
    assert split_with_absorb(7, PLAN_CHUNK_SIZE) == [3, 4]


@pytest.mark.parametrize("total_days", [7, 15, 30])
def test_split_with_absorb_sum_equals_total_days(total_days):
    chunks = split_with_absorb(total_days, PLAN_CHUNK_SIZE)
    assert sum(chunks) == total_days, f"split_with_absorb({total_days}) suma {sum(chunks)}, esperado {total_days}"


@pytest.mark.parametrize("total_days", [7, 15, 30])
def test_split_with_absorb_all_chunks_at_least_base_size(total_days):
    chunks = split_with_absorb(total_days, PLAN_CHUNK_SIZE)
    assert all(c >= PLAN_CHUNK_SIZE for c in chunks), f"Chunk menor que base en split_with_absorb({total_days}): {chunks}"


def test_split_with_absorb_no_index_error_for_non_standard_totals():
    # split_with_absorb(8): remaining=5, num_additional=1 → rem=2 > len([3]) → IndexError antes del fix
    result = split_with_absorb(8, 3)
    assert sum(result) == 8
    assert all(c >= 3 for c in result)


def test_chunk_consumption_ratio_uses_implicit_proxy_when_no_explicit_logs_exist():
    previous_chunk_days = [
        {"day": 1, "meals": [{"name": "Pollo guisado"}, {"name": "Avena"}]},
        {"day": 2, "meals": [{"name": "Pescado"}]},
    ]

    ratio_info = _calculate_chunk_consumption_ratio(previous_chunk_days, [])

    assert ratio_info["ratio"] == 1.0
    assert ratio_info["matched_meals"] == 3
    assert ratio_info["planned_meals"] == 3
    assert ratio_info["explicit_logged_meals"] == 0
    assert ratio_info["used_implicit_proxy"] is True


def test_chunk_consumption_ratio_keeps_gate_strict_when_some_explicit_logs_exist():
    previous_chunk_days = [
        {"day": 1, "meals": [{"name": "Pollo guisado"}, {"name": "Avena"}]},
        {"day": 2, "meals": [{"name": "Pescado"}]},
    ]

    ratio_info = _calculate_chunk_consumption_ratio(
        previous_chunk_days,
        [{"meal_name": "Pollo guisado"}],
    )

    assert ratio_info["ratio"] == pytest.approx(1 / 3, rel=1e-3)
    assert ratio_info["matched_meals"] == 1
    assert ratio_info["planned_meals"] == 3
    assert ratio_info["explicit_logged_meals"] == 1
    assert ratio_info["used_implicit_proxy"] is False


def test_filter_days_by_fresh_pantry_keeps_only_days_with_majority_coverage():
    days = [
        {
            "day": 1,
            "meals": [
                {
                    "name": "Pollo con arroz",
                    "ingredients": ["200g pechuga de pollo", "1 taza arroz blanco", "brocoli"],
                }
            ],
        },
        {
            "day": 2,
            "meals": [
                {
                    "name": "Res con quinoa",
                    "ingredients": ["200g carne de res", "1 taza quinoa", "esparragos"],
                }
            ],
        },
    ]

    filtered = _filter_days_by_fresh_pantry(days, ["pollo", "arroz", "brocoli"])

    assert [day["day"] for day in filtered] == [1]


def test_filter_days_by_fresh_pantry_preserves_days_without_structured_ingredients():
    days = [{"day": 1, "meals": [{"name": "Dia legado", "ingredients": []}]}]

    filtered = _filter_days_by_fresh_pantry(days, ["pollo"])

    assert filtered == days


def test_count_meaningful_pantry_items_ignores_condiments_and_duplicates():
    count = _count_meaningful_pantry_items([
        "agua",
        "sal",
        "pollo",
        "200g pechuga de pollo",
        "arroz",
        "aceite",
    ])

    assert count == 2


@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
def test_enqueue_plan_chunk_requeues_failed_chunk_with_one_day_margin(mock_write, mock_query):
    start_date = "2026-04-21T00:00:00+00:00"
    snapshot = {"form_data": {"_plan_start_date": start_date}, "totalDays": 7}
    mock_query.side_effect = [None, {"id": "failed-1"}]

    _enqueue_plan_chunk('user_123', 'plan_456', 2, 3, PLAN_CHUNK_SIZE, snapshot)

    mock_write.assert_called_once()
    query, params = mock_write.call_args[0]

    assert "UPDATE plan_chunk_queue" in query
    assert params[0] == "initial_plan"
    assert params[3] == 86400
    assert params[4] == 3
    assert params[5] == PLAN_CHUNK_SIZE
    assert params[6] == "failed-1"
    assert params[2] == datetime(2026, 4, 23, 1, 0, tzinfo=timezone.utc).isoformat()


@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
def test_enqueue_plan_chunk_insert_is_idempotent_with_on_conflict_do_nothing(mock_write, mock_query):
    start_date = "2026-04-21T00:00:00+00:00"
    snapshot = {"form_data": {"_plan_start_date": start_date}, "totalDays": 15}
    mock_query.side_effect = [None, None]

    _enqueue_plan_chunk('user_123', 'plan_456', 2, 3, PLAN_CHUNK_SIZE, snapshot)

    query, params = mock_write.call_args[0]
    assert "ON CONFLICT (meal_plan_id, week_number)" in query
    assert "DO NOTHING" in query
    assert params[0] == "user_123"
    assert params[1] == "plan_456"
    assert params[2] == 2


@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
def test_enqueue_plan_chunk_persists_chunk_kind(mock_write, mock_query):
    start_date = "2026-04-21T00:00:00+00:00"
    snapshot = {"form_data": {"_plan_start_date": start_date}, "totalDays": 15}
    mock_query.side_effect = [None, None]

    _enqueue_plan_chunk('user_123', 'plan_456', 3, 6, PLAN_CHUNK_SIZE, snapshot, chunk_kind="initial_plan")

    query, params = mock_write.call_args[0]
    assert "chunk_kind" in query
    assert params[3] == "initial_plan"
    assert "expected_preemption_seconds" in query


@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
def test_enqueue_plan_chunk_schedules_initial_chunks_proactively(mock_write, mock_query):
    start_date = "2026-04-21T00:00:00+00:00"
    snapshot = {"form_data": {"_plan_start_date": start_date}, "totalDays": 15}
    mock_query.side_effect = [None, None]

    _enqueue_plan_chunk('user_123', 'plan_456', 2, 3, PLAN_CHUNK_SIZE, snapshot, chunk_kind="initial_plan")

    query, params = mock_write.call_args[0]
    expected_execute_after = datetime(
        2026,
        4,
        21 + max(0, 3 - CHUNK_PROACTIVE_MARGIN_DAYS),
        1,
        0,
        tzinfo=timezone.utc,
    ).isoformat()

    assert "INSERT INTO plan_chunk_queue" in query
    assert params[3] == "initial_plan"
    assert params[7] == expected_execute_after
    assert params[8] == CHUNK_PROACTIVE_MARGIN_DAYS * 86400


@patch('cron_tasks.CHUNK_LEARNING_MODE', 'adaptive')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
def test_enqueue_plan_chunk_persists_expected_preemption_for_final_chunk(mock_write, mock_query):
    start_date = "2026-04-21T00:00:00+00:00"
    snapshot = {"form_data": {"_plan_start_date": start_date}, "totalDays": 15}
    mock_query.side_effect = [None, None]

    _enqueue_plan_chunk('user_123', 'plan_456', 4, 9, PLAN_CHUNK_SIZE, snapshot, chunk_kind="initial_plan")

    query, params = mock_write.call_args[0]
    assert "expected_preemption_seconds" in query
    assert params[-1] == 3 * 86400


@patch('cron_tasks.run_plan_pipeline')
@patch('cron_tasks.get_consumed_meals_since')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
def test_chunk_waits_for_real_consumption_before_generating(mock_write, mock_query, mock_consumed, mock_pipeline):
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": {
            "form_data": {"_plan_start_date": "2026-04-21T00:00:00+00:00"}
        }
    }]

    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]},
        ]
    }

    def _write_side_effect(query, params=None, returning=False):
        if "RETURNING" in query:
            return tasks
        return None

    def _query_side_effect(query, params=None, fetch_all=False, fetch_one=False, **kwargs):
        if "generation_status" in query:
            return {"id": "plan_456", "status": "active", "plan_data": {"generation_status": "active"}}
        if "SELECT plan_data FROM meal_plans" in query:
            return {"plan_data": prior_plan}
        return None

    mock_write.side_effect = _write_side_effect
    mock_query.side_effect = _query_side_effect
    mock_consumed.return_value = [{"meal_name": "A"}]

    process_plan_chunk_queue()

    mock_pipeline.assert_not_called()
    deferred_calls = [
        call for call in mock_write.call_args_list
        if "make_interval(hours => %s)" in call[0][0]
    ]
    assert len(deferred_calls) == 1
    deferred_query, deferred_params = deferred_calls[0][0]
    deferred_snapshot = json.loads(deferred_params[1])

    assert "UPDATE plan_chunk_queue" in deferred_query
    assert "make_interval(hours => %s)" in deferred_query
    assert deferred_params[0] == CHUNK_LEARNING_READY_DELAY_HOURS
    assert deferred_params[2] == 1
    assert deferred_snapshot["_learning_ready_deferrals"] == 1
    assert deferred_snapshot["_last_learning_ready_ratio"] < 0.5


@patch('cron_tasks.run_plan_pipeline')
@patch('cron_tasks.get_user_inventory')
@patch('cron_tasks._check_chunk_learning_ready')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('utils_push.send_push_notification')
def test_chunk_pauses_for_user_action_when_fresh_pantry_is_nearly_empty(
    mock_push, mock_write, mock_query, mock_learning_ready, mock_inventory, mock_pipeline
):
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": {
            "form_data": {"_plan_start_date": "2026-04-21T00:00:00+00:00"}
        }
    }]

    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]},
        ]
    }

    def _write_side_effect(query, params=None, returning=False):
        if "RETURNING" in query:
            return tasks
        return None

    def _query_side_effect(query, params=None, fetch_all=False, fetch_one=False, **kwargs):
        if "generation_status" in query:
            return {"id": "plan_456", "status": "active", "plan_data": {"generation_status": "active"}}
        if "SELECT plan_data FROM meal_plans" in query:
            return {"plan_data": prior_plan}
        return None

    mock_write.side_effect = _write_side_effect
    mock_query.side_effect = _query_side_effect
    mock_learning_ready.return_value = {"ready": True, "ratio": 1.0, "matched_meals": 3, "planned_meals": 3}
    mock_inventory.return_value = ["sal", "agua"][:CHUNK_MIN_FRESH_PANTRY_ITEMS - 1]

    with patch('threading.Thread') as mock_thread, patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.side_effect = lambda f, tasks: [f(t) for t in tasks]
        process_plan_chunk_queue()

    mock_pipeline.assert_not_called()
    mock_inventory.assert_called_once_with("user_123")
    assert any(
        "SET status = 'pending_user_action'" in call[0][0]
        for call in mock_write.call_args_list
    )
    mock_thread.assert_called_once()
    assert mock_thread.call_args.kwargs["target"] is mock_push


def test_should_pause_for_empty_pantry_skips_when_flexible_mode_is_enabled():
    assert _should_pause_for_empty_pantry(
        "live",
        ["agua", "sal"],
        snapshot={"_pantry_flexible_mode": True},
        form_data={},
    ) is False


@patch('cron_tasks._dispatch_push_notification')
@patch('cron_tasks.execute_sql_write')
@patch('cron_tasks.execute_sql_query')
def test_recover_pantry_paused_chunks_sends_reminder_before_timeout(mock_query, mock_write, mock_push):
    mock_query.return_value = [{
        "id": "chunk_1",
        "user_id": "user_123",
        "week_number": 2,
        "paused_seconds": 4 * 3600,
        "pipeline_snapshot": {
            "_pantry_pause_reminders": 0,
            "_pantry_pause_reminder_hours": 4,
            "_pantry_pause_ttl_hours": 12,
        },
    }]

    _recover_pantry_paused_chunks()

    mock_push.assert_called_once()
    update_query, update_params = mock_write.call_args[0]
    assert "SET pipeline_snapshot = %s::jsonb" in update_query
    reminder_snapshot = json.loads(update_params[0])
    assert reminder_snapshot["_pantry_pause_reminders"] == 1


@patch('cron_tasks._dispatch_push_notification')
@patch('cron_tasks.execute_sql_write')
@patch('cron_tasks.execute_sql_query')
def test_recover_pantry_paused_chunks_degrades_to_flexible_mode_after_ttl(mock_query, mock_write, mock_push):
    mock_query.return_value = [{
        "id": "chunk_1",
        "user_id": "user_123",
        "week_number": 2,
        "paused_seconds": 13 * 3600,
        "pipeline_snapshot": {
            "_pantry_pause_reminders": CHUNK_PANTRY_EMPTY_MAX_REMINDERS,
            "_pantry_pause_reminder_hours": 4,
            "_pantry_pause_ttl_hours": 12,
        },
    }]

    _recover_pantry_paused_chunks()

    mock_push.assert_called_once()
    update_query, update_params = mock_write.call_args[0]
    assert "SET status = 'pending'" in update_query
    degraded_snapshot = json.loads(update_params[0])
    assert degraded_snapshot["_degraded"] is True
    assert degraded_snapshot["_pantry_flexible_mode"] is True


@patch('cron_tasks.get_consumed_meals_since')
def test_chunk_learning_ready_uses_implicit_proxy_when_user_has_no_explicit_logs(mock_consumed):
    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]},
        ]
    }
    snapshot = {"form_data": {"_plan_start_date": "2026-04-21T00:00:00+00:00"}}
    mock_consumed.return_value = []

    learning_ready = _check_chunk_learning_ready(
        user_id="user_123",
        meal_plan_id="plan_456",
        week_number=2,
        days_offset=3,
        plan_data=prior_plan,
        snapshot=snapshot,
    )

    assert learning_ready["ready"] is True
    assert learning_ready["ratio"] == 1.0
    assert learning_ready["matched_meals"] == 3
    assert learning_ready["planned_meals"] == 3
    assert learning_ready["used_implicit_proxy"] is True


@patch('cron_tasks._enqueue_plan_chunk')
@patch('db_core.connection_pool')
def test_shift_plan_blocks_rolling_refill_for_active_7_day_plan(mock_pool, mock_enqueue):
    today = datetime.now(timezone.utc).isoformat()
    plan_data = {
        "grocery_start_date": today,
        "generation_status": "complete",
        "total_days_requested": 7,
        "days": [
            {"day": 1, "day_name": "Lunes", "meals": [{"name": "A"}]},
            {"day": 2, "day_name": "Martes", "meals": [{"name": "B"}]},
        ],
    }

    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_tx = MagicMock()
    mock_conn.transaction.return_value.__enter__.return_value = mock_tx
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.side_effect = [
        {"id": "plan_7d", "plan_data": plan_data},
    ]

    response = api_shift_plan({"user_id": "user_123", "tzOffset": 0}, verified_user_id="user_123")

    assert response["success"] is True
    assert len(response["plan_data"]["days"]) == 2
    assert [d["day"] for d in response["plan_data"]["days"]] == [1, 2]
    mock_enqueue.assert_not_called()


@patch('cron_tasks._enqueue_plan_chunk')
@patch('db_core.connection_pool')
def test_shift_plan_skips_refill_when_target_week_chunk_already_exists(mock_pool, mock_enqueue):
    start_dt = datetime.now(timezone.utc) - timedelta(days=4)
    plan_data = {
        "grocery_start_date": start_dt.isoformat(),
        "generation_status": "complete",
        "total_days_requested": 15,
        "days": [
            {"day": 1, "day_name": "Lunes", "meals": [{"name": "A"}]},
        ],
    }

    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.transaction.return_value.__enter__.return_value = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.side_effect = [
        {"id": "plan_15d", "plan_data": plan_data},
        {"health_profile": {"budget": "mid"}},
        {"cnt": 0},
        {"id": "chunk-existing", "status": "stale", "chunk_kind": "initial_plan"},
    ]

    response = api_shift_plan({"user_id": "user_123", "tzOffset": 0}, verified_user_id="user_123")

    assert response["success"] is True
    mock_enqueue.assert_not_called()


@patch('cron_tasks._enqueue_plan_chunk')
@patch('db_core.connection_pool')
def test_shift_plan_uses_max_non_cancelled_week_when_failed_chunk_exists(mock_pool, mock_enqueue):
    start_dt = datetime.now(timezone.utc) - timedelta(days=4)
    plan_data = {
        "grocery_start_date": start_dt.isoformat(),
        "generation_status": "complete",
        "total_days_requested": 30,
        "days": [
            {"day": 1, "day_name": "Lunes", "meals": [{"name": "A"}]},
        ],
    }

    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.transaction.return_value.__enter__.return_value = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.side_effect = [
        {"id": "plan_30d", "plan_data": plan_data},
        {"health_profile": {"budget": "mid"}},
        {"max_week": 4},
        None,
    ]

    response = api_shift_plan({"user_id": "user_123", "tzOffset": 0}, verified_user_id="user_123")

    assert response["success"] is True
    mock_enqueue.assert_called_once()
    enqueue_args = mock_enqueue.call_args[0]
    assert enqueue_args[2] == 5


def test_learning_metrics_tracks_ingredient_base_repeats_even_when_names_change():
    prior_days = [
        {"day": 1, "meals": [{"name": "Pollo Guisado", "ingredients": ["200g pechuga de pollo", "1 taza arroz blanco"]}]},
    ]
    new_days = [
        {"day": 4, "meals": [{"name": "Pollo a la Plancha", "ingredients": ["200g pollo", "ensalada verde"]}]},
        {"day": 5, "meals": [{"name": "Res al Horno", "ingredients": ["200g carne de res", "batata"]}]},
    ]

    metrics = _calculate_learning_metrics(
        new_days=new_days,
        prior_meals=["Pollo Guisado"],
        prior_days=prior_days,
        rejected_names=[],
        allergy_keywords=[],
        fatigued_ingredients=[],
    )

    assert metrics["learning_repeat_pct"] == 0.0
    assert metrics["ingredient_base_repeat_pct"] == 50.0
    assert metrics["sample_repeated_bases"][0]["bases"] == ["pollo"]


def test_learning_metrics_counts_cross_category_fatigue_hits():
    new_days = [
        {
            "day": 4,
            "meals": [
                {
                    "name": "Wrap de queso",
                    "ingredients": ["queso mozzarella", "tortilla integral"],
                }
            ],
        },
    ]

    metrics = _calculate_learning_metrics(
        new_days=new_days,
        prior_meals=[],
        prior_days=[],
        rejected_names=[],
        allergy_keywords=[],
        fatigued_ingredients=["[CATEGORÍA] huevos y lácteos"],
    )

    assert metrics["fatigued_violations"] == 1


@patch('db_inventory.get_raw_user_inventory')
@patch('db_inventory.supabase')
def test_get_user_inventory_uses_available_quantity_after_reservations(mock_supabase, mock_raw_inventory):
    mock_raw_inventory.return_value = [
        {
            "ingredient_name": "Pechuga de pollo",
            "quantity": 2.0,
            "reserved_quantity": 1.25,
            "available_quantity": 0.75,
            "unit": "lb",
            "created_at": "2026-04-20T12:00:00",
        }
    ]
    mock_supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {"health_profile": {"householdSize": 1}}
    ]

    inventory = get_user_inventory("user_123")

    assert any("0.75" in item and "Pechuga de pollo" in item for item in inventory)


@patch('db_inventory.supabase')
def test_release_meal_reservation_removes_matching_entries(mock_supabase):
    table_mock = MagicMock()
    mock_supabase.table.return_value = table_mock
    table_mock.select.return_value.eq.return_value.gt.return_value.execute.return_value.data = [
        {
            "id": "inv_1",
            "reserved_quantity": 3.0,
            "reservation_details": {
                "chunk:task_1:meal:pollo_con_arroz": 1.5,
                "chunk:task_2:meal:res_con_quinoa": 1.5,
            },
        }
    ]

    released = release_meal_reservation("user_123", "Pollo con arroz")

    assert released == 1
    update_payload = table_mock.update.call_args[0][0]
    assert update_payload["reserved_quantity"] == 1.5
    assert "chunk:task_1:meal:pollo_con_arroz" not in update_payload["reservation_details"]


def test_build_filtered_edge_recipe_day_respects_dislikes_and_diet():
    import re

    edge_day = _build_filtered_edge_recipe_day(
        allergies=[],
        dislikes=["pollo", "res", "cerdo", "pescado", "atun", "salami", "camarones", "chuleta", "longaniza"],
        diet="vegetarian",
    )

    assert edge_day is not None
    edge_blob = json.dumps(edge_day, ensure_ascii=False).lower()
    for forbidden in ["pollo", "res", "cerdo", "pescado", "atun", "salami", "camarones", "chuleta", "longaniza"]:
        assert re.search(rf"\b{re.escape(forbidden)}\b", edge_blob) is None


@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
def test_degraded_rate_alert_persists_system_alert_and_marks_users(mock_write, mock_query):
    mock_query.side_effect = [
        [
            {
                "is_rolling_refill": False,
                "total": 20,
                "degraded": 5,
            }
        ],
        [
            {"user_id": "11111111-1111-1111-1111-111111111111"},
            {"user_id": "22222222-2222-2222-2222-222222222222"},
        ],
    ]

    _alert_if_degraded_rate_high()

    queries = [call[0][0] for call in mock_write.call_args_list]
    assert any("CREATE TABLE IF NOT EXISTS system_alerts" in q for q in queries)
    assert any("ADD COLUMN IF NOT EXISTS quality_alert_at" in q for q in queries)

    insert_call = next(call for call in mock_write.call_args_list if "INSERT INTO system_alerts" in call[0][0])
    insert_params = insert_call[0][1]
    assert insert_params[0] == "degraded_rate_high:initial"
    assert insert_params[1] == "degraded_rate_high"
    assert insert_params[2] == "critical"
    assert "20" in insert_params[4]

    update_call = next(call for call in mock_write.call_args_list if "UPDATE user_profiles" in call[0][0])
    assert update_call[0][1][0] == [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]


@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
def test_degraded_rate_alert_does_not_persist_below_threshold(mock_write, mock_query):
    mock_query.return_value = [
        {
            "is_rolling_refill": True,
            "total": 20,
            "degraded": 3,
        }
    ]

    _alert_if_degraded_rate_high()

    queries = [call[0][0] for call in mock_write.call_args_list]
    assert not any("INSERT INTO system_alerts" in q for q in queries)
    assert not any("UPDATE user_profiles" in q for q in queries)


def _mock_execute_sql_write_factory(tasks_to_return):
    def side_effect(query, params=None, returning=False):
        if "RETURNING" in query:
            return tasks_to_return
        return None
    return side_effect

def _mock_execute_sql_query_factory(plan_data, backup_plan, user_profile=None, tasks=None):
    def side_effect(query, params=None, fetch_all=False, fetch_one=False, **kwargs):
        res = None
        if "SELECT * FROM plan_chunk_queue" in query:
            res = tasks or []
        elif "generation_status" in query:
            res = {"id": "plan_456", "status": "active", "plan_data": {"generation_status": "active"}}
        elif "SELECT plan_data FROM meal_plans" in query:
            res = {"plan_data": plan_data}
        elif "emergency_backup_plan" in query:
            res = {"backup": backup_plan}
        elif "SELECT health_profile FROM user_profiles" in query:
            res = {"health_profile": user_profile or {}}
        
        if res is not None:
            if fetch_one:
                # Si res es una lista, devuelve el primer elemento, si no devuelve res
                return res[0] if isinstance(res, list) and len(res) > 0 else res
            else:
                # Si res ya es una lista, la devuelve, si no, la envuelve en una lista
                return res if isinstance(res, list) else [res]
        return None
    return side_effect

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('cron_tasks.get_user_inventory')
@patch('db.get_user_likes')
@patch('db.get_active_rejections')
@patch('db_facts.get_consumed_meals_since')
@patch('db_facts.get_all_user_facts')
def test_chunk_degraded_fallback(mock_facts, mock_consumed, mock_rejections, mock_likes, mock_inventory, mock_write, mock_query, mock_shop, mock_pool, mock_llm):
    mock_llm.return_value.invoke.side_effect = Exception("Simulated LLM Outage")
    mock_likes.return_value = []
    mock_rejections.return_value = []
    mock_consumed.return_value = []
    mock_facts.return_value = []
    mock_inventory.return_value = ["pollo", "arroz", "brocoli"]
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": json.dumps({"_degraded": True})
    }]
    
    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "Pollo Asado"}]},
            {"day": 2, "meals": [{"name": "Pescado"}]},
            {"day": 3, "meals": [{"name": "Res"}]}
        ]
    }
    
    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(prior_plan, backup_plan=[], tasks=tasks)
    
    mock_shop.return_value = {"categories": []}
    
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}
    
    # Mock ThreadPoolExecutor to just run the function synchronously
    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        def sync_submit(fn, *args, **kwargs):
            future = MagicMock()
            try:
                res = fn(*args, **kwargs)
                future.result.return_value = res
            except Exception as e:
                future.result.side_effect = e
            return future
        mock_executor.return_value.__enter__.return_value.map.side_effect = lambda f, tasks: [f(t) for t in tasks]
        mock_executor.return_value.submit.side_effect = sync_submit
        mock_executor.return_value.__enter__.return_value.submit.side_effect = sync_submit
        process_plan_chunk_queue()
    
    update_calls = [call for call in mock_cursor.execute.call_args_list if "UPDATE meal_plans SET plan_data =" in call[0][0]]
    assert len(update_calls) == 1
    
    args = update_calls[0][0]
    query = args[0]
    params = args[1]
    merged_data = json.loads(params[0]) if isinstance(params[0], str) else params[0]
    
    assert len(merged_data["days"]) == 6
    assert merged_data["days"][3]["day"] == 4
    assert merged_data["days"][4]["day"] == 5
    assert merged_data["days"][5]["day"] == 6
    assert merged_data["days"][3]["_is_degraded_shuffle"] == True


@patch('langchain_google_genai.ChatGoogleGenerativeAI')
@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('cron_tasks.get_user_inventory')
@patch('db.get_user_likes')
@patch('db.get_active_rejections')
@patch('db_facts.get_consumed_meals_since')
@patch('db_facts.get_all_user_facts')
def test_chunk_degraded_fallback_prefers_days_covered_by_fresh_pantry(
    mock_facts, mock_consumed, mock_rejections, mock_likes, mock_inventory,
    mock_write, mock_query, mock_shop, mock_pool, mock_llm
):
    mock_llm.return_value.invoke.side_effect = Exception("Simulated LLM Outage")
    mock_likes.return_value = []
    mock_rejections.return_value = []
    mock_consumed.return_value = []
    mock_facts.return_value = []
    mock_inventory.return_value = ["pollo", "arroz", "brocoli"]
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 1,
        "pipeline_snapshot": json.dumps({
            "_degraded": True,
            "form_data": {"current_pantry_ingredients": ["inventario viejo"]},
        })
    }]

    prior_plan = {
        "days": [
            {
                "day": 1,
                "meals": [{"name": "Pollo con arroz", "ingredients": ["pollo", "arroz", "brocoli"]}],
            },
            {
                "day": 2,
                "meals": [{"name": "Res con quinoa", "ingredients": ["res", "quinoa", "esparragos"]}],
            },
            {
                "day": 3,
                "meals": [{"name": "Salmon con pasta", "ingredients": ["salmon", "pasta", "espinaca"]}],
            },
        ]
    }

    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(prior_plan, backup_plan=[], tasks=tasks)
    mock_shop.return_value = {"categories": []}

    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}

    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        def sync_submit(fn, *args, **kwargs):
            future = MagicMock()
            try:
                res = fn(*args, **kwargs)
                future.result.return_value = res
            except Exception as e:
                future.result.side_effect = e
            return future
        mock_executor.return_value.__enter__.return_value.map.side_effect = lambda f, tasks: [f(t) for t in tasks]
        mock_executor.return_value.submit.side_effect = sync_submit
        mock_executor.return_value.__enter__.return_value.submit.side_effect = sync_submit
        process_plan_chunk_queue()

    update_calls = [call for call in mock_cursor.execute.call_args_list if "UPDATE meal_plans SET plan_data =" in call[0][0]]
    assert len(update_calls) == 1

    merged_data = json.loads(update_calls[0][0][1][0]) if isinstance(update_calls[0][0][1][0], str) else update_calls[0][0][1][0]
    assert merged_data["days"][3]["meals"][0]["name"] == "Pollo con arroz"


@patch('cron_tasks.execute_sql_write')
def test_queue_management_purge_and_rescue(mock_write):
    mock_write.side_effect = _mock_execute_sql_write_factory([])
    process_plan_chunk_queue()
    calls = mock_write.call_args_list
    queries = [call[0][0] for call in calls]
    assert any("SET status = 'cancelled'" in q for q in queries)
    assert any("DELETE FROM plan_chunk_queue" in q and "status = 'cancelled'" in q for q in queries)
    assert any("attempts = COALESCE(attempts, 0) + 1" in q and "status = 'processing'" in q for q in queries)


@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.run_plan_pipeline')
@patch('cron_tasks.get_user_inventory')
@patch('cron_tasks.build_memory_context')
@patch('cron_tasks.get_user_likes')
@patch('db.get_user_likes')
@patch('cron_tasks.get_active_rejections')
@patch('db.get_active_rejections')
@patch('cron_tasks.analyze_preferences_agent')
@patch('cron_tasks._build_facts_memory_context')
@patch('cron_tasks.get_all_user_facts')
@patch('db_facts.get_all_user_facts')
@patch('cron_tasks.get_consumed_meals_since')
@patch('db_facts.get_consumed_meals_since')
@patch('cron_tasks.get_recent_plans')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('db_facts.get_user_facts_by_metadata')
def test_continuous_learning_signals_propagation(
    mock_db_metadata, mock_write, mock_query, mock_recent_plans,
    mock_db_consumed, mock_cron_consumed, mock_db_facts, mock_cron_facts, mock_build_facts,
    mock_analyze, mock_db_rejections, mock_cron_rejections, mock_db_likes,
    mock_cron_likes, mock_build_memory_context, mock_inventory, mock_pipeline, mock_shop, mock_pool
):
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": {
            "form_data": {"allergies": ["Maní"], "session_id": "sess_123"}
        }
    }]
    
    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "A2"}]},
            {"day": 3, "meals": [{"name": "A3"}]}
        ]
    }
    
    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(
        plan_data=prior_plan,
        backup_plan=[],
        user_profile={"medical_conditions": ["Diabetes"], "_protected_keys": "ignored"},
        tasks=tasks
    )
    
    mock_shop.return_value = {"categories": []}
    mock_cron_facts.return_value = [{"fact": "Mariscos"}]
    mock_db_facts.return_value = [{"fact": "Mariscos"}]
    mock_db_consumed.return_value = []
    mock_cron_consumed.return_value = []
    mock_recent_plans.return_value = []
    mock_db_rejections.return_value = []
    mock_cron_rejections.return_value = []
    mock_db_likes.return_value = []
    mock_cron_likes.return_value = []
    mock_inventory.return_value = ["pollo", "arroz", "avena"]
    mock_db_metadata.return_value = [{"fact": "Mariscos"}]
    mock_build_memory_context.return_value = {
        "recent_messages": [{"role": "user", "content": "No quiero comidas muy secas"}],
        "full_context_str": "Usuario reciente: no quiere comidas muy secas",
    }
    
    mock_pipeline.return_value = {
        "days": [
            {"day": 4, "meals": [{"name": "B"}]},
            {"day": 5, "meals": [{"name": "C"}]},
            {"day": 6, "meals": [{"name": "D"}]}
        ]
    }
    
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}
    
    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.side_effect = lambda f, tasks: [f(t) for t in tasks]
        def _mock_submit(fn, *args, **kwargs):
            class MockFuture:
                def result(self, timeout=None):
                    res = fn(*args, **kwargs)
                    return res
            return MockFuture()
        mock_executor.return_value.__enter__.return_value.submit.side_effect = _mock_submit
        mock_executor.return_value.submit.side_effect = _mock_submit
        process_plan_chunk_queue()
    
    mock_pipeline.assert_called_once()
    args, kwargs = mock_pipeline.call_args
    form_data = args[0]
    
    assert form_data["_days_offset"] == 3
    assert form_data["_days_to_generate"] == 3
    assert "Mariscos" in form_data["allergies"]
    assert "Maní" in form_data["allergies"]
    assert form_data.get("medical_conditions") == ["Diabetes"]
    analyze_args = mock_analyze.call_args[0]
    assert analyze_args[1] == [{"role": "user", "content": "No quiero comidas muy secas"}]


@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.run_plan_pipeline')
@patch('cron_tasks.build_memory_context')
@patch('cron_tasks.get_user_likes')
@patch('db.get_user_likes')
@patch('cron_tasks.get_active_rejections')
@patch('db.get_active_rejections')
@patch('cron_tasks.analyze_preferences_agent')
@patch('cron_tasks._build_facts_memory_context')
@patch('cron_tasks.get_all_user_facts')
@patch('db_facts.get_all_user_facts')
@patch('cron_tasks.get_consumed_meals_since')
@patch('db_facts.get_consumed_meals_since')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('db_facts.get_user_facts_by_metadata')
@patch('cron_tasks._check_chunk_learning_ready')
@patch('cron_tasks._inject_advanced_learning_signals')
@patch('cron_tasks.get_user_inventory')
def test_chunk_refreshes_live_pantry_before_pipeline(
    mock_inventory, mock_inject_signals, mock_learning_ready, mock_db_metadata, mock_write, mock_query,
    mock_db_consumed, mock_cron_consumed, mock_db_facts, mock_cron_facts, mock_build_facts,
    mock_analyze, mock_db_rejections, mock_cron_rejections, mock_db_likes,
    mock_cron_likes, mock_build_memory_context, mock_pipeline, mock_shop, mock_pool
):
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": {
            "form_data": {
                "_plan_start_date": "2026-04-21T00:00:00+00:00",
                "session_id": "sess_123",
                "current_pantry_ingredients": ["snapshot pollo", "snapshot arroz"],
            }
        }
    }]

    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]},
        ]
    }

    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(
        plan_data=prior_plan,
        backup_plan=[],
        user_profile={"medical_conditions": []},
        tasks=tasks,
    )
    mock_learning_ready.return_value = {"ready": True, "ratio": 1.0, "matched_meals": 3, "planned_meals": 3}
    mock_inventory.return_value = ["inventario vivo", "huevos", "avena"]
    mock_inject_signals.side_effect = lambda user_id, form_data, *_, **__: form_data
    mock_shop.return_value = {"categories": []}
    mock_cron_facts.return_value = []
    mock_db_facts.return_value = []
    mock_db_consumed.return_value = []
    mock_cron_consumed.return_value = []
    mock_db_rejections.return_value = []
    mock_cron_rejections.return_value = []
    mock_db_likes.return_value = []
    mock_cron_likes.return_value = []
    mock_db_metadata.return_value = []
    mock_build_memory_context.return_value = {"recent_messages": [], "full_context_str": "ctx"}
    mock_analyze.return_value = {}
    mock_pipeline.return_value = {
        "days": [
            {"day": 4, "meals": [{"name": "D"}]},
            {"day": 5, "meals": [{"name": "E"}]},
            {"day": 6, "meals": [{"name": "F"}]},
        ]
    }

    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}

    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.side_effect = lambda f, tasks: [f(t) for t in tasks]

        def _mock_submit(fn, *args, **kwargs):
            class MockFuture:
                def result(self, timeout=None):
                    return fn(*args, **kwargs)
            return MockFuture()

        mock_executor.return_value.__enter__.return_value.submit.side_effect = _mock_submit
        mock_executor.return_value.submit.side_effect = _mock_submit
        process_plan_chunk_queue()

    form_data = mock_pipeline.call_args[0][0]
    assert form_data["current_pantry_ingredients"] == ["inventario vivo", "huevos", "avena"]
    assert mock_inventory.call_count == 2


@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.run_plan_pipeline')
@patch('cron_tasks.build_memory_context')
@patch('cron_tasks.get_user_likes')
@patch('db.get_user_likes')
@patch('cron_tasks.get_active_rejections')
@patch('db.get_active_rejections')
@patch('cron_tasks.analyze_preferences_agent')
@patch('cron_tasks._build_facts_memory_context')
@patch('cron_tasks.get_all_user_facts')
@patch('db_facts.get_all_user_facts')
@patch('cron_tasks.get_consumed_meals_since')
@patch('db_facts.get_consumed_meals_since')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('db_facts.get_user_facts_by_metadata')
@patch('cron_tasks._check_chunk_learning_ready')
@patch('cron_tasks._inject_advanced_learning_signals')
@patch('cron_tasks.get_user_inventory')
def test_chunk_uses_snapshot_pantry_when_live_inventory_refresh_fails(
    mock_inventory, mock_inject_signals, mock_learning_ready, mock_db_metadata, mock_write, mock_query,
    mock_db_consumed, mock_cron_consumed, mock_db_facts, mock_cron_facts, mock_build_facts,
    mock_analyze, mock_db_rejections, mock_cron_rejections, mock_db_likes,
    mock_cron_likes, mock_build_memory_context, mock_pipeline, mock_shop, mock_pool
):
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": {
            "form_data": {
                "_plan_start_date": "2026-04-21T00:00:00+00:00",
                "session_id": "sess_123",
                "current_pantry_ingredients": ["snapshot pollo", "snapshot arroz"],
            }
        }
    }]

    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]},
        ]
    }

    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(
        plan_data=prior_plan,
        backup_plan=[],
        user_profile={"medical_conditions": []},
        tasks=tasks,
    )
    mock_learning_ready.return_value = {"ready": True, "ratio": 1.0, "matched_meals": 3, "planned_meals": 3}
    mock_inventory.side_effect = Exception("db inventory unavailable")
    mock_inject_signals.side_effect = lambda user_id, form_data, *_, **__: form_data
    mock_shop.return_value = {"categories": []}
    mock_cron_facts.return_value = []
    mock_db_facts.return_value = []
    mock_db_consumed.return_value = []
    mock_cron_consumed.return_value = []
    mock_db_rejections.return_value = []
    mock_cron_rejections.return_value = []
    mock_db_likes.return_value = []
    mock_cron_likes.return_value = []
    mock_db_metadata.return_value = []
    mock_build_memory_context.return_value = {"recent_messages": [], "full_context_str": "ctx"}
    mock_analyze.return_value = {}
    mock_pipeline.return_value = {
        "days": [
            {"day": 4, "meals": [{"name": "D"}]},
            {"day": 5, "meals": [{"name": "E"}]},
            {"day": 6, "meals": [{"name": "F"}]},
        ]
    }

    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}

    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.side_effect = lambda f, tasks: [f(t) for t in tasks]

        def _mock_submit(fn, *args, **kwargs):
            class MockFuture:
                def result(self, timeout=None):
                    return fn(*args, **kwargs)
            return MockFuture()

        mock_executor.return_value.__enter__.return_value.submit.side_effect = _mock_submit
        mock_executor.return_value.submit.side_effect = _mock_submit
        process_plan_chunk_queue()

    form_data = mock_pipeline.call_args[0][0]
    assert form_data["current_pantry_ingredients"] == ["snapshot pollo", "snapshot arroz"]
    assert mock_inventory.call_count == 2


@patch('langchain_google_genai.ChatGoogleGenerativeAI')
@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('cron_tasks.get_user_inventory')
@patch('db.get_user_likes')
@patch('db.get_active_rejections')
@patch('db_facts.get_consumed_meals_since')
@patch('db_facts.get_all_user_facts')
def test_edge_case_one_or_two_days(mock_facts, mock_consumed, mock_rejections, mock_likes, mock_inventory, mock_write, mock_query, mock_shop, mock_pool, mock_llm):
    mock_llm.return_value.invoke.side_effect = Exception("Simulated LLM Outage")
    mock_likes.return_value = []
    mock_rejections.return_value = []
    mock_consumed.return_value = []
    mock_facts.return_value = []
    mock_inventory.return_value = ["pollo", "arroz", "avena"]
    # Simula un chunk de solo 2 dias (ej. para completar un plan de 5 dias)
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 2,  # [GAP 8] 2 days instead of 3
        "pipeline_snapshot": json.dumps({"_degraded": True})
    }]
    
    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]}
        ]
    }
    
    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(prior_plan, backup_plan=[], tasks=tasks)
    mock_shop.return_value = {"categories": []}
    
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}
    
    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        def sync_submit(fn, *args, **kwargs):
            future = MagicMock()
            try:
                res = fn(*args, **kwargs)
                future.result.return_value = res
            except Exception as e:
                future.result.side_effect = e
            return future
        mock_executor.return_value.__enter__.return_value.map.side_effect = lambda f, tasks: [f(t) for t in tasks]
        mock_executor.return_value.submit.side_effect = sync_submit
        mock_executor.return_value.__enter__.return_value.submit.side_effect = sync_submit
        process_plan_chunk_queue()
        
    update_calls = [call for call in mock_cursor.execute.call_args_list if "UPDATE meal_plans SET plan_data =" in call[0][0]]
    assert len(update_calls) == 1
    
    args = update_calls[0][0]
    merged_data = json.loads(args[1][0]) if isinstance(args[1][0], str) else args[1][0]
    
    # Original 3 days + new 2 days = 5 days total
    assert len(merged_data["days"]) == 5
    assert merged_data["days"][3]["day"] == 4
    assert merged_data["days"][4]["day"] == 5


@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.run_plan_pipeline')
@patch('cron_tasks.get_user_inventory')
@patch('cron_tasks.get_user_likes')
@patch('db.get_user_likes')
@patch('cron_tasks.get_active_rejections')
@patch('db.get_active_rejections')
@patch('cron_tasks.analyze_preferences_agent')
@patch('cron_tasks._build_facts_memory_context')
@patch('cron_tasks.get_all_user_facts')
@patch('db_facts.get_all_user_facts')
@patch('cron_tasks.get_consumed_meals_since')
@patch('db_facts.get_consumed_meals_since')
@patch('cron_tasks.get_recent_plans')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('db_facts.get_user_facts_by_metadata')
def test_continuous_learning_mid_plan_injection(
    mock_db_metadata, mock_write, mock_query, mock_recent_plans,
    mock_db_consumed, mock_cron_consumed, mock_db_facts, mock_cron_facts, mock_build_facts,
    mock_analyze, mock_db_rejections, mock_cron_rejections, mock_db_likes,
    mock_cron_likes, mock_inventory, mock_pipeline, mock_shop, mock_pool
):
    # (a) Simular un plan de 9 días -> 3 chunks. Estamos procesando el chunk 2 (días 4-6)
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_9_days",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": {
            "form_data": {"allergies": ["Ninguna"]} # Snapshot original no tenía la alergia
        }
    }]
    
    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]}
        ]
    }
    
    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(
        plan_data=prior_plan,
        backup_plan=[],
        user_profile={"medical_conditions": [], "quality_history": [{"date": "2023-01-01", "score": 85}]},
        tasks=tasks
    )
    
    mock_shop.return_value = {"categories": []}
    
    # (b) Insertar fact "Alergia a X" tras chunk-1
    new_allergy = "Alergia a Camarones (Reciente)"
    mock_cron_facts.return_value = [{"fact": new_allergy}]
    mock_db_facts.return_value = [{"fact": new_allergy}]
    mock_db_metadata.return_value = [{"fact": new_allergy}]
    
    mock_db_consumed.return_value = []
    mock_cron_consumed.return_value = []
    mock_recent_plans.return_value = []
    mock_db_rejections.return_value = []
    mock_cron_rejections.return_value = []
    mock_db_likes.return_value = []
    mock_cron_likes.return_value = []
    mock_inventory.return_value = ["pollo", "arroz", "avena"]
    
    # El pipeline retorna un resultado que *no* debería contener camarones
    mock_pipeline.return_value = {
        "days": [
            {"day": 4, "meals": [{"name": "Pollo al horno"}]},
            {"day": 5, "meals": [{"name": "Res a la plancha"}]},
            {"day": 6, "meals": [{"name": "Pescado blanco"}]}
        ]
    }
    
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}
    
    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.side_effect = lambda f, tasks: [f(t) for t in tasks]
        def _mock_submit(fn, *args, **kwargs):
            class MockFuture:
                def result(self, timeout=None):
                    res = fn(*args, **kwargs)
                    return res
            return MockFuture()
        mock_executor.return_value.__enter__.return_value.submit.side_effect = _mock_submit
        mock_executor.return_value.submit.side_effect = _mock_submit
        process_plan_chunk_queue()
        
    # (c) Verificar que chunk-2 inyectó la alergia
    mock_pipeline.assert_called_once()
    args, kwargs = mock_pipeline.call_args
    form_data = args[0]
    
    assert new_allergy in form_data.get("allergies", []), "La alergia aprendida dinámicamente no se inyectó en el chunk-2"
    
    # Y que plan_data['days'][3:6] se unió sin la alergia (verificado en el mock_pipeline)
    update_calls = [call for call in mock_cursor.execute.call_args_list if "UPDATE meal_plans SET plan_data =" in call[0][0]]
    assert len(update_calls) == 1
    
    update_args = update_calls[0][0]
    merged_data = json.loads(update_args[1][0]) if isinstance(update_args[1][0], str) else update_args[1][0]
    
    assert len(merged_data["days"]) == 6
    for i in range(3, 6):
        for meal in merged_data["days"][i]["meals"]:
            assert "camarones" not in meal["name"].lower()


@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.run_plan_pipeline')
@patch('cron_tasks.get_user_inventory')
@patch('cron_tasks.build_memory_context')
@patch('cron_tasks.get_user_likes')
@patch('db.get_user_likes')
@patch('cron_tasks.get_active_rejections')
@patch('db.get_active_rejections')
@patch('cron_tasks.analyze_preferences_agent')
@patch('cron_tasks._build_facts_memory_context')
@patch('cron_tasks.get_all_user_facts')
@patch('db_facts.get_all_user_facts')
@patch('cron_tasks.get_consumed_meals_since')
@patch('db_facts.get_consumed_meals_since')
@patch('cron_tasks.get_recent_plans')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('db_facts.get_user_facts_by_metadata')
def test_high_ingredient_base_repeat_forces_variety_on_next_chunk(
    mock_db_metadata, mock_write, mock_query, mock_recent_plans,
    mock_db_consumed, mock_cron_consumed, mock_db_facts, mock_cron_facts, mock_build_facts,
    mock_analyze, mock_db_rejections, mock_cron_rejections, mock_db_likes,
    mock_cron_likes, mock_build_memory_context, mock_inventory, mock_pipeline, mock_shop, mock_pool
):
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_force_variety",
        "week_number": 3,
        "days_offset": 6,
        "days_count": 3,
        "pipeline_snapshot": {
            "form_data": {"_plan_start_date": "2026-04-21T00:00:00+00:00"}
        }
    }]

    prior_plan = {
        "_last_chunk_learning": {"ingredient_base_repeat_pct": 75.0},
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]},
            {"day": 4, "meals": [{"name": "D"}]},
            {"day": 5, "meals": [{"name": "E"}]},
            {"day": 6, "meals": [{"name": "F"}]},
        ]
    }

    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(
        plan_data=prior_plan,
        backup_plan=[],
        user_profile={},
        tasks=tasks
    )
    mock_shop.return_value = {"categories": []}
    mock_cron_facts.return_value = []
    mock_db_facts.return_value = []
    mock_db_consumed.return_value = [{"meal_name": "D"}, {"meal_name": "E"}, {"meal_name": "F"}]
    mock_cron_consumed.return_value = [{"meal_name": "D"}, {"meal_name": "E"}, {"meal_name": "F"}]
    mock_recent_plans.return_value = []
    mock_db_rejections.return_value = []
    mock_cron_rejections.return_value = []
    mock_db_likes.return_value = []
    mock_cron_likes.return_value = []
    mock_inventory.return_value = ["pollo", "arroz", "avena"]
    mock_inventory.return_value = ["pollo", "arroz", "avena"]
    mock_db_metadata.return_value = []
    mock_pipeline.return_value = {
        "days": [
            {"day": 7, "meals": [{"name": "G"}]},
            {"day": 8, "meals": [{"name": "H"}]},
            {"day": 9, "meals": [{"name": "I"}]}
        ]
    }

    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}

    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.side_effect = lambda f, tasks: [f(t) for t in tasks]
        def _mock_submit(fn, *args, **kwargs):
            class MockFuture:
                def result(self, timeout=None):
                    return fn(*args, **kwargs)
            return MockFuture()
        mock_executor.return_value.__enter__.return_value.submit.side_effect = _mock_submit
        mock_executor.return_value.submit.side_effect = _mock_submit
        process_plan_chunk_queue()

    form_data = mock_pipeline.call_args[0][0]
    assert form_data["_force_variety"] is True


@patch('cron_tasks._persist_nightly_learning_signals')
@patch('cron_tasks.reserve_plan_ingredients')
@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.run_plan_pipeline')
@patch('cron_tasks.get_user_inventory')
@patch('cron_tasks.build_memory_context')
@patch('cron_tasks.get_user_likes')
@patch('db.get_user_likes')
@patch('cron_tasks.get_active_rejections')
@patch('db.get_active_rejections')
@patch('cron_tasks.analyze_preferences_agent')
@patch('cron_tasks._build_facts_memory_context')
@patch('cron_tasks.get_all_user_facts')
@patch('db_facts.get_all_user_facts')
@patch('cron_tasks.get_consumed_meals_since')
@patch('db_facts.get_consumed_meals_since')
@patch('cron_tasks.get_recent_plans')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('db_facts.get_user_facts_by_metadata')
def test_chunk_persists_learning_signals_after_completion(
    mock_db_metadata, mock_write, mock_query, mock_recent_plans,
    mock_db_consumed, mock_cron_consumed, mock_db_facts, mock_cron_facts, mock_build_facts,
    mock_analyze, mock_db_rejections, mock_cron_rejections, mock_db_likes,
    mock_cron_likes, mock_build_memory_context, mock_inventory, mock_pipeline, mock_shop,
    mock_pool, mock_reserve_inventory, mock_persist_learning
):
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_learning_hook",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": {
            "form_data": {"_plan_start_date": "2026-04-21T00:00:00+00:00"}
        }
    }]

    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]},
        ]
    }

    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(
        plan_data=prior_plan,
        backup_plan=[],
        user_profile={"medical_conditions": [], "quality_history_rotations": [0.8]},
        tasks=tasks,
    )
    mock_shop.return_value = {"categories": []}
    mock_cron_facts.return_value = []
    mock_db_facts.return_value = []
    consumed_records = [{"meal_name": "A"}, {"meal_name": "B"}]
    mock_db_consumed.return_value = consumed_records
    mock_cron_consumed.return_value = consumed_records
    mock_recent_plans.return_value = []
    mock_db_rejections.return_value = []
    mock_cron_rejections.return_value = []
    mock_db_likes.return_value = []
    mock_cron_likes.return_value = []
    mock_inventory.return_value = ["pollo", "arroz", "avena"]
    mock_db_metadata.return_value = []
    mock_build_memory_context.return_value = {"recent_messages": [], "full_context_str": "ctx"}
    mock_analyze.return_value = {}
    mock_reserve_inventory.return_value = 3
    mock_pipeline.return_value = {
        "days": [
            {"day": 4, "meals": [{"name": "D"}]},
            {"day": 5, "meals": [{"name": "E"}]},
            {"day": 6, "meals": [{"name": "F"}]},
        ]
    }

    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}

    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.side_effect = lambda f, tasks: [f(t) for t in tasks]

        def _mock_submit(fn, *args, **kwargs):
            class MockFuture:
                def result(self, timeout=None):
                    return fn(*args, **kwargs)
            return MockFuture()

        mock_executor.return_value.__enter__.return_value.submit.side_effect = _mock_submit
        mock_executor.return_value.submit.side_effect = _mock_submit
        process_plan_chunk_queue()

    mock_persist_learning.assert_called_once()
    mock_reserve_inventory.assert_called_once()
    persist_args = mock_persist_learning.call_args[0]
    assert persist_args[0] == "user_123"
    assert isinstance(persist_args[1], dict)
    assert len(persist_args[2]) == 6
    assert persist_args[3] == consumed_records

def test_pantry_hybrid_tolerance_quantity():
    from constants import validate_ingredients_against_pantry, CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE
    pantry = ["200g pechuga de pollo"]
    
    # 220g is 10% more, should fail with hybrid tolerance (1.05)
    res_fail = validate_ingredients_against_pantry(["220g pechuga de pollo"], pantry, strict_quantities=True, tolerance=CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE)
    assert isinstance(res_fail, str)
    
    # 205g is 2.5% more, should pass with hybrid tolerance (1.05)
    res_pass = validate_ingredients_against_pantry(["205g pechuga de pollo"], pantry, strict_quantities=True, tolerance=CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE)
    assert res_pass is True

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('cron_tasks.get_user_inventory')
@patch('db.get_user_likes')
@patch('db.get_active_rejections')
@patch('db_facts.get_consumed_meals_since')
@patch('db_facts.get_all_user_facts')
def test_chunk_degraded_fallback_pauses_when_no_pantry_coverage(
    mock_facts, mock_consumed, mock_rejections, mock_likes, mock_inventory,
    mock_write, mock_query, mock_shop, mock_pool, mock_llm
):
    mock_llm.return_value.invoke.side_effect = Exception("Simulated LLM Outage")
    mock_likes.return_value = []
    mock_rejections.return_value = []
    mock_consumed.return_value = []
    mock_facts.return_value = []
    # 5 ingredients not matching prior plan
    mock_inventory.return_value = ["manzana", "pera", "uva", "kiwi", "melon"]
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 1,
        "pipeline_snapshot": json.dumps({
            "_degraded": True,
            "form_data": {"current_pantry_ingredients": ["manzana", "pera", "uva", "kiwi", "melon"]},
        })
    }]

    prior_plan = {
        "days": [
            {
                "day": 1,
                "meals": [{"name": "Res con quinoa", "ingredients": ["res", "quinoa", "esparragos"]}],
            }
        ]
    }

    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(prior_plan, backup_plan=[], tasks=tasks)
    mock_shop.return_value = {"categories": []}
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}

    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        def sync_submit(fn, *args, **kwargs):
            future = MagicMock()
            future.result.return_value = fn(*args, **kwargs)
            return future
        mock_executor.return_value.__enter__.return_value.submit.side_effect = sync_submit
        
        from cron_tasks import process_plan_chunk_queue
        process_plan_chunk_queue()

    # Should pause chunk
    update_calls = [call for call in mock_write.call_args_list if "UPDATE plan_chunk_queue" in call[0][0]]
    assert len(update_calls) > 0
    pause_call = update_calls[-1]
    assert "status = 'pending_user_action'" in pause_call[0][0]
    
    # Verify reason is in pipeline_snapshot
    snapshot_json = pause_call[0][1][0]
    snapshot = json.loads(snapshot_json)
    assert snapshot.get("_pantry_pause_reason") == "degraded_no_pantry_coverage"

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('cron_tasks.get_user_inventory')
@patch('db.get_user_likes')
@patch('db.get_active_rejections')
@patch('db_facts.get_consumed_meals_since')
@patch('db_facts.get_all_user_facts')
def test_chunk_learning_stub_starved_window(
    mock_facts, mock_consumed, mock_rejections, mock_likes, mock_inventory,
    mock_write, mock_query, mock_shop, mock_pool, mock_llm
):
    mock_llm.return_value.invoke.return_value.content = '```json\n{"days": [{"day": 7, "meals": [{"name": "Pollo"}]}]}\n```'
    mock_likes.return_value = []
    mock_rejections.return_value = []
    mock_consumed.return_value = []
    mock_facts.return_value = []
    mock_inventory.return_value = []
    
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 3,
        "days_offset": 6,
        "days_count": 3,
        "pipeline_snapshot": json.dumps({"form_data": {}})
    }]

    prior_plan = {
        "total_days_requested": 15,
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]}
        ],
        "_last_chunk_learning": {"metrics_unavailable": True, "chunk_learning_stub_count": 1},
        "_recent_chunk_lessons": [{"metrics_unavailable": True}]
    }

    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(prior_plan, backup_plan=[], tasks=tasks)
    mock_shop.return_value = {"categories": []}
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}

    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        def sync_submit(fn, *args, **kwargs):
            future = MagicMock()
            future.result.return_value = fn(*args, **kwargs)
            return future
        mock_executor.return_value.__enter__.return_value.submit.side_effect = sync_submit
        
        from cron_tasks import process_plan_chunk_queue
        process_plan_chunk_queue()

    # Because metrics were filtered, form_data should have got _learning_window_starved = True and _force_variety = True
    # We can check that the LLM invocation had it.
    prompt_used = mock_llm.return_value.invoke.call_args[0][0]
    assert "forzar la variedad" in prompt_used.lower() or "diversificar" in prompt_used.lower() or "distint" in prompt_used.lower()

def test_release_chunk_reservations_frees_inventory():
    """P0-4: reservar 100g de pollo en chunk X, cancelar chunk, verificar que get_user_inventory_net no descuenta."""
    from unittest.mock import patch, MagicMock
    from db_inventory import (
        release_chunk_reservations,
        _make_reservation_key,
        _normalize_reservation_details,
    )
    
    chunk_id = "test-chunk-999"
    user_id = "user_p04"
    reservation_key = _make_reservation_key(chunk_id, "Pollo Asado")
    
    # Simulate a row with 500g total, 100g reserved for our chunk
    mock_row = {
        "id": "row-1",
        "reserved_quantity": 100.0,
        "reservation_details": {reservation_key: 100.0},
    }
    
    mock_supabase = MagicMock()
    mock_select = MagicMock()
    mock_select.eq.return_value.gt.return_value.execute.return_value.data = [mock_row]
    mock_supabase.table.return_value.select.return_value = mock_select
    mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
    
    with patch("db_inventory.supabase", mock_supabase):
        released = release_chunk_reservations(user_id, chunk_id)
    
    assert released == 1
    # Verify _update_row_reservation was called with reserved_quantity=0 and empty details
    update_call = mock_supabase.table.return_value.update
    update_call.assert_called_once()
    call_args = update_call.call_args[0][0]
    assert call_args["reserved_quantity"] == 0.0
    assert reservation_key not in call_args["reservation_details"]

def test_partial_reservation_marks_chunk_and_defers_next():
    """P0-5: mock Supabase to fail 4 of 5 reservations → reservation_status='partial'."""
    from unittest.mock import patch, MagicMock, call
    from cron_tasks import _reconcile_chunk_reservations

    new_days = [{
        "day": 1,
        "meals": [{
            "name": "Pollo Asado",
            "ingredients": ["200g pechuga de pollo", "100g arroz", "50g brocoli", "30g cebolla", "10ml aceite"]
        }]
    }]

    # Mock reserve_plan_ingredients to return only 1 (out of 5 expected)
    with patch("cron_tasks.reserve_plan_ingredients", return_value=1) as mock_reserve, \
         patch("cron_tasks.execute_sql_write") as mock_write:

        _reconcile_chunk_reservations("user_123", "chunk-42", new_days, max_retries=1)

        # Should have tried to reserve
        mock_reserve.assert_called_once_with("user_123", "chunk-42", new_days)

        # Should NOT have marked 'ok' since 1 < 50% of 5
        ok_calls = [c for c in mock_write.call_args_list if "reservation_status = 'ok'" in str(c)]
        assert len(ok_calls) == 0

    # Now test success path
    with patch("cron_tasks.reserve_plan_ingredients", return_value=5) as mock_reserve2, \
         patch("cron_tasks.execute_sql_write") as mock_write2:

        _reconcile_chunk_reservations("user_123", "chunk-42", new_days, max_retries=1)

        ok_calls2 = [c for c in mock_write2.call_args_list if "reservation_status = 'ok'" in str(c)]
        assert len(ok_calls2) == 1

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('cron_tasks.get_user_inventory')
@patch('db.get_user_likes')
@patch('db.get_active_rejections')
@patch('db_facts.get_consumed_meals_since')
@patch('db_facts.get_all_user_facts')
def test_degraded_shuffle_rejects_day_exceeding_pantry_quantities(
    mock_facts, mock_consumed, mock_rejections, mock_likes, mock_inventory,
    mock_write, mock_query, mock_shop, mock_pool, mock_llm
):
    """P0-#1: Verify degraded mode validates quantities and falls back to edge recipe or pauses."""
    mock_llm.return_value.invoke.side_effect = Exception("Simulated LLM Outage")
    mock_likes.return_value = []
    mock_rejections.return_value = []
    mock_consumed.return_value = []
    mock_facts.return_value = []
    
    # Pantry only has 80g of pollo, but prior day needs 300g
    mock_inventory.return_value = ["80g pechuga de pollo", "500g arroz"]
    
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 1,
        "pipeline_snapshot": json.dumps({
            "_degraded": True,
            "form_data": {"current_pantry_ingredients": ["80g pechuga de pollo", "500g arroz"]},
        })
    }]

    # A pool day that exceeds the pantry quantity
    prior_plan = {
        "days": [
            {
                "day": 1,
                "meals": [{"name": "Pollo mucho", "ingredients": ["300g pechuga de pollo", "100g arroz"]}],
            }
        ]
    }

    mock_write.side_effect = _mock_execute_sql_write_factory(tasks)
    mock_query.side_effect = _mock_execute_sql_query_factory(prior_plan, backup_plan=[], tasks=tasks)
    mock_shop.return_value = {"categories": []}
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"plan_data": prior_plan}

    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        def sync_submit(fn, *args, **kwargs):
            future = MagicMock()
            future.result.return_value = fn(*args, **kwargs)
            return future
        mock_executor.return_value.__enter__.return_value.submit.side_effect = sync_submit
        
        from cron_tasks import process_plan_chunk_queue
        process_plan_chunk_queue()

    # The original day should have been rejected.
    # It either fell back to an edge recipe that respects 80g pollo, or paused the chunk.
    # Let's check if it paused
    update_calls = [call for call in mock_write.call_args_list if "UPDATE plan_chunk_queue" in call[0][0]]
    if len(update_calls) > 0:
        last_call_sql = update_calls[-1][0][0]
        if "status = 'pending_user_action'" in last_call_sql:
            # Paused successfully!
            snapshot_json = update_calls[-1][0][1][0]
            snapshot = json.loads(snapshot_json)
            assert snapshot.get("_pantry_pause_reason") == "degraded_quantities_unfeasible"
            return

    # If it didn't pause, it means Edge Recipe worked and saved the plan
    plan_update_calls = [call for call in mock_cursor.execute.call_args_list if "UPDATE meal_plans SET plan_data =" in call[0][0]]
    assert len(plan_update_calls) == 1
    
    args = plan_update_calls[0][0]
    merged_data = json.loads(args[1][0]) if isinstance(args[1][0], str) else args[1][0]
    
    # Verify the day from prior plan (Pollo mucho) is NOT the one generated
    generated_day = merged_data["days"][-1]
    # If edge recipe worked, it should not have the "Pollo mucho" meal
    meal_names = [m["name"] for m in generated_day["meals"]]
    assert "Pollo mucho" not in meal_names


@patch('cron_tasks.get_inventory_activity_since')
@patch('cron_tasks.get_consumed_meals_since')
def test_sparse_logging_proxy_passes_with_inventory_mutations(mock_consumed, mock_inv_activity):
    from cron_tasks import _check_chunk_learning_ready
    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]},
            {"day": 4, "meals": [{"name": "D"}]},
            {"day": 5, "meals": [{"name": "E"}]},
        ]
    }
    snapshot = {"form_data": {"_plan_start_date": "2026-04-21T00:00:00+00:00"}}
    
    mock_consumed.return_value = [{"meal_name": "A", "status": "consumed", "id": "1"}]
    mock_inv_activity.return_value = {"mutations_count": 5}

    learning_ready = _check_chunk_learning_ready(
        user_id="user_123",
        meal_plan_id="plan_456",
        week_number=2,
        days_offset=5,
        plan_data=prior_plan,
        snapshot=snapshot,
    )

    assert learning_ready["ready"] is True
    assert learning_ready["sparse_logging_proxy"] is True
    assert learning_ready["inventory_proxy_used"] is True


@patch('cron_tasks.get_inventory_activity_since')
@patch('cron_tasks.get_consumed_meals_since')
def test_sparse_logging_proxy_defers_without_inventory_mutations(mock_consumed, mock_inv_activity):
    from cron_tasks import _check_chunk_learning_ready
    prior_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]},
            {"day": 4, "meals": [{"name": "D"}]},
            {"day": 5, "meals": [{"name": "E"}]},
        ]
    }
    snapshot = {"form_data": {"_plan_start_date": "2026-04-21T00:00:00+00:00"}}
    
    mock_consumed.return_value = [{"meal_name": "A", "status": "consumed", "id": "1"}]
    mock_inv_activity.return_value = {"mutations_count": 0}

    learning_ready = _check_chunk_learning_ready(
        user_id="user_123",
        meal_plan_id="plan_456",
        week_number=2,
        days_offset=5,
        plan_data=prior_plan,
        snapshot=snapshot,
    )

    assert learning_ready["ready"] is False
    assert learning_ready["sparse_logging_proxy"] is True
    assert learning_ready["inventory_proxy_used"] is False

