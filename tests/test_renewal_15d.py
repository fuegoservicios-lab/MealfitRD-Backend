import sys
from unittest.mock import MagicMock, patch
sys.modules['apscheduler'] = MagicMock()
sys.modules['apscheduler.triggers'] = MagicMock()
sys.modules['apscheduler.triggers.cron'] = MagicMock()
sys.modules['apscheduler.schedulers'] = MagicMock()
sys.modules['apscheduler.schedulers.background'] = MagicMock()

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
import json
from routers.plans import api_shift_plan

@patch("db_core.connection_pool")
@patch("cron_tasks._enqueue_plan_chunk")
def test_15d_renewal(mock_enqueue, mock_pool):
    # Setup mock DB cursor
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.transaction.return_value.__enter__.return_value = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    today = datetime.now(timezone.utc)
    # Simulate a plan created 15 days ago
    start_date = (today - timedelta(days=15)).isoformat()
    
    plan_data = {
        "days": [{"day": i, "meals": [{"name": f"Meal {i}"}]} for i in range(1, 16)],
        "grocery_start_date": start_date,
        "total_days_requested": 15,
        "generation_status": "complete",
        "_lifetime_lessons_history": [{"rejection_violations": 1}],
        "_lifetime_lessons_summary": {"total_rejection_violations": 1}
    }
    
    mock_cursor.fetchone.side_effect = [
        # 1. SELECT FOR UPDATE returns the plan
        {"id": "plan-123", "plan_data": plan_data},
        # 2. SELECT COUNT(*) chunks in flight -> 0
        {"cnt": 0},
        # 3. SELECT health_profile
        {"health_profile": {"diet": "keto"}},
        # 4. SELECT COALESCE(MAX(week_number), 0)
        {"max_week": 5}
    ]
    
    from fastapi import Response
    data = {"user_id": "test_user"}

    res = api_shift_plan(Response(), data, verified_user_id="test_user")
    
    assert res.get("success") is True, f"Failed: {res.get('message')}"
    
    assert mock_enqueue.call_count == 5, f"Expected 5 chunks, got {mock_enqueue.call_count}"
    
    offsets = []
    chunk_sizes = []
    inherited = False
    
    for i, call in enumerate(mock_enqueue.call_args_list):
        args, kwargs = call
        user_id, plan_id, next_week, current_offset, chunk_count, snapshot = args
        offsets.append(current_offset)
        chunk_sizes.append(chunk_count)
        
        if i == 0:
            assert "_inherited_lifetime_lessons" in snapshot
            assert snapshot["_inherited_lifetime_lessons"]["history"] == [{"rejection_violations": 1}]
            inherited = True
        else:
            assert "_inherited_lifetime_lessons" not in snapshot
            
    assert offsets == [0, 3, 6, 9, 12], f"Expected offsets [0, 3, 6, 9, 12], got {offsets}"
    assert chunk_sizes == [3, 3, 3, 3, 3], f"Expected sizes [3, 3, 3, 3, 3], got {chunk_sizes}"
    assert inherited is True


@patch("db_core.connection_pool")
@patch("cron_tasks._enqueue_plan_chunk")
@patch("routers.plans.save_partial_plan_get_id")
@patch("routers.plans.run_plan_pipeline")
@patch("routers.plans.analyze_preferences_agent")
@patch("routers.plans.build_memory_context")
@patch("routers.plans.get_or_create_session")
@patch("routers.plans.get_user_likes")
@patch("routers.plans.get_active_rejections")
@patch("routers.plans._user_has_profile")
@patch("db_core.execute_sql_query")
def test_new_plan_inherits_lifetime_lessons_from_previous_plan(
    mock_execute_sql, mock_has_profile, mock_get_rejections, mock_get_likes,
    mock_get_session, mock_build_memory, mock_analyze_pref, mock_run_pipeline,
    mock_save_partial, mock_enqueue, mock_pool
):
    from routers.plans import api_analyze
    from fastapi import BackgroundTasks

    # Mock user and profile to allow chunking
    mock_has_profile.return_value = True
    mock_get_likes.return_value = []
    mock_get_rejections.return_value = []
    mock_build_memory.return_value = {"recent_messages": [], "full_context_str": ""}
    mock_analyze_pref.return_value = "Test taste profile"
    
    # Mock the pipeline result to simulate a 3-day initial chunk
    mock_run_pipeline.return_value = {
        "days": [{"day": i, "meals": [{"name": f"Meal {i}"}]} for i in range(1, 4)],
        "_selected_techniques": ["technique1"]
    }
    
    # Mock saving the partial plan to get a plan ID
    mock_save_partial.return_value = "new-plan-id"
    
    # Mock the prior plan DB fetch (execute_sql_query)
    prior_plan_data = {
        "plan_data": {
            "_lifetime_lessons_history": [{"rejection_violations": 2}],
            "_lifetime_lessons_summary": {"total_rejection_violations": 2}
        }
    }
    mock_execute_sql.return_value = prior_plan_data
    
    # Prepare API request data
    data = {
        "user_id": "test_user",
        "session_id": "test_session",
        "totalDays": 15,
        "tzOffset": 240,
        "mainGoal": "Perder peso"
    }
    
    bg_tasks = BackgroundTasks()
    res = api_analyze(bg_tasks, data, verified_user_id="test_user")
    
    # Verify the response
    assert res.get("generation_status") == "partial"
    assert res.get("id") == "new-plan-id"
    
    # Verify the DB call for the prior plan
    called_queries = [call[0][0] for call in mock_execute_sql.call_args_list]
    assert any("SELECT plan_data FROM meal_plans WHERE user_id" in q for q in called_queries)
    
    # Verify that enqueue was called for chunks 2 to 5
    assert mock_enqueue.call_count == 4, f"Expected 4 chunks enqueued, got {mock_enqueue.call_count}"
    
    inherited = False
    for i, call in enumerate(mock_enqueue.call_args_list):
        args, kwargs = call
        user_id, plan_id, wk, offset, count, snapshot = args
        
        if wk == 2:
            assert "_inherited_lifetime_lessons" in snapshot
            assert snapshot["_inherited_lifetime_lessons"]["history"] == [{"rejection_violations": 2}]
            inherited = True
        else:
            assert "_inherited_lifetime_lessons" not in snapshot
            
    assert inherited is True
