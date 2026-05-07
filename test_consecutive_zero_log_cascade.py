import sys
from unittest.mock import MagicMock, patch

# Mocks setup
sys.modules['apscheduler'] = MagicMock()
sys.modules['apscheduler.triggers'] = MagicMock()
sys.modules['apscheduler.triggers.cron'] = MagicMock()
sys.modules['apscheduler.schedulers'] = MagicMock()
sys.modules['apscheduler.schedulers.background'] = MagicMock()
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.messages'] = MagicMock()
sys.modules['langchain_core.tools'] = MagicMock()
sys.modules['langchain_core.prompts'] = MagicMock()

import pytest
import json
from cron_tasks import process_plan_chunk, CHUNK_LEARNING_READY_MAX_DEFERRALS

@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks._check_chunk_learning_ready")
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.run_plan_pipeline")
def test_consecutive_zero_log_cascade_and_push(
    mock_pipeline, mock_push, mock_ready, mock_write, mock_query
):
    """
    Test P0-4 fix: Verify that consecutive zero-log chunks are tracked,
    future chunks are delayed by 24h, and reaching 3 consecutive hits
    degrades the plan and sends a specific push notification.
    """
    task_id = "task-1"
    user_id = "user-1"
    meal_plan_id = "plan-15d"
    week_number = 3
    
    # 1. Setup mock to simulate a zero-log failure after max deferrals
    # So we trigger the exhaustion block for _is_zero_log
    mock_ready.return_value = {
        "ready": False,
        "reason": "learning_zero_logs",
        "zero_log_proxy": True,
        "ratio": 0.0
    }
    
    # Mock queries
    # First query is plan_chunk_queue
    # Second query is meal_plans (plan_data prior)
    def mock_query_side_effect(query, params=None, fetch_one=False, fetch_all=False):
        if "FROM plan_chunk_queue" in query:
            return {
                "pipeline_snapshot": {
                    # Set deferrals to max so it goes to exhaustion block
                    "_learning_ready_deferrals": CHUNK_LEARNING_READY_MAX_DEFERRALS,
                    "_pantry_flexible_mode": False,
                    "_learning_flexible_mode": False
                }
            }
        elif "FROM meal_plans" in query:
            return {
                "plan_data": {
                    # Simulate we already had 2 consecutive zero log chunks!
                    "_consecutive_zero_log_chunks": 2
                }
            }
        return {}

    mock_query.side_effect = mock_query_side_effect

    # 2. Execute process_plan_chunk
    process_plan_chunk(task_id, user_id, meal_plan_id, week_number, {})

    # 3. Assertions
    # A) Push notification should have been sent with specific 3-strike copy
    mock_push.assert_called_once()
    push_args, push_kwargs = mock_push.call_args
    assert push_kwargs["title"] == "Tu plan se está generando sin tu feedback"
    assert "varios bloques sin registrar comidas" in push_kwargs["body"]

    # B) Plan data should be updated with consecutive_zero_log_chunks = 3 and generation_status degraded
    write_calls = mock_write.call_args_list
    plan_update_called = False
    cascade_delay_called = False
    
    for call in write_calls:
        q = call[0][0]
        p = call[0][1]
        
        if "UPDATE meal_plans SET plan_data" in q:
            plan_update_called = True
            # The params for this query should be: json_data, generation_status, meal_plan_id
            # Wait, in the code we did COALESCE or direct assignment depending on condition
            # Actually we did:
            # "UPDATE meal_plans SET plan_data = %s::jsonb, generation_status = %s WHERE id = %s"
            if "generation_status = %s" in q:
                plan_data_arg = json.loads(p[0])
                assert plan_data_arg["_consecutive_zero_log_chunks"] == 3
                assert p[1] == "degraded_pending_engagement"
                assert p[2] == meal_plan_id

        if "UPDATE plan_chunk_queue" in q and "execute_after = NOW() + interval '24 hours'" in q:
            cascade_delay_called = True
            assert p[0] == meal_plan_id
            assert p[1] == week_number
            
    assert plan_update_called, "meal_plans update for 3 strikes missing"
    assert cascade_delay_called, "future chunks were not delayed by 24h"
