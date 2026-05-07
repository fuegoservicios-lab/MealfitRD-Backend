import sys
from unittest.mock import MagicMock, patch
import time

# Mock APScheduler early since it's commonly imported
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

import pytest
from datetime import datetime, timezone, timedelta
import json
from cron_tasks import _background_shift_plan_for_user

@patch("db_core.connection_pool")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks._enqueue_plan_chunk")
@patch("utils_push.send_push_notification")
def test_renewal_pantry_empty_aborts_generation(
    mock_push, mock_enqueue, mock_inventory, mock_pool
):
    """
    Test P0-2 fix: Ensures that if background auto-renewal triggers for an expired 7/15/30d plan,
    but the pantry is empty (or unreachable), the system aborts the generation, sets 
    expired_pending_pantry, and sends a push notification instead of blindly enqueuing chunks.
    """
    # 1. Setup mock DB cursor
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.transaction.return_value.__enter__.return_value = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    user_id = "test_user_empty_pantry"
    plan_id = "plan-777"

    today = datetime.now(timezone.utc)
    # Simulate a plan created 7 days ago (now fully expired)
    start_date = (today - timedelta(days=7)).isoformat()
    
    plan_data = {
        "days": [{"day": i, "meals": [{"name": f"Meal {i}"}]} for i in range(1, 8)],
        "grocery_start_date": start_date,
        "total_days_requested": 7,
        "generation_status": "complete"
    }

    mock_cursor.fetchone.side_effect = [
        # 1. SELECT FOR UPDATE returns the expired plan
        {"id": plan_id, "plan_data": plan_data},
        # 2. SELECT health_profile
        {"health_profile": {"diet": "mediterranean"}},
    ]

    # 2. Simulate empty pantry (less than CHUNK_MIN_FRESH_PANTRY_ITEMS)
    mock_inventory.return_value = ["Sal"] # Only 1 item

    # 3. Run background shift task
    res = _background_shift_plan_for_user(user_id)

    # 4. Validations
    assert res is False, "The task should abort and return False when pantry is empty"

    # Chunks should NEVER be enqueued if pantry is empty
    assert mock_enqueue.call_count == 0, f"Expected 0 chunks, but got {mock_enqueue.call_count}"

    # DB UPDATE must set generation_status = "expired_pending_pantry"
    update_called = False
    for call in mock_cursor.execute.call_args_list:
        query = call[0][0]
        if "UPDATE meal_plans SET plan_data" in query:
            update_called = True
            params = call[0][1]
            updated_data = json.loads(params[0])
            assert updated_data.get("generation_status") == "expired_pending_pantry"
            assert "pending_user_action" in updated_data
            assert updated_data["pending_user_action"]["type"] == "pantry_required"
            assert updated_data["pending_user_action"]["message"] == "Actualiza tu nevera para renovar tu plan"
            break
            
    assert update_called is True, "Missing DB UPDATE for meal_plans generation_status"

    # Wait a bit for the daemon thread to start and call the mock
    time.sleep(0.1)
    
    # Push notification should be triggered
    assert mock_push.call_count == 1, "Expected push notification to be dispatched for empty pantry"
    args, kwargs = mock_push.call_args
    assert kwargs.get("user_id") == user_id
    assert kwargs.get("title") == "Renovación pausada"
    assert kwargs.get("url") == "/dashboard"
