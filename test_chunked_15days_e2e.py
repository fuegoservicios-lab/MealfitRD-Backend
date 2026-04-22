import pytest
import uuid
import json
import time
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

from db_core import execute_sql_write, execute_sql_query, connection_pool
from cron_tasks import process_plan_chunk_queue, _enqueue_plan_chunk
from graph_orchestrator import run_plan_pipeline

import logging
logging.getLogger().setLevel(logging.INFO)

if connection_pool and not getattr(connection_pool, '_opened', False):
    connection_pool.open()

def _mock_run_plan_pipeline(form_data, *args, **kwargs):
    offset = form_data.get("_days_offset", 0)
    count = form_data.get("_days_to_generate", 3)
    
    days = []
    for i in range(count):
        day_num = offset + i + 1
        days.append({
            "day": day_num,
            "daily_summary": f"Day {day_num} summary",
            "meals": [
                {"name": f"Breakfast {day_num}", "type": "Desayuno"},
                {"name": f"Lunch {day_num}", "type": "Almuerzo"}
            ]
        })
        
    return {"days": days, "generation_status": "partial"}

@pytest.fixture
def cleanup_user_db():
    # Use an existing user_id to avoid foreign key errors on auth.users
    user_row = execute_sql_query("SELECT id FROM user_profiles LIMIT 1", fetch_one=True)
    if not user_row:
        pytest.skip("No user_profiles found in DB to use for E2E tests.")
    
    user_id = user_row["id"]
    plan_id = str(uuid.uuid4())
    
    yield user_id, plan_id
    
    # Teardown - remove all inserted rows for this test
    execute_sql_write("DELETE FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,))
    execute_sql_write("DELETE FROM meal_plans WHERE id = %s", (plan_id,))


@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_15days_e2e(mock_pipeline, cleanup_user_db):
    user_id, plan_id = cleanup_user_db
    
    # 1. Generate first 3 days (Simulating synchronous agent call)
    initial_form_data = {"user_id": user_id, "_days_offset": 0, "_days_to_generate": 3}
    initial_plan = _mock_run_plan_pipeline(initial_form_data)
    initial_plan["total_days_requested"] = 15
    
    # Insert initial plan in DB
    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(initial_plan))
    )
    
    # Debug: Check if inserted
    inserted = execute_sql_query("SELECT id FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True)
    assert inserted is not None, "Plan was not inserted!"
    
    active_plan = execute_sql_query(
        "SELECT id, plan_data->>'generation_status' as status FROM meal_plans WHERE id = %s",
        (plan_id,), fetch_all=False
    )
    logging.error(f"DEBUG ACTIVE PLAN IN MAIN THREAD: {active_plan}")
    
    # Queue the remaining 4 chunks (days 4-6, 7-9, 10-12, 13-15)
    valid_snapshot = {"totalDays": 15, "form_data": {"_days_offset": 0, "householdSize": 1, "dietType": "Omnívora"}}
    _enqueue_plan_chunk(user_id, plan_id, 2, 3, 3, valid_snapshot)
    _enqueue_plan_chunk(user_id, plan_id, 3, 6, 3, valid_snapshot)
    _enqueue_plan_chunk(user_id, plan_id, 4, 9, 3, valid_snapshot)
    _enqueue_plan_chunk(user_id, plan_id, 5, 12, 3, valid_snapshot)
    
    # Verify 4 items in queue
    queued = execute_sql_query("SELECT * FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,), fetch_all=True)
    assert len(queued) == 4
    
    # Force them to process
    execute_sql_write(
        "UPDATE plan_chunk_queue SET execute_after = NOW() - INTERVAL '1 MINUTE' WHERE meal_plan_id = %s",
        (plan_id,)
    )
    
    # Debug check: is it orphaned?
    orphaned = execute_sql_query(
        "SELECT id FROM plan_chunk_queue WHERE meal_plan_id = %s AND meal_plan_id::text NOT IN (SELECT id::text FROM meal_plans)", 
        (plan_id,), fetch_all=True
    )
    logging.error(f"DEBUG ORPHANED: {orphaned}")
    
    # Debug check: why is tasks empty?
    debug_state = execute_sql_query("SELECT id, status, execute_after, execute_after <= NOW() as is_ready FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,), fetch_all=True)
    logging.error(f"DEBUG QUEUE STATE: {debug_state}")
    
    debug_subquery = execute_sql_query("""
        SELECT q1.id FROM plan_chunk_queue q1
        WHERE q1.status IN ('pending', 'stale')
        AND q1.execute_after <= NOW()
        AND q1.meal_plan_id = %s
    """, (plan_id,), fetch_all=True)
    logging.error(f"DEBUG SUBQUERY: {debug_subquery}")
    
    # Run the processor until all chunks are done
    for i in range(4):
        logging.error(f"DEBUG ITERATION {i}")
        debug_iter_state = execute_sql_query("SELECT week_number, status, execute_after, NOW() as now FROM plan_chunk_queue WHERE meal_plan_id = %s ORDER BY week_number", (plan_id,), fetch_all=True)
        logging.error(f"  QUEUE BEFORE ITER {i}: {debug_iter_state}")
        
        debug_subquery_iter = execute_sql_query("""
            SELECT q1.id, q1.week_number FROM plan_chunk_queue q1
            WHERE q1.status IN ('pending', 'stale')
            AND q1.meal_plan_id = %s
            AND q1.id = (
                SELECT q2.id FROM plan_chunk_queue q2 
                WHERE q2.meal_plan_id = q1.meal_plan_id 
                AND q2.status IN ('pending', 'stale')
                ORDER BY q2.week_number ASC 
                LIMIT 1
            )
        """, (plan_id,), fetch_all=True)
        logging.error(f"  SUBQUERY MATCH FOR ITER {i}: {debug_subquery_iter}")
        
        process_plan_chunk_queue(target_plan_id=plan_id)
        
    # Verify queue is clear (completed chunks are NOT deleted immediately, they are marked 'completed')
    queued_after = execute_sql_query("SELECT status FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,), fetch_all=True)
    for q in queued_after:
        assert q['status'] == 'completed', f"Chunk not completed: {q}"
    
    # 3. Verify final plan has 15 days
    final_plan_row = execute_sql_query("SELECT plan_data FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True)
    assert final_plan_row is not None
    
    final_plan = final_plan_row["plan_data"]
    if isinstance(final_plan, str):
        final_plan = json.loads(final_plan)
        
    assert len(final_plan["days"]) == 15
    for i in range(15):
        assert final_plan["days"][i]["day"] == i + 1
        assert "Breakfast" in final_plan["days"][i]["meals"][0]["name"]
