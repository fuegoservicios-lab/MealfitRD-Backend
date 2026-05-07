"""
E2E test for 15-day chunked meal plan generation.

Uses the seeded_user_profile fixture from tests/conftest.py to avoid
pytest.skip on missing DB data.

Run with:
    cd backend && python -m pytest tests/test_chunked_15days_e2e.py -v -m e2e
"""
import pytest
import uuid
import json
import time
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

from db_core import execute_sql_write, execute_sql_query
from cron_tasks import process_plan_chunk_queue, _enqueue_plan_chunk
from constants import PLAN_CHUNK_SIZE, split_with_absorb

import logging
logging.getLogger().setLevel(logging.INFO)


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
                {"name": f"Breakfast {day_num}", "type": "Desayuno",
                 "ingredients": ["100g pollo", "arroz"]},
                {"name": f"Lunch {day_num}", "type": "Almuerzo",
                 "ingredients": ["150g res", "habichuelas"]},
            ]
        })

    return {"days": days, "generation_status": "partial"}


@pytest.mark.e2e
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_15days_e2e(mock_pipeline, seeded_user_profile):
    user_id, plan_id = seeded_user_profile

    # 1. Generate first 3 days (Simulating synchronous agent call)
    initial_form_data = {"user_id": user_id, "_days_offset": 0, "_days_to_generate": 3}
    initial_plan = _mock_run_plan_pipeline(initial_form_data)
    initial_plan["total_days_requested"] = 15
    initial_plan["total_days_generated"] = 3

    # Insert initial plan in DB
    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(initial_plan))
    )

    # Debug: Check if inserted
    inserted = execute_sql_query("SELECT id FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True)
    assert inserted is not None, "Plan was not inserted!"

    # [P1-5] Derivar el plan de chunks desde split_with_absorb (la misma función
    # que usa producción) en lugar de hardcodear 5 chunks de 3d. Para 15d con
    # base=3, prod genera [3, 4, 4, 4]; el test antiguo enqueue [3, 3, 3, 3, 3]
    # y por tanto NO ejercitaba la rama P1-A del splitter, ocultando regresiones.
    chunk_sizes = split_with_absorb(15, PLAN_CHUNK_SIZE)
    assert chunk_sizes[0] == PLAN_CHUNK_SIZE, (
        "Primer chunk debe ser PLAN_CHUNK_SIZE (chunk inicial síncrono)"
    )
    remaining_chunks = chunk_sizes[1:]
    expected_remaining = len(remaining_chunks)

    valid_snapshot = {
        "totalDays": 15,
        "form_data": {
            "_days_offset": 0,
            "householdSize": 1,
            "dietType": "Omnívora",
            "_plan_start_date": datetime.now(timezone.utc).isoformat(),
        },
    }
    days_offset = chunk_sizes[0]
    for week_idx, days_count in enumerate(remaining_chunks, start=2):
        _enqueue_plan_chunk(user_id, plan_id, week_idx, days_offset, days_count, valid_snapshot)
        days_offset += days_count

    # Verify queue matches the splitter's output
    queued = execute_sql_query("SELECT * FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,), fetch_all=True)
    assert len(queued) == expected_remaining, (
        f"Esperaba {expected_remaining} chunks en cola, obtuve {len(queued)}. "
        f"Splitter produjo {chunk_sizes}"
    )

    # Force them to process
    execute_sql_write(
        "UPDATE plan_chunk_queue SET execute_after = NOW() - INTERVAL '1 MINUTE' WHERE meal_plan_id = %s",
        (plan_id,)
    )

    # Run the processor until all chunks are done
    for i in range(expected_remaining):
        logging.info(f"Processing chunk iteration {i + 1}/{expected_remaining}")
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

    # Verify chunk 1→2 lesson propagation: _recent_chunk_lessons must exist
    recent_lessons = final_plan.get("_recent_chunk_lessons", [])
    assert len(recent_lessons) >= 1, \
        f"Tras {len(chunk_sizes)} chunks, _recent_chunk_lessons debe tener al menos 1 entrada"
