"""E2E test for the canonical 7-day plan scenario described by product:

User picks a 7-day plan starting on Thursday. The system must:
  1. Generate the first 3 days (Thursday, Friday, Saturday) as chunk 1.
  2. Once Saturday concludes, generate the next 4 days (Sun-Wed) as chunk 2,
     using the learning from chunk 1.

This split (3 + 4) is what `split_with_absorb(7, base=3)` returns at
constants.py:298. The 15-day E2E test exercises chunks of size 3 only;
neither it nor any other test in the suite exercises the 4-day chunk that
makes a 7-day plan complete.

This test also exercises the P0-4 last-resort synthesis path: chunk 1 is
inserted directly into plan_data without a corresponding plan_chunk_queue
row, so when chunk 2 starts and calls _rebuild_last_chunk_learning_from_queue,
that returns None. The new _synthesize_last_chunk_learning_from_plan_days
helper should then build a stub from plan_data.days. Verified working in
the run that introduced this test:
    [P0-4/SYNTHESIZED] _last_chunk_learning sintetizado desde plan_data.days
    chunk 2 (prev_week=1, meals=6, bases=4, ...). Last-resort tras fallar
    plan_chunk_queue.learning_metrics.

STATUS al 2026-05-02:
  1. ✅ FIXED — `chunk_user_locks.locked_by_chunk_id` ya es uuid en producción
     (migración `supabase/p1_chunk_user_locks_uuid_fix.sql` aplicada). El INSERT
     del lock acquire en cron_tasks.py:11325 y el DELETE del release en :16319
     ya no emiten el "column ... is of type bigint but expression is of type uuid".
     Regression guard a nivel schema: tests/test_p0_3_chunk_user_locks_schema.py.
  2. ✅ FIXED — el log mentiroso "ratio 50% < 50%" se reescribió en
     cron_tasks.py:12321-12345: ahora explica la causa real (zero_log,
     sparse_log, o ratio<umbral). Adicionalmente, este test usa
     `_learning_flexible_mode=True` que bypasa el gate por completo.

KNOWN FAILURE residual (NO P0-3):
  El chunk 2 termina en `pending_user_action` porque el inventario fresco
  del usuario seeded está vacío y el reconciliador de pantry agota sus 3
  intentos con 0/8 reservas. Esto NO es un bug del flujo de chunks — es
  comportamiento correcto cuando faltan ingredientes. Rompe el test porque
  la fixture `seeded_user_profile` no siembra inventario suficiente para
  cubrir las 4 días de comidas que pide chunk 2.

  Este síntoma pertenece a la familia P0-5 (validación de pantry por chunk
  + fixture de seeding). Para que este test E2E pase end-to-end hay que:
    a) Sembrar inventario de pollo, arroz, res, habichuelas en cantidades
       suficientes para 4 días en la fixture, O
    b) Mockear release_chunk_reservations / pantry validation en este test.

Run with:
    cd backend && python -m pytest tests/test_chunked_7days_thursday_e2e.py -v -m e2e
"""
import json
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from db_core import execute_sql_write, execute_sql_query
from cron_tasks import process_plan_chunk_queue, _enqueue_plan_chunk

logging.getLogger().setLevel(logging.INFO)


def _mock_run_plan_pipeline(form_data, *args, **kwargs):
    """Mirror the 15d-e2e mock: returns N days based on offset and count.

    This avoids hitting the real LLM and keeps the test deterministic.
    """
    offset = form_data.get("_days_offset", 0)
    count = form_data.get("_days_to_generate", 3)

    days = []
    for i in range(count):
        day_num = offset + i + 1
        days.append({
            "day": day_num,
            "daily_summary": f"Day {day_num} summary",
            "meals": [
                {
                    "name": f"Breakfast {day_num}",
                    "type": "Desayuno",
                    "ingredients": ["100g pollo", "arroz"],
                },
                {
                    "name": f"Lunch {day_num}",
                    "type": "Almuerzo",
                    "ingredients": ["150g res", "habichuelas"],
                },
            ],
        })

    return {"days": days, "generation_status": "partial"}


@pytest.mark.e2e
@patch("cron_tasks.run_plan_pipeline", side_effect=_mock_run_plan_pipeline)
def test_thursday_7day_3plus4_split_e2e(mock_pipeline, seeded_user_profile):
    """Canonical 7-day Thursday-start plan: chunk 1 is 3 days, chunk 2 is 4 days.

    Asserts:
      - Final plan has exactly 7 days, numbered 1..7 (no off-by-one in the
        4-day chunk).
      - Chunk 2 reaches status='completed' (proves the temporal gate, learning
        gate, and pantry validation all let the 4-day chunk through).
      - After chunk 2 runs, plan_data._last_chunk_learning is non-empty —
        the P0-4 synthesis fallback OR the existing chunk-completion stub
        path populated something. An empty dict here would mean the
        continuous-learning promise is broken for the most common plan length.
    """
    user_id, plan_id = seeded_user_profile

    # Anchor the plan on a Thursday safely in the past so the temporal gate
    # for chunk 2 ("today_user must be past the end of chunk 1") is trivially
    # satisfied without needing to mock _dt_p0b_now().
    today = datetime.now(timezone.utc)
    days_back = (today.weekday() - 3) % 7 + 14  # 3 == Thursday in weekday()
    plan_start = today - timedelta(days=days_back)
    assert plan_start.weekday() == 3, "anchor must be a Thursday"

    # 1. Seed the initial plan with chunk 1 already generated (Thu/Fri/Sat).
    initial_form_data = {"user_id": user_id, "_days_offset": 0, "_days_to_generate": 3}
    initial_plan = _mock_run_plan_pipeline(initial_form_data)
    initial_plan["total_days_requested"] = 7
    initial_plan["total_days_generated"] = 3
    initial_plan["grocery_start_date"] = plan_start.isoformat()

    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(initial_plan)),
    )

    inserted = execute_sql_query(
        "SELECT id FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True
    )
    assert inserted is not None, "plan was not inserted"

    # 2. Enqueue chunk 2 — this is the 4-day chunk that makes 7d plans special.
    # _learning_flexible_mode=True bypasses the learning-readiness gate at
    # cron_tasks.py:7491-7496. Without it, a freshly-seeded user with 0 explicit
    # logs and 0 inventory mutations triggers zero_log_proxy → ready=False (by
    # design at cron_tasks.py:6076-6079) and the chunk gets deferred 12h. The
    # bypass is a real production feature (set when the user opts into "skip
    # logging" or after a recovery cron escalation), so exercising it here is
    # legitimate — the test is about the 3+4 split and learning propagation,
    # not about gate behavior under fresh-user conditions.
    valid_snapshot = {
        "totalDays": 7,
        "_learning_flexible_mode": True,
        "form_data": {
            "_days_offset": 3,
            "_days_to_generate": 4,
            "householdSize": 1,
            "dietType": "Omnívora",
            "_plan_start_date": plan_start.isoformat(),
        },
    }
    _enqueue_plan_chunk(user_id, plan_id, week_number=2, days_offset=3, days_count=4,
                        pipeline_snapshot=valid_snapshot)

    queued = execute_sql_query(
        "SELECT * FROM plan_chunk_queue WHERE meal_plan_id = %s",
        (plan_id,), fetch_all=True,
    )
    assert len(queued) == 1, f"expected 1 chunk queued, got {len(queued)}"
    assert queued[0]["days_count"] == 4, "chunk 2 must be a 4-day chunk"
    assert queued[0]["days_offset"] == 3, "chunk 2 must start at day-offset 3"

    # 3. Force the chunk to be eligible for processing now.
    execute_sql_write(
        "UPDATE plan_chunk_queue SET execute_after = NOW() - INTERVAL '1 MINUTE' "
        "WHERE meal_plan_id = %s",
        (plan_id,),
    )

    # 4. Process. One iteration should be enough since there is only one chunk.
    # Run twice as a safety margin in case a deferral inserts a follow-up row.
    for i in range(2):
        logging.info(f"processing iteration {i + 1}/2")
        process_plan_chunk_queue(target_plan_id=plan_id)

    # 5. Verify chunk 2 reached status='completed'.
    chunk2_rows = execute_sql_query(
        "SELECT status, days_offset, days_count FROM plan_chunk_queue "
        "WHERE meal_plan_id = %s AND week_number = 2",
        (plan_id,), fetch_all=True,
    )
    assert chunk2_rows, "chunk 2 row vanished from queue"
    assert chunk2_rows[0]["status"] == "completed", (
        f"chunk 2 not completed: status={chunk2_rows[0]['status']}. "
        "Inspect cron_tasks logs for [P0-3], [P0-4], or temporal gate deferrals."
    )

    # 6. Verify the final plan has exactly 7 days, numbered 1..7.
    final_row = execute_sql_query(
        "SELECT plan_data FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True
    )
    assert final_row is not None
    final_plan = final_row["plan_data"]
    if isinstance(final_plan, str):
        final_plan = json.loads(final_plan)

    days = final_plan.get("days") or []
    assert len(days) == 7, f"expected 7 days total, got {len(days)}"
    day_numbers = sorted(int(d["day"]) for d in days)
    assert day_numbers == list(range(1, 8)), (
        f"day numbering off: {day_numbers}. The 4-day chunk likely produced "
        "an off-by-one in days 4-7."
    )

    # 7. Verify learning was populated for chunk 2 — empty here means the
    # continuous-learning promise is broken for 7-day plans.
    last_lesson = final_plan.get("_last_chunk_learning")
    assert isinstance(last_lesson, dict) and last_lesson, (
        "_last_chunk_learning is missing or empty after chunk 2 completed. "
        "Either the P0-4 synthesis fallback failed or the chunk-completion "
        "stub path at cron_tasks.py:9549 didn't run."
    )
    assert last_lesson.get("chunk") in (1, 2), (
        f"_last_chunk_learning.chunk={last_lesson.get('chunk')}, expected 1 or 2"
    )
