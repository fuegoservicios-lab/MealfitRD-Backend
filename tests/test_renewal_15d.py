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
    
    # [test-drift fix] api_shift_plan resuelve plan_id (SELECT id), luego
    # SELECT plan_data FOR UPDATE (split P2-LOCK-2), y dentro de la rama de
    # renovación lee health_profile, COUNT chunks-en-vuelo y MAX(week_number);
    # además helpers compartidos (get_user_inventory_net) consumen el MISMO cursor
    # mockeado. Un side_effect posicional rígido se rompe ante cualquier reorden.
    # Devolvemos por contenido de la última query → robusto ante el orden exacto.
    def _fetchone_by_query(*_a, **_k):
        last_q = ""
        for call in reversed(mock_cursor.execute.call_args_list):
            if call.args:
                last_q = call.args[0]
                break
        if "SELECT id FROM meal_plans" in last_q:
            return {"id": "plan-123"}
        if "SELECT plan_data" in last_q:
            return {"plan_data": plan_data}
        if "health_profile" in last_q:
            return {"health_profile": {"diet": "keto"}}
        if "COUNT(*)" in last_q:
            return {"cnt": 0}
        if "MAX(week_number)" in last_q:
            return {"max_week": 5}
        return {}

    mock_cursor.fetchone.side_effect = _fetchone_by_query
    mock_cursor.fetchall.side_effect = lambda *a, **k: []

    from fastapi import Response
    data = {"user_id": "test_user"}

    res = api_shift_plan(Response(), data, verified_user_id="test_user")

    assert res.get("success") is True, f"Failed: {res.get('message')}"

    # [test-drift fix] split_with_absorb(15, 3) == [3, 4, 4, 4] tras P1-A
    # (antes [3,3,3,3,3]). 4 chunks, offsets acumulados [0, 3, 7, 11].
    assert mock_enqueue.call_count == 4, f"Expected 4 chunks, got {mock_enqueue.call_count}"

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

    assert offsets == [0, 3, 7, 11], f"Expected offsets [0, 3, 7, 11], got {offsets}"
    assert chunk_sizes == [3, 4, 4, 4], f"Expected sizes [3, 4, 4, 4], got {chunk_sizes}"
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
    # [test-drift fix] api_analyze ahora valida presencia + rangos + enums de
    # los campos del formulario (P1-5/P1-3/P0-FORM-5) ANTES de llegar a la
    # lógica de chunking. Un payload mínimo de 5 campos disparaba 422; aquí
    # proveemos un set completo y VÁLIDO (refleja un request real del wizard).
    data = {
        "user_id": "test_user",
        "session_id": "test_session",
        "totalDays": 15,
        "tzOffset": 240,
        "mainGoal": "lose_fat",
        "age": 30,
        "weight": 70,
        "weightUnit": "kg",
        "height": 170,
        "gender": "male",
        "activityLevel": "moderate",
        "householdSize": 1,
        "groceryDuration": "weekly",
        "motivation": "Quiero sentirme mejor",
        "allergies": ["Ninguna"],
        "medicalConditions": ["Ninguna"],
        "scheduleType": "standard",
        "cookingTime": "30min",
        "budget": "medium",
        "sleepHours": "7-8 horas",
        "stressLevel": "Bajo",
        "dislikes": [],
        "struggles": [],
    }
    
    from fastapi import Response
    bg_tasks = BackgroundTasks()
    # [test-drift fix] api_analyze ahora toma `response: Response` como 2º
    # parámetro posicional (entre background_tasks y data). El test pasaba `data`
    # en la posición de `response`, dejando `data` con su default Body(...).
    res = api_analyze(bg_tasks, Response(), data, verified_user_id="test_user")
    
    # Verify the response
    assert res.get("generation_status") == "partial"
    assert res.get("id") == "new-plan-id"
    
    # Verify the DB call for the prior plan
    called_queries = [call[0][0] for call in mock_execute_sql.call_args_list]
    assert any("SELECT plan_data FROM meal_plans WHERE user_id" in q for q in called_queries)
    
    # [test-drift fix] Plan de 15 días: la semana 1 (3 días) se genera de forma
    # SÍNCRONA (run_plan_pipeline) y solo se encolan chunks[1:].
    # split_with_absorb(15, 3) == [3, 4, 4, 4] tras P1-A → semanas 2,3,4
    # encoladas = 3 chunks (antes [3,3,3,3,3] → 4). Verificado contra
    # _postprocess_pipeline_result (routers/plans.py:1312-1333).
    assert mock_enqueue.call_count == 3, f"Expected 3 chunks enqueued, got {mock_enqueue.call_count}"

    # [test-drift fix] _postprocess_pipeline_result aplica
    # `_inherited_lifetime_lessons` a TODOS los chunks encolados del plan nuevo
    # (no solo a la semana 2). El consumo aguas abajo es idempotente
    # (`if not _history`), así que no hay doble conteo. La invariante real que
    # protege este test es que la herencia cross-plan se propaga con el
    # contenido correcto desde el plan previo.
    weeks_seen = []
    inherited = False
    for i, call in enumerate(mock_enqueue.call_args_list):
        args, kwargs = call
        user_id, plan_id, wk, offset, count, snapshot = args
        weeks_seen.append(wk)
        assert "_inherited_lifetime_lessons" in snapshot
        assert snapshot["_inherited_lifetime_lessons"]["history"] == [{"rejection_violations": 2}]
        inherited = True

    assert weeks_seen == [2, 3, 4], f"Expected weeks [2, 3, 4], got {weeks_seen}"
            
    assert inherited is True
