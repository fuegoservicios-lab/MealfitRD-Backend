import pytest
import json
from unittest.mock import patch, MagicMock

from cron_tasks import _enqueue_plan_chunk, process_plan_chunk_queue

from constants import PLAN_CHUNK_SIZE

@patch('cron_tasks.execute_sql_write')
def test_enqueue_plan_chunk_offset_and_delay(mock_write):
    _enqueue_plan_chunk('user_123', 'plan_456', 2, 7, PLAN_CHUNK_SIZE, {"fake": "snapshot"})
    
    mock_write.assert_called_once()
    args, kwargs = mock_write.call_args
    query = args[0]
    params = args[1]
    
    assert "INSERT INTO plan_chunk_queue" in query
    assert params[0] == 'user_123'
    assert params[1] == 'plan_456'
    assert params[2] == 2
    assert params[3] == 7
    assert params[4] == PLAN_CHUNK_SIZE
    assert params[6] == 6


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
@patch('db.get_user_likes')
@patch('db.get_active_rejections')
@patch('db_facts.get_consumed_meals_since')
@patch('db_facts.get_all_user_facts')
def test_chunk_degraded_fallback(mock_facts, mock_consumed, mock_rejections, mock_likes, mock_write, mock_query, mock_shop, mock_pool, mock_llm):
    mock_llm.return_value.invoke.side_effect = Exception("Simulated LLM Outage")
    mock_likes.return_value = []
    mock_rejections.return_value = []
    mock_consumed.return_value = []
    mock_facts.return_value = []
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
    mock_cron_likes, mock_pipeline, mock_shop, mock_pool
):
    tasks = [{
        "id": 1,
        "user_id": "user_123",
        "meal_plan_id": "plan_456",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": {
            "form_data": {"allergies": ["Maní"]}
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
    mock_db_metadata.return_value = [{"fact": "Mariscos"}]
    
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


@patch('langchain_google_genai.ChatGoogleGenerativeAI')
@patch('db_core.connection_pool')
@patch('shopping_calculator.get_shopping_list_delta')
@patch('cron_tasks.execute_sql_query')
@patch('cron_tasks.execute_sql_write')
@patch('db.get_user_likes')
@patch('db.get_active_rejections')
@patch('db_facts.get_consumed_meals_since')
@patch('db_facts.get_all_user_facts')
def test_edge_case_one_or_two_days(mock_facts, mock_consumed, mock_rejections, mock_likes, mock_write, mock_query, mock_shop, mock_pool, mock_llm):
    mock_llm.return_value.invoke.side_effect = Exception("Simulated LLM Outage")
    mock_likes.return_value = []
    mock_rejections.return_value = []
    mock_consumed.return_value = []
    mock_facts.return_value = []
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
    mock_cron_likes, mock_pipeline, mock_shop, mock_pool
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