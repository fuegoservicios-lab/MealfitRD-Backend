import sys
with open('tests/test_pantry_validation_runs_in_llm_path.py', 'a', encoding='utf-8') as f:
    f.write('''
# ---------------------------------------------------------------------------
# Helpers compartidos (simplificados)
# ---------------------------------------------------------------------------

def _run_process(tasks, prior_plan, mock_pipeline_return, user_profile=None, inventory=None):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    def _execute_sql_query(query, params=None, fetch_one=False, **_kwargs):
        if "SELECT * FROM plan_chunk_queue" in query:
            return tasks
        if "SELECT health_profile" in query:
            res = {"health_profile": user_profile or {"_pantry_quantity_mode": "strict"}}
            return res if fetch_one else [res]
        if "SELECT plan_data FROM meal_plans" in query:
            res = {"plan_data": prior_plan}
            return res if fetch_one else [res]
        if "SELECT status" in query and "plan_chunk_queue" in query:
            res = {"status": "processing", "attempts": 0}
            return res if fetch_one else [res]
        return None

    def _execute_sql_write(*args, **kwargs):
        return []

    patches = dict(
        mock_pool="db_core.connection_pool",
        mock_shop="shopping_calculator.get_shopping_list_delta",
        mock_pipeline="cron_tasks.run_plan_pipeline",
        mock_inventory="cron_tasks.get_user_inventory",
        mock_inventory_net="cron_tasks.get_user_inventory_net",
        mock_build_memory="cron_tasks.build_memory_context",
        mock_analyze="cron_tasks.analyze_preferences_agent",
        mock_write="cron_tasks.execute_sql_write",
        mock_query="cron_tasks.execute_sql_query",
        mock_vip="constants.validate_ingredients_against_pantry",
        mock_push="cron_tasks._dispatch_push_notification",
    )
    
    active_patches = {k: patch(v) for k, v in patches.items()}
    mocks = {k: ctx.__enter__() for k, ctx in active_patches.items()}

    try:
        mocks["mock_pool"].connection.return_value.__enter__.return_value = mock_conn
        mocks["mock_shop"].return_value = {"categories": []}
        
        # Pipeline return
        mocks["mock_pipeline"].return_value = mock_pipeline_return
        
        mocks["mock_inventory"].return_value = inventory if inventory is not None else ["pollo", "arroz", "tomate"]
        mocks["mock_inventory_net"].return_value = inventory if inventory is not None else ["pollo", "arroz", "tomate"]
        mocks["mock_build_memory"].return_value = {"recent_messages": [], "full_context_str": "ctx"}
        mocks["mock_analyze"].return_value = {}
        mocks["mock_write"].side_effect = _execute_sql_write
        mocks["mock_query"].side_effect = _execute_sql_query
        
        # Mock de VIP para simular comportamiento real si strict_quantities=False (existencia) 
        # y strict_quantities=True (cantidades)
        def _vip_mock(gen_ing, inv, strict_quantities=False, tolerance=1.0):
            if not strict_quantities:
                for i in gen_ing:
                    if "fantasma" in i.lower():
                        return "INEXISTENTES en inventario: ingrediente fantasma"
                return True
            else:
                return True
        mocks["mock_vip"].side_effect = _vip_mock

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_exec:
            def _sync_map(f, t):
                return [f(item) for item in t]
            def _sync_submit(fn, *args, **kwargs):
                fut = MagicMock()
                fut.result.return_value = fn(*args, **kwargs)
                return fut
            mock_exec.return_value.__enter__.return_value.map.side_effect = _sync_map
            mock_exec.return_value.__enter__.return_value.submit.side_effect = _sync_submit
            mock_exec.return_value.submit.side_effect = _sync_submit
            process_plan_chunk_queue()
    finally:
        for ctx in active_patches.values():
            ctx.__exit__(None, None, None)

    return mocks


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------
def test_phantom_ingredient_triggers_retry():
    """Test que el contador de invocaciones a validate_ingredients_against_pantry es = 1 cuando el chunk se genera por LLM con un ingrediente fantasma."""
    tasks = [{
        "id": 1,
        "user_id": "user_test",
        "meal_plan_id": "plan_test",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": {"form_data": {"current_pantry_ingredients": ["pollo", "arroz", "tomate"]}},
        "status": "pending",
        "attempts": 0
    }]
    prior_plan = {"total_days_requested": 7, "days": []}
    pipeline_return = {
        "days": [
            {"day": 4, "meals": [{"name": "Comida 1", "ingredients": ["ingrediente fantasma"]}]},
        ]
    }
    
    mocks = _run_process(tasks, prior_plan, pipeline_return, inventory=["pollo", "arroz", "tomate"])
    
    # Assert LLM was called multiple times because of the retry loop
    assert mocks["mock_pipeline"].call_count > 1
    
    # Assert validation was called
    assert mocks["mock_vip"].call_count >= 1

# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------
def test_valid_ingredients_only_invokes_llm_once():
    """Test que el LLM se invoca exactamente 1 vez cuando devuelve un new_days válido en el primer intento."""
    tasks = [{
        "id": 1,
        "user_id": "user_test",
        "meal_plan_id": "plan_test",
        "week_number": 2,
        "days_offset": 3,
        "days_count": 3,
        "pipeline_snapshot": {"form_data": {"current_pantry_ingredients": ["pollo", "arroz", "tomate"]}},
        "status": "pending",
        "attempts": 0
    }]
    prior_plan = {"total_days_requested": 7, "days": []}
    pipeline_return = {
        "days": [
            {"day": 4, "meals": [{"name": "Comida 1", "ingredients": ["pollo", "arroz"]}]},
        ]
    }
    
    mocks = _run_process(tasks, prior_plan, pipeline_return, inventory=["pollo", "arroz", "tomate"])
    
    # El LLM debió invocarse exactamente 1 vez porque no hubo drift y los ingredientes existen.
    assert mocks["mock_pipeline"].call_count == 1
    
    # Y la validación debió invocarse (una para existencia, una para cantidades, una en final step si es strict)
    assert mocks["mock_vip"].call_count >= 1
''')
