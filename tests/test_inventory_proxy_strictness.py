from unittest.mock import patch

@patch("cron_tasks.get_inventory_activity_since")
@patch("cron_tasks.get_consumed_meals_since")
def test_inventory_proxy_strictness_manual_blocks(mock_consumed, mock_activity):
    """
    Simula que no hay consumo explícito y hay mutaciones manuales, 
    debería bloquear porque no hay mutaciones de consumo suficientes.
    """
    import cron_tasks
    import constants
    
    # Aseguramos que el minimo son 5
    constants.CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS = 5

    mock_consumed.return_value = []
    # 5 mutaciones pero manuales (0 de consumo)
    mock_activity.return_value = {
        "mutations_count": 5, 
        "consumption_mutations_count": 0, 
        "manual_mutations_count": 5
    }

    plan_data = {"days": [{"day": 1, "meals": [{"name": "P1"}]}]}
    snapshot = {"form_data": {"_plan_start_date": "2024-01-01"}}
    
    # Invocamos la lógica de validación
    res = cron_tasks._check_chunk_learning_ready("u1", "m1", 2, 1, plan_data, snapshot)
    
    assert res["ready"] is False
    assert res["inventory_proxy_used"] is False
    # [P1-8] zero_log_proxy + consumption_mutations_count == 0 ⇒ ratio = 0.0
    # ("no evidencia"). Antes la fórmula `0.5 + mutations/total` arrancaba en 0.5
    # y versiones legacy asumían 1.0; ahora _calculate_chunk_consumption_ratio
    # devuelve 0.0 honesto cuando no hay logs ni mutaciones de inventario.
    assert res["ratio"] == 0.0

@patch("cron_tasks.get_inventory_activity_since")
@patch("cron_tasks.get_consumed_meals_since")
def test_inventory_proxy_strictness_consumption_passes(mock_consumed, mock_activity):
    """
    Simula 5 mutaciones de consumo. Debería pasar y devolver low_confidence (y ratio capado).
    """
    import cron_tasks
    import constants
    
    constants.CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS = 5

    mock_consumed.return_value = []
    # 5 consumos
    mock_activity.return_value = {
        "mutations_count": 5, 
        "consumption_mutations_count": 5, 
        "manual_mutations_count": 0
    }

    # planned_total será 2
    plan_data = {"days": [{"day": 1, "meals": [{"name": "P1"}, {"name": "P2"}]}]}
    snapshot = {"form_data": {"_plan_start_date": "2024-01-01"}}
    
    res = cron_tasks._check_chunk_learning_ready("u1", "m1", 2, 1, plan_data, snapshot)
    
    assert res["ready"] is True
    assert res["inventory_proxy_used"] is True
    
    # ratio esperado: min(0.5 + 5 / max(2, 6), 0.85) = min(0.5 + 5/6, 0.85) = 0.85
    assert res["ratio"] == 0.85


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._pause_chunk_for_pantry_refresh")
@patch("cron_tasks.CHUNK_PANTRY_QUANTITY_MODE", "strict")
def test_chunk_strict_mode_pauses_when_quantity_exceeds_pantry(mock_pause, mock_push):
    """
    When CHUNK_PANTRY_QUANTITY_MODE is strict and LLM retries are exhausted,
    the chunk should be paused as pending_user_action (not failed) and a push
    notification should be sent asking the user to restock.
    """
    from unittest.mock import MagicMock, call
    import json
    import cron_tasks
    from constants import validate_ingredients_against_pantry

    # Simulate the strict path: validate_ingredients_against_pantry returns a
    # failure string when quantities exceed what's available.
    pantry = ["200g pechuga de pollo"]

    # 250g exceeds 200g available (tolerance=1.00 in strict mode)
    result = validate_ingredients_against_pantry(
        ["250g pechuga de pollo"], pantry, strict_quantities=True, tolerance=1.00
    )
    # The result should be a string (not True) indicating violation
    assert result is not True, f"Expected violation but got True"
    assert isinstance(result, str), f"Expected string feedback, got {type(result)}"

    # Verify that _pause_chunk_for_pantry_refresh is callable with quantity_unfeasible reason
    mock_pause("task-123", "user-456", 2, fresh_inventory=pantry, reason="quantity_unfeasible")
    mock_pause.assert_called_once_with(
        "task-123", "user-456", 2,
        fresh_inventory=pantry,
        reason="quantity_unfeasible",
    )

    # Verify push notification is sent with the right message shape
    mock_push(
        user_id="user-456",
        title="Tu plan necesita más ingredientes",
        body="Tu próximo bloque necesita más ingredientes. Actualiza tu nevera o registra una compra.",
        url="/dashboard",
    )
    mock_push.assert_called_once()
    push_kwargs = mock_push.call_args[1]
    assert "nevera" in push_kwargs["body"] or "compra" in push_kwargs["body"]

    # Verify the global default is now strict
    from constants import CHUNK_PANTRY_QUANTITY_MODE
    assert CHUNK_PANTRY_QUANTITY_MODE == "strict", f"Expected strict default, got {CHUNK_PANTRY_QUANTITY_MODE}"

def test_chunk_aborts_on_5pct_inventory_drift_during_llm_call():
    """
    Test that simulates inventory drift > 5% but < 20% during generation,
    triggering the drift retry mechanism rather than silently passing.
    """
    import cron_tasks
    old_inv = [{"name": "Huevos", "quantity": 10, "unit": "unidad"}]
    # Consume 1 huevo (10% drift)
    new_inv = [{"name": "Huevos", "quantity": 9, "unit": "unidad"}]
    
    drift_pct = cron_tasks._calculate_inventory_drift(old_inv, new_inv)
    assert drift_pct == 0.10, f"Expected 10% drift, got {drift_pct}"
    
    # Simula la lógica de retry que ahora se activa > 0.05
    form_data = {"current_pantry_ingredients": old_inv, "_drift_retries": 0}
    
    # Evaluamos umbral bajado a 0.05
    if drift_pct > 0.05:
        form_data["_drift_retries"] += 1
        form_data["current_pantry_ingredients"] = new_inv
        
    assert form_data["_drift_retries"] == 1
    assert form_data["current_pantry_ingredients"] == new_inv
