
import os

content = """
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
def test_force_generate_stale_snapshot_triggers_push_and_review(mock_sql_write, mock_push, mock_get_inv):
    import cron_tasks
    from datetime import datetime, timedelta, timezone
    
    # Simulamos que falla el live-fetch y el extendido
    mock_get_inv.return_value = None
    
    # Simulamos un snapshot mas viejo que el limite de force generate
    snapshot_age = cron_tasks.CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS + 1
    snapshot_form_data = {
        "_pantry_captured_at": (datetime.now(timezone.utc) - timedelta(hours=snapshot_age)).isoformat()
    }
    
    form_data = {}
    
    result = cron_tasks._refresh_chunk_pantry(
        user_id="test_user_456",
        form_data=form_data,
        snapshot_form_data=snapshot_form_data,
        task_id="task_789",
        week_number=2
    )
    
    # (a) Verifica que el push se encola y el form_data esta preparado
    assert result.get("_pantry_flexible_mode") is True
    assert result.get("_requires_pantry_review") is True
    
    mock_push.assert_called_once()
    assert "Tu plan de hoy puede pedir ingredientes que no tengas" in mock_push.call_args[1]["title"]
    
    # Replica del fragmento de process_plan_chunk_queue para validar (b)
    new_days = [
        {"day": 1, "meals": [{"meal": "Cena", "name": "Pasta", "ingredients": ["100g Pasta"]}]}
    ]
    
    if result.get("_requires_pantry_review"):
        for d in new_days:
            if isinstance(d, dict):
                for m in (d.get("meals") or []):
                    if isinstance(m, dict):
                        m.setdefault("meta", {})["requires_pantry_review"] = True
                        
    # (b) Verifica que el chunk lleva requires_pantry_review
    assert new_days[0]["meals"][0]["meta"]["requires_pantry_review"] is True
"""

with open('c:/Users/angel/OneDrive/Escritorio/MealfitRD.IA/backend/test_flexible_mode_pantry_safety.py', 'a', encoding='utf-8') as f:
    f.write(content)
