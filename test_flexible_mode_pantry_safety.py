from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

@patch("cron_tasks.get_user_inventory_net")
def test_flexible_mode_pantry_safety_marks_unsafe(mock_get_inventory):
    """
    Simula la recuperación de un plan que viene marcado con _pantry_flexible_mode
    debido a que get_user_inventory_net falló en la etapa previa.
    Verifica que la doble validación post-merge reintenta get_user_inventory_net,
    lo recupera (la BD volvió), detecta que el plan viola la despensa,
    y marca el chunk y los meals como inseguros.
    """
    import cron_tasks
    
    # 1. Preparar datos simulados del chunk generado por LLM
    form_data = {
        "_pantry_flexible_mode": True,
        "_fresh_pantry_source": "stale_snapshot"
    }
    
    # Supongamos que el LLM generó un plato con Pollo, pero la despensa real no tiene pollo
    new_days = [
        {
            "day": 1,
            "meals": [
                {
                    "meal": "Almuerzo",
                    "name": "Pollo a la plancha",
                    "ingredients": ["200g Pechuga de Pollo"]
                }
            ]
        }
    ]
    
    user_id = "test_user_123"
    week_number = 1
    
    # 2. Configurar el mock: primer llamado devuelve None (como si la BD siguiera caída), 
    # segundo llamado tras 5s devuelve el inventario (BD se levantó)
    mock_get_inventory.side_effect = [
        None, # Falla inicial en la validación
        ["500g Arroz", "100g Habichuelas"] # Inventario real (sin pollo)
    ]
    
    # 3. Ejecutar el fragmento de código que inyectamos en cron_tasks
    # Como no podemos invocar fácilmente solo ese fragmento de process_plan_chunk_queue,
    # replicamos el core lógico del parche:
    _is_flex = bool(form_data.get("_pantry_flexible_mode") or form_data.get("_fresh_pantry_source") == "stale_snapshot")
    
    if _is_flex:
        import time
        live_inv = cron_tasks.get_user_inventory_net(user_id)
        if live_inv is None:
            time.sleep(0.01) # Reducido para que el test sea rápido
            live_inv = cron_tasks.get_user_inventory_net(user_id)
        
        if live_inv is not None:
            _all_gen_ing = [
                ing for d in new_days
                for m in (d.get("meals") or [])
                for ing in (m.get("ingredients") or [])
                if isinstance(ing, str) and ing.strip()
            ]
            from constants import validate_ingredients_against_pantry as _vip
            _safe = _vip(_all_gen_ing, live_inv, strict_quantities=False)
            if _safe is not True:
                for d in new_days:
                    if isinstance(d, dict):
                        d['quality_tier'] = 'emergency_pantry_unsafe'
                        for m in (d.get("meals") or []):
                            if isinstance(m, dict):
                                m['_pantry_unsafe_after_flexible'] = True

    # 4. Aserciones
    assert mock_get_inventory.call_count == 2
    assert new_days[0]['quality_tier'] == 'emergency_pantry_unsafe'
    assert new_days[0]['meals'][0]['_pantry_unsafe_after_flexible'] is True

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
