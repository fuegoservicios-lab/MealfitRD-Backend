import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from cron_tasks import _check_chunk_learning_ready

@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.execute_sql_query")
def test_chunk_temporal_gate_race_condition(mock_sql_query, mock_now):
    """
    Test que simula chunk1=[D1,D2,D3] terminado el día 3 a las 23:00.
    Dispara cron, assert learning_ready={"ready":False,"reason":"prev_chunk_day_not_yet_elapsed"} 
    hasta que now >= D4 00:00.
    """
    user_id = "test_user_race"
    meal_plan_id = "plan_123"
    
    # Supongamos que el plan empezó hace 2 días (D1 = hace 2 días, D2 = ayer, D3 = hoy)
    # Por lo tanto plan_start_dt fue hace 2 días a las 00:00.
    real_now = datetime.now(timezone.utc)
    plan_start_dt = (real_now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    snapshot = {
        "form_data": {
            "_plan_start_date": plan_start_dt.isoformat(),
            "tz_offset_minutes": 0
        },
        "totalDays": 7
    }
    
    # El chunk previo fue de D1 a D3. 
    # week_number = 2 (significa que es el chunk 2), days_offset = 3
    week_number = 2
    days_offset = 3
    
    plan_data = {
        "days": [
            {"day": 1, "meals": []},
            {"day": 2, "meals": []},
            {"day": 3, "meals": []}
        ]
    }
    
    mock_sql_query.return_value = None  # profile no tiene info extra
    
    # 1. Simulamos que son las 23:00 del día 3.
    # El final del día 3 es plan_start_dt + 3 días (es decir mañana a las 00:00).
    now_2300 = plan_start_dt + timedelta(days=2, hours=23)
    mock_now.return_value = now_2300
    
    result_early = _check_chunk_learning_ready(
        user_id=user_id,
        meal_plan_id=meal_plan_id,
        week_number=week_number,
        days_offset=days_offset,
        plan_data=plan_data,
        snapshot=snapshot
    )
    
    assert result_early["ready"] is False
    assert result_early["reason"] == "prev_chunk_day_not_yet_elapsed"
    
    # 2. Simulamos que ya es el día 4 a las 00:01
    now_d4_0001 = plan_start_dt + timedelta(days=3, minutes=1)
    mock_now.return_value = now_d4_0001
    
    # Para la segunda llamada mockeamos get_consumed_meals_since y get_inventory_activity_since 
    # ya que pasará el gate temporal y requerirá evaluar base de datos.
    with patch("cron_tasks.get_consumed_meals_since", return_value=[]), \
         patch("cron_tasks.get_inventory_activity_since", return_value={}):
         
        result_late = _check_chunk_learning_ready(
            user_id=user_id,
            meal_plan_id=meal_plan_id,
            week_number=week_number,
            days_offset=days_offset,
            plan_data=plan_data,
            snapshot=snapshot
        )
        
    assert result_late.get("reason") != "prev_chunk_day_not_yet_elapsed"
