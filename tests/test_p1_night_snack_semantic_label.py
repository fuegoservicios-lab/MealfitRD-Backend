"""
Tests para [BB-033 / P1-NIGHT-SNACK-SEMANTIC-LABEL · 2026-11-04]
Etiquetado semántico en el "Rescate Nocturno" cuando ya existe una cena previa (BB-026 + BB-004).
"""

import pytest
from unittest.mock import patch, MagicMock
from tools import _buscar_cena_principal_hoy, log_consumed_meal
from respuestas_de_la_foto import instruccion, instruccion_solo_foto


def test_buscar_cena_principal_hoy_queries_correctly():
    """_buscar_cena_principal_hoy filtra por meal_type='cena' y calories >= 350 en el día biológico."""
    user_id = "user-test-bb033"

    with patch("tools.user_tz_offset_min", return_value=240), \
         patch("db_facts._diary_rollover_hour", return_value=4), \
         patch("db.execute_sql_query", return_value=[{
             "id": "c1",
             "meal_type": "cena",
             "meal_name": "Pechuga con moro y ensalada",
             "calories": 650,
             "protein": 52,
             "carbs": 60,
             "healthy_fats": 18,
             "consumed_at": "2026-11-04T20:30:00Z",
             "created_at": "2026-11-04T20:30:00Z"
         }]) as mock_query:

        res = _buscar_cena_principal_hoy(user_id)
        assert res is not None
        assert res["id"] == "c1"
        assert res["calories"] == 650

        # Verificar que el SQL exige calories >= 350 y meal_type = 'cena'
        sql = mock_query.call_args[0][0]
        assert "meal_type = 'cena'" in sql
        assert "calories >= 350" in sql


def test_night_moderate_intake_with_prior_cena_classified_as_merienda():
    """Si a las 11:40 PM el usuario sube un postre o picadera de 220 kcal sin franja explícita

    y ya tiene su cena principal registrada hoy (650 kcal), se registra como 'merienda'
    y el aviso instruye al coach a confirmarlo como merienda nocturna / extra de la noche."""
    user_id = "user-test-bb033"

    mock_cena = {
        "id": "cena-01",
        "meal_type": "cena",
        "meal_name": "Pechuga con moro",
        "calories": 650
    }

    with patch("tools._hora_local_float", return_value=23.67), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._buscar_cena_principal_hoy", return_value=mock_cena), \
         patch("tools.db_log_consumed_meal", return_value="meal-snack-night-1") as mock_db_log:

        res = log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Postre tres leches",
            "calories": 220,
            "protein": 5,
            "carbs": 30,
            "healthy_fats": 8,
            "meal_type": "cena",  # El LLM mandó 'cena' por defecto de noche
            "explicit_slot": False,
        })

        # Se registra en base de datos como 'merienda'
        assert mock_db_log.call_args[1]["meal_type"] == "merienda"

        # El mensaje retornado al asistente incluye la directiva conversacional
        assert "merienda nocturna / extra de la noche" in res
        assert "650 kcal" in res
        assert "NUNCA digas «Anoté como cena» por segunda vez en la misma noche" in res


def test_night_intake_without_prior_cena_rescued_to_cena():
    """Si el usuario NO tiene cena previa registrada hoy y envía una comida sustancial de noche,

    se rescata a cena normalmente."""
    user_id = "user-test-bb033"

    with patch("tools._hora_local_float", return_value=21.0), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._buscar_cena_principal_hoy", return_value=None), \
         patch("tools._rescue_dinner_slot", return_value="cena"), \
         patch("tools.db_log_consumed_meal", return_value="meal-cena-1") as mock_db_log:

        res = log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Plátano hervido con queso y salami",
            "calories": 550,
            "protein": 28,
            "carbs": 58,
            "healthy_fats": 22,
            "meal_type": "merienda",
            "explicit_slot": False,
        })

        assert mock_db_log.call_args[1]["meal_type"] == "cena"
        assert "merienda nocturna / extra de la noche" not in res


def test_night_intake_with_explicit_slot_preserves_explicit_cena():
    """Si el usuario indicó explícitamente la franja (explicit_slot=True), se respeta su decisión."""
    user_id = "user-test-bb033"

    with patch("tools._hora_local_float", return_value=23.5), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools.db_log_consumed_meal", return_value="meal-cena-2") as mock_db_log:

        res = log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Segunda cena de tacos",
            "calories": 300,
            "protein": 20,
            "carbs": 25,
            "healthy_fats": 10,
            "meal_type": "cena",
            "explicit_slot": True,
        })

        assert mock_db_log.call_args[1]["meal_type"] == "cena"
        assert "merienda nocturna / extra de la noche" not in res


def test_night_light_beverage_becomes_complemento_not_merienda():
    """Una infusión o té ligera (<= 50 kcal) de noche tras la cena se clasifica como complemento."""
    user_id = "user-test-bb033"

    mock_cena = {
        "id": "cena-01",
        "meal_type": "cena",
        "meal_name": "Pechuga con moro",
        "calories": 650
    }

    with patch("tools._hora_local_float", return_value=23.5), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._buscar_cena_principal_hoy", return_value=mock_cena), \
         patch("tools.db_log_consumed_meal", return_value="meal-comp-1") as mock_db_log:

        res = log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Té de manzanilla caliente",
            "calories": 2,
            "protein": 0,
            "carbs": 0,
            "healthy_fats": 0,
            "meal_type": "cena",
            "explicit_slot": False,
        })

        assert mock_db_log.call_args[1]["meal_type"] == "complemento"
        assert "merienda nocturna / extra de la noche" not in res


def test_night_heavy_intake_is_not_merienda():
    """Una ingesta pesada (> 350 kcal) tardía no se degrada a merienda nocturna."""
    user_id = "user-test-bb033"

    mock_cena = {
        "id": "cena-01",
        "meal_type": "cena",
        "meal_name": "Sopa ligera",
        "calories": 380
    }

    with patch("tools._hora_local_float", return_value=23.5), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._buscar_cena_principal_hoy", return_value=mock_cena), \
         patch("tools.db_log_consumed_meal", return_value="meal-heavy-1") as mock_db_log:

        res = log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Hamburguesa doble con papas",
            "calories": 750,
            "protein": 45,
            "carbs": 65,
            "healthy_fats": 35,
            "meal_type": "cena",
            "explicit_slot": False,
        })

        # No entra en merienda nocturna porque calories > 350
        assert mock_db_log.call_args[1]["meal_type"] == "cena"
        assert "merienda nocturna / extra de la noche" not in res


def test_vision_instructions_contain_night_snack_rule():
    """Las instrucciones del prompt de visión contemplan el caso de merienda nocturna."""
    v_dudas = {
        "kind": "multi",
        "items": [{"kind": "plato", "description": "Postre de noche"}],
        "respuestas": "Chocolate",
        "has_text": True
    }
    instr = instruccion(v_dudas)
    assert "merienda nocturna" in instr
    assert "extra de la noche" in instr
    assert "NUNCA repitas «Anoté como cena»" in instr

    v_solo = {"solo_foto": True}
    instr_solo = instruccion_solo_foto(v_solo)
    assert "merienda nocturna" in instr_solo
