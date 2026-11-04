"""[BB-032 / P1-BRUNCH-SATIETY-CLAIM-SUPPRESSION · 2026-11-04]
Tests para la supresión de reclamo de comidas sin registrar por brunch
o ventana de saciedad tras ingesta pesada.
"""

from datetime import datetime, timezone
from unittest.mock import patch
import pytest

from tools import _nota_comidas_sin_registrar, _row_hora_local_float


def test_row_hora_local_float_formats():
    """Verifica que _row_hora_local_float maneje float directo, strings ISO y datetime."""
    # 1. Float directo
    r1 = {"hora": 11.75, "calories": 850}
    assert _row_hora_local_float(r1, tz_offset_min=240) == 11.75

    # 2. String ISO UTC (15:45Z con offset 240min -> 11:45 local = 11.75)
    r2 = {"consumed_at": "2026-11-04T15:45:00Z", "calories": 850}
    assert pytest.approx(_row_hora_local_float(r2, tz_offset_min=240), rel=1e-3) == 11.75

    # 3. Datetime tz-aware
    r3 = {"consumed_at": datetime(2026, 11, 4, 15, 45, tzinfo=timezone.utc), "calories": 850}
    assert pytest.approx(_row_hora_local_float(r3, tz_offset_min=240), rel=1e-3) == 11.75

    # 4. Fallback None si no hay datos válidos
    assert _row_hora_local_float({}) is None
    assert _row_hora_local_float("no-dict") is None


def test_brunch_bridge_window_suppresses_lunch_claim():
    """Un desayuno pesado de 850 kcal a las 11:45 AM en fin de semana no debe provocar
    que a las 2:30 PM el coach reclame falsamente el almuerzo."""
    user_id = "test_user_brunch_1"
    rows = [
        {"meal_type": "desayuno", "calories": 850, "hora": 11.75, "meal_name": "Los tres golpes"}
    ]
    with patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._diary_date_str_for_user", return_value="2026-11-04"), \
         patch("tools.get_latest_usable_meal_plan", return_value=None), \
         patch("db_facts.get_consumed_meals_today", return_value=[]):
        nota = _nota_comidas_sin_registrar(user_id, days_ago=0, rows_extra=rows, ahora_local=14.5)

    # No debe reclamar almuerzo
    assert "almuerzo" not in nota.lower()
    assert "¿qué comiste en el almuerzo" not in nota
    assert "ayunas" not in nota
    # Debe reconocer que todo lo que tocaba hasta ahora está registrado y apuntar a cena
    assert "cena" in nota.lower()


def test_brunch_bridge_window_suppresses_breakfast_claim():
    """Si el plato pesado de 850 kcal entra a las 12:15 PM como 'almuerzo', el coach
    no debe reclamar falsamente el desayuno a las 2:30 PM."""
    user_id = "test_user_brunch_2"
    rows = [
        {"meal_type": "almuerzo", "calories": 850, "hora": 12.25, "meal_name": "Los tres golpes"}
    ]
    with patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._diary_date_str_for_user", return_value="2026-11-04"), \
         patch("tools.get_latest_usable_meal_plan", return_value=None), \
         patch("db_facts.get_consumed_meals_today", return_value=[]):
        nota = _nota_comidas_sin_registrar(user_id, days_ago=0, rows_extra=rows, ahora_local=14.5)

    # No debe reclamar desayuno ni ayuno
    assert "desayuno" not in nota.lower()
    assert "ayunas" not in nota
    assert "cena" in nota.lower()


def test_heavy_morning_meal_satiety_window_suppresses_lunch():
    """Un desayuno pesado (700 kcal) a las 10:45 AM suprime el reclamo de almuerzo a la 1:50 PM
    (han pasado 3.08h < 3.5h)."""
    user_id = "test_user_satiety_active"
    rows = [
        {"meal_type": "desayuno", "calories": 700, "hora": 10.75, "meal_name": "Huevos, tocineta y mofongo"}
    ]
    with patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._diary_date_str_for_user", return_value="2026-11-04"), \
         patch("tools.get_latest_usable_meal_plan", return_value=None), \
         patch("db_facts.get_consumed_meals_today", return_value=[]):
        nota = _nota_comidas_sin_registrar(user_id, days_ago=0, rows_extra=rows, ahora_local=13.83)

    assert "almuerzo" not in nota.lower()
    assert "cena" in nota.lower()


def test_heavy_morning_meal_satiety_window_expires_after_3_5_hours():
    """Tras 4.25h de un desayuno de 700 kcal (a las 3:00 PM), la ventana de saciedad ha expirado
    y el coach sí puede reclamar el almuerzo si no ha sido registrado."""
    user_id = "test_user_satiety_expired"
    rows = [
        {"meal_type": "desayuno", "calories": 700, "hora": 10.75, "meal_name": "Huevos, tocineta y mofongo"}
    ]
    with patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._diary_date_str_for_user", return_value="2026-11-04"), \
         patch("tools.get_latest_usable_meal_plan", return_value=None), \
         patch("db_facts.get_consumed_meals_today", return_value=[]):
        nota = _nota_comidas_sin_registrar(user_id, days_ago=0, rows_extra=rows, ahora_local=15.0)

    # Ahora sí reclama almuerzo
    assert "almuerzo" in nota.lower()


def test_light_breakfast_claims_lunch_normally():
    """Un desayuno ligero habitual (300 kcal a las 9:00 AM) no suprime el almuerzo a las 2:00 PM."""
    user_id = "test_user_light_breakfast"
    rows = [
        {"meal_type": "desayuno", "calories": 300, "hora": 9.0, "meal_name": "Avena ligera con manzana"}
    ]
    with patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._diary_date_str_for_user", return_value="2026-11-04"), \
         patch("tools.get_latest_usable_meal_plan", return_value=None), \
         patch("db_facts.get_consumed_meals_today", return_value=[]):
        nota = _nota_comidas_sin_registrar(user_id, days_ago=0, rows_extra=rows, ahora_local=14.0)

    assert "almuerzo" in nota.lower()


def test_named_brunch_suppresses_both_slots():
    """Un plato nombrado explícitamente 'Brunch' con 550 kcal a las 11:30 AM cubre
    implícitamente desayuno y almuerzo."""
    user_id = "test_user_named_brunch"
    rows = [
        {"meal_type": "desayuno", "calories": 550, "hora": 11.5, "meal_name": "Brunch dominicano ligero"}
    ]
    with patch("tools.user_tz_offset_min", return_value=240), \
         patch("tools._diary_date_str_for_user", return_value="2026-11-04"), \
         patch("tools.get_latest_usable_meal_plan", return_value=None), \
         patch("db_facts.get_consumed_meals_today", return_value=[]):
        nota = _nota_comidas_sin_registrar(user_id, days_ago=0, rows_extra=rows, ahora_local=14.5)

    assert "almuerzo" not in nota.lower()
    assert "desayuno" not in nota.lower()
    assert "cena" in nota.lower()
