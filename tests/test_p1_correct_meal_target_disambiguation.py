# backend/tests/test_p1_correct_meal_target_disambiguation.py
# [BB-031 / P1-CORRECT-MEAL-TARGET-DISAMBIGUATION · 2026-11-04]
# Tests para desambiguación de target en correct_consumed_meal cuando hay múltiples
# comidas en el mismo slot (efecto BB-004) mediante scoring de intersección léxica.
from __future__ import annotations

import sys
from pathlib import Path
import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import tools
import db_facts


@pytest.fixture
def sample_multiple_meals(monkeypatch):
    """Fixture con 2 meriendas y 2 cenas en el mismo día."""
    meals = [
        # Merienda más reciente (18:15)
        {
            "id": "uuid-merienda-1815",
            "meal_name": "Pan con queso",
            "meal_type": "merienda",
            "ingredients": ["pan tostado", "queso gouda"],
            "calories": 280,
            "protein": 12,
            "carbs": 30,
            "healthy_fats": 10,
            "consumed_at": "2026-11-04T18:15:00+00:00",
            "created_at": "2026-11-04T18:16:00+00:00",
        },
        # Merienda anterior (15:30)
        {
            "id": "uuid-merienda-1530",
            "meal_name": "Frutas",
            "meal_type": "merienda",
            "ingredients": ["1 rueda de piña fresca", "agua"],
            "calories": 45,
            "protein": 0,
            "carbs": 12,
            "healthy_fats": 0,
            "consumed_at": "2026-11-04T15:30:00+00:00",
            "created_at": "2026-11-04T15:31:00+00:00",
        },
        # Cena más reciente (22:00)
        {
            "id": "uuid-cena-2200",
            "meal_name": "Cena",
            "meal_type": "cena",
            "ingredients": ["rebanada de pizza de queso"],
            "calories": 320,
            "protein": 12,
            "carbs": 35,
            "healthy_fats": 14,
            "consumed_at": "2026-11-04T22:00:00+00:00",
            "created_at": "2026-11-04T22:01:00+00:00",
        },
        # Cena anterior (19:00)
        {
            "id": "uuid-cena-1900",
            "meal_name": "Cena ligera",
            "meal_type": "cena",
            "ingredients": ["ensalada verde con tomate", "lata de atún al agua"],
            "calories": 220,
            "protein": 28,
            "carbs": 5,
            "healthy_fats": 6,
            "consumed_at": "2026-11-04T19:00:00+00:00",
            "created_at": "2026-11-04T19:01:00+00:00",
        },
    ]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda uid, *args, **kwargs: list(meals))
    return meals


def test_desambiguacion_dos_meriendas_por_ingredientes(sample_multiple_meals):
    """[BB-031] Caso emblemático: usuario dice 'fueron 2 ruedas de piña en la merienda'.
    El resolver debe elegir la merienda de frutas (15:30) y NO el pan con queso (18:15)."""
    mid, fila = tools._resolver_fila_diario_flexible(
        user_id="user-1",
        meal_type="merienda",
        ingredients=["2 ruedas de piña"],
    )
    assert mid == "uuid-merienda-1530"
    assert fila["meal_name"] == "Frutas"
    assert "piña" in str(fila["ingredients"]).lower()


def test_desambiguacion_dos_cenas_por_nombre(sample_multiple_meals):
    """[BB-031] Con 2 cenas, corregir mencionando 'atún' debe elegir la cena de las 19:00, no la pizza."""
    mid, fila = tools._resolver_fila_diario_flexible(
        user_id="user-1",
        meal_type="cena",
        meal_name="atún",
    )
    assert mid == "uuid-cena-1900"
    assert "atún" in str(fila["ingredients"]).lower()


def test_fallback_a_mas_reciente_si_no_hay_tokens(sample_multiple_meals):
    """[BB-031] Si no se especifican ingredientes ni nombre, mantiene el fallback seguro
    a la comida más reciente del slot."""
    mid, fila = tools._resolver_fila_diario_flexible(
        user_id="user-1",
        meal_type="merienda",
    )
    assert mid == "uuid-merienda-1815"
    assert fila["meal_name"] == "Pan con queso"


def test_busqueda_general_encuentra_por_ingredientes_sin_slot(sample_multiple_meals):
    """[BB-031] Si no se pasa meal_type, la búsqueda por tokens encuentra la comida
    gracias a la inspección de ingredients."""
    mid, fila = tools._resolver_fila_diario_flexible(
        user_id="user-1",
        ingredients=["piña"],
    )
    assert mid == "uuid-merienda-1530"


def test_correct_consumed_meal_delete_desambigua_target(sample_multiple_meals, monkeypatch):
    """[BB-031] Borrar una merienda especificando el ingrediente borra la merienda correcta."""
    deleted_ids = []
    monkeypatch.setattr(tools, "db_delete_consumed_meal", lambda uid, mid: deleted_ids.append((uid, mid)) or True)

    res = tools.correct_consumed_meal.func(
        user_id="user-1",
        meal_type="merienda",
        meal_name="piña",
        delete=True,
    )
    assert "¡Eliminado!" in res
    assert deleted_ids == [("user-1", "uuid-merienda-1530")]


def test_correct_consumed_meal_actualiza_target_desambiguado(sample_multiple_meals, monkeypatch):
    """[BB-031] Ejecución end-to-end de update: correct_consumed_meal actualiza la fila
    correcta de piña (15:30) y no la de pan con queso (18:15)."""
    updated_calls = []

    def fake_update(user_id, meal_id, **kwargs):
        updated_calls.append((user_id, meal_id, kwargs))
        return True

    monkeypatch.setattr(tools, "db_update_consumed_meal", fake_update)

    res = tools.correct_consumed_meal.func(
        user_id="user-1",
        meal_type="merienda",
        meal_name="2 ruedas de piña",
        ingredients=["2 ruedas de piña"],
        calories=90,
        protein=1,
    )
    assert "¡Corregido!" in res
    assert len(updated_calls) == 1
    uid, mid, kwargs = updated_calls[0]
    assert mid == "uuid-merienda-1530"
    assert kwargs.get("calories") == 90
