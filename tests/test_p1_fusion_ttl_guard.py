"""[P1-FUSION-TTL-GUARD · 2026-11-03] Test suite para la ventana de expiración (TTL de 30 min)
en la autofusión silenciosa de adiciones (BB-005 vs BB-004).
Verifica que:
1. Adiciones implícitas ('Además un café', 'Extra atún') dentro de 30 min sí se fusionan.
2. Adiciones implícitas fuera de 30 min NO se fusionan y crean un registro independiente.
3. Anclajes explícitos ('al desayuno súmale...', 'se me quedó del almuerzo...') sí se fusionan sin límite de TTL.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from agent import (
    _intentar_fusionar_alimento_en_comida_existente,
    _intentar_fusionar_repeticion_en_fila_previa,
    _es_anclaje_explicito_de_slot,
    _comida_dentro_de_ttl_fusion,
)
import db_facts
import tools


def test_helper_es_anclaje_explicito():
    assert _es_anclaje_explicito_de_slot("Al desayuno súmale un café") is True
    assert _es_anclaje_explicito_de_slot("Se me quedó del almuerzo una galleta") is True
    assert _es_anclaje_explicito_de_slot("Agrégalo a la cena") is True
    assert _es_anclaje_explicito_de_slot("Fue parte del desayuno") is True
    assert _es_anclaje_explicito_de_slot("Lo comí en el almuerzo") is True

    # Adiciones implícitas no son anclaje explícito
    assert _es_anclaje_explicito_de_slot("Además un café") is False
    assert _es_anclaje_explicito_de_slot("Extra una lata de atún") is False
    assert _es_anclaje_explicito_de_slot("Y una manzana") is False
    assert _es_anclaje_explicito_de_slot("También dos galletas") is False
    assert _es_anclaje_explicito_de_slot("Un café con leche y una galleta") is False


def test_comida_dentro_de_ttl_evaluator():
    now = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)

    # Hace 5 minutos -> Dentro de TTL
    comida_5m = {"created_at": (now - timedelta(minutes=5)).isoformat()}
    assert _comida_dentro_de_ttl_fusion(comida_5m, ttl_min=30.0, ahora_utc=now) is True

    # Hace 29 minutos -> Dentro de TTL
    comida_29m = {"created_at": (now - timedelta(minutes=29)).isoformat()}
    assert _comida_dentro_de_ttl_fusion(comida_29m, ttl_min=30.0, ahora_utc=now) is True

    # Hace 31 minutos -> TTL expirado
    comida_31m = {"created_at": (now - timedelta(minutes=31)).isoformat()}
    assert _comida_dentro_de_ttl_fusion(comida_31m, ttl_min=30.0, ahora_utc=now) is False

    # Hace 3 horas -> TTL expirado
    comida_3h = {"created_at": (now - timedelta(hours=3)).isoformat()}
    assert _comida_dentro_de_ttl_fusion(comida_3h, ttl_min=30.0, ahora_utc=now) is False


def test_adicion_inmediata_dentro_de_ttl_se_fusiona(monkeypatch):
    """A los 5 minutos de desayunar, 'Además un café negro' se fusiona con el desayuno."""
    now = datetime.now(timezone.utc)
    mock_meals = [{
        "id": "meal-des-1",
        "meal_name": "Mangú con 2 huevos",
        "meal_type": "desayuno",
        "calories": 450,
        "protein": 20,
        "carbs": 50,
        "healthy_fats": 15,
        "ingredients": ["mangú (200 g)", "2 huevos"],
        "created_at": (now - timedelta(minutes=5)).isoformat(),
    }]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda uid: mock_meals)

    messages = [
        HumanMessage(content="Desayuné mangú con 2 huevos"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {"meal_type": "desayuno", "meal_name": "Mangú con 2 huevos"}, "id": "t1"}]),
        ToolMessage(content="Registrado como desayuno. [ID_REGISTRO_DIARIO: meal-des-1]", tool_call_id="t1"),
        AIMessage(content="Anoté tu desayuno..."),
        HumanMessage(content="Además un café negro"),
    ]
    tool_args = {"meal_name": "café negro", "calories": 5, "meal_type": "desayuno"}

    fused = _intentar_fusionar_alimento_en_comida_existente("u-test", tool_args, messages)
    assert fused is not None
    assert fused["meal_id"] == "meal-des-1"
    assert fused["calories"] == 455
    assert "café negro" in fused["meal_name"].lower() or "cafe negro" in fused["meal_name"].lower()


def test_adicion_fuera_de_ttl_sin_anclaje_no_se_fusiona(monkeypatch):
    """3 horas después de desayunar, 'Además un café con leche y una galleta' NO se fusiona; debe crear comida nueva."""
    now = datetime.now(timezone.utc)
    mock_meals = [{
        "id": "meal-des-1",
        "meal_name": "Mangú con 2 huevos",
        "meal_type": "desayuno",
        "calories": 450,
        "protein": 20,
        "carbs": 50,
        "healthy_fats": 15,
        "ingredients": ["mangú (200 g)", "2 huevos"],
        "created_at": (now - timedelta(hours=3, minutes=15)).isoformat(),  # 3h 15m atrás
    }]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda uid: mock_meals)

    messages = [
        HumanMessage(content="Desayuné mangú con 2 huevos"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {"meal_type": "desayuno", "meal_name": "Mangú con 2 huevos"}, "id": "t1"}]),
        ToolMessage(content="Registrado como desayuno. [ID_REGISTRO_DIARIO: meal-des-1]", tool_call_id="t1"),
        AIMessage(content="Anoté tu desayuno..."),
        HumanMessage(content="Además un café con leche y una galleta"),
    ]
    tool_args = {"meal_name": "café con leche y galleta", "calories": 180, "meal_type": "desayuno"}

    # Fusión rechazada por TTL
    fused = _intentar_fusionar_alimento_en_comida_existente("u-test", tool_args, messages)
    assert fused is None


def test_anclaje_explicito_fuera_de_ttl_si_se_fusiona(monkeypatch):
    """3 horas después, si el usuario dice 'Al desayuno súmale un café', SÍ se fusiona porque fue explícito."""
    now = datetime.now(timezone.utc)
    mock_meals = [{
        "id": "meal-des-1",
        "meal_name": "Mangú con 2 huevos",
        "meal_type": "desayuno",
        "calories": 450,
        "protein": 20,
        "carbs": 50,
        "healthy_fats": 15,
        "ingredients": ["mangú (200 g)", "2 huevos"],
        "created_at": (now - timedelta(hours=3, minutes=15)).isoformat(),
    }]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda uid: mock_meals)

    messages = [
        HumanMessage(content="Al desayuno súmale un café negro"),
    ]
    tool_args = {"meal_name": "café negro", "calories": 5, "meal_type": "desayuno"}

    fused = _intentar_fusionar_alimento_en_comida_existente("u-test", tool_args, messages)
    assert fused is not None
    assert fused["meal_id"] == "meal-des-1"
    assert fused["calories"] == 455


def test_repeticion_ttl_guard(monkeypatch):
    """Verifica que la repetición respeta el TTL."""
    now = datetime.now(timezone.utc)
    valid_uuid = "11112222-3333-4444-5555-666677778888"

    # 1. Dentro de TTL (10 min) -> Fusión autorizada
    fila_fresca = {
        "id": valid_uuid, "meal_name": "Arroz con pollo", "calories": 500, "protein": 35,
        "carbs": 60, "healthy_fats": 10, "meal_type": "almuerzo",
        "created_at": (now - timedelta(minutes=10)).isoformat(),
    }
    monkeypatch.setattr(tools, "_leer_fila_diario", lambda uid, mid: fila_fresca)
    msgs = [
        ToolMessage(content=f"Registrado. [ID_REGISTRO_DIARIO: {valid_uuid}]", tool_call_id="t1"),
        HumanMessage(content="Repetí el plato"),
    ]
    tool_args = {"meal_name": "Arroz con pollo", "calories": 500, "protein": 35, "carbs": 60, "healthy_fats": 10}
    res_fresco = _intentar_fusionar_repeticion_en_fila_previa("u-test", tool_args, msgs)
    assert res_fresco is not None
    assert res_fresco["calories"] == 1000

    # 2. Fuera de TTL (3 horas) -> Fusión rechazada
    fila_vieja = {
        "id": valid_uuid, "meal_name": "Arroz con pollo", "calories": 500, "protein": 35,
        "carbs": 60, "healthy_fats": 10, "meal_type": "almuerzo",
        "created_at": (now - timedelta(hours=3)).isoformat(),
    }
    monkeypatch.setattr(tools, "_leer_fila_diario", lambda uid, mid: fila_vieja)
    res_viejo = _intentar_fusionar_repeticion_en_fila_previa("u-test", tool_args, msgs)
    assert res_viejo is None
