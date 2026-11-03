# -*- coding: utf-8 -*-
"""[BB-022 / P1-CLONE-HABITUAL-DECONTAMINATE · 2026-11-03]
Pruebas de descontaminación de platos habituales al clonar del historial (BB-014 + BB-005).

Verifica que:
1. Al fusionar un extra inmediato (BB-005), se almacena en `nutrition` el plato base inmutable (`base_meal`)
   y la lista de extras (`addons`).
2. Múltiples adiciones consecutivas preservan la primera comida como base inmutable.
3. `build_past_diary_block` formatea el desglose base vs extras fusionados en el historial de días pasados.
4. El helper `_extraer_plato_base_de_comida` aísla los datos limpios del plato base.
5. El endpoint `/api/diary/consumed/repeat` con `base_only=True` persiste exclusivamente el plato base.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta, date
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import pytest

from agent import (
    _intentar_fusionar_alimento_en_comida_existente,
    _intentar_fusionar_repeticion_en_fila_previa,
    _extraer_plato_base_de_comida,
)
from chat_history_context import build_past_diary_block
from prompts.chat_agent import _REGLA_UNA_LLAMADA_POR_COMIDA
import db_facts
import tools


def test_fusion_guarda_base_meal_y_addons_en_nutrition(monkeypatch):
    """Verifica que una fusión de un extra (ej. café + postre) conserva el plato base original en nutrition."""
    now = datetime.now(timezone.utc)
    mock_meals = [{
        "id": "meal-lunch-101",
        "meal_name": "Arroz, habichuelas y pollo guisado",
        "meal_type": "almuerzo",
        "calories": 680,
        "protein": 45,
        "carbs": 70,
        "healthy_fats": 12,
        "ingredients": ["arroz (150 g)", "habichuelas (100 g)", "pollo guisado (150 g)"],
        "nutrition": {},
        "created_at": (now - timedelta(minutes=5)).isoformat(),
    }]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda uid: mock_meals)

    messages = [
        HumanMessage(content="Almorcé arroz, habichuelas y pollo guisado"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {"meal_type": "almuerzo", "meal_name": "Arroz, habichuelas y pollo guisado"}, "id": "t1"}]),
        ToolMessage(content="¡Éxito! Se ha registrado el consumo como almuerzo en tu diario. [ID_REGISTRO_DIARIO: meal-lunch-101]", tool_call_id="t1"),
        AIMessage(content="Anoté tu almuerzo..."),
        HumanMessage(content="Además un café con leche y un postre"),
    ]
    tool_args = {
        "meal_name": "café con leche y postre",
        "calories": 220,
        "protein": 3,
        "carbs": 25,
        "healthy_fats": 10,
        "ingredients": ["café con leche", "postre"],
        "meal_type": "almuerzo",
    }

    fused = _intentar_fusionar_alimento_en_comida_existente("u-test", tool_args, messages)
    assert fused is not None
    assert fused["calories"] == 900
    assert fused["protein"] == 48

    nut = fused.get("nutrition") or {}
    assert "base_meal" in nut
    base = nut["base_meal"]
    assert base["meal_name"] == "Arroz, habichuelas y pollo guisado"
    assert base["calories"] == 680
    assert base["protein"] == 45
    assert "arroz (150 g)" in base["ingredients"]

    assert "addons" in nut
    assert len(nut["addons"]) == 1
    addon = nut["addons"][0]
    assert addon["meal_name"] == "café con leche y postre"
    assert addon["calories"] == 220

    assert "addon_ingredients" in nut
    assert "café con leche" in nut["addon_ingredients"]
    assert "postre" in nut["addon_ingredients"]


def test_fusiones_multiples_preservan_base_meal_inmutable(monkeypatch):
    """Si se realizan dos fusiones sucesivas, base_meal sigue siendo el plato original."""
    now = datetime.now(timezone.utc)
    # Comida ya con una primera fusión (café +50 kcal)
    mock_meals = [{
        "id": "meal-lunch-102",
        "meal_name": "Arroz, habichuelas y pollo guisado con café",
        "meal_type": "almuerzo",
        "calories": 730,
        "protein": 46,
        "carbs": 72,
        "healthy_fats": 12,
        "ingredients": ["arroz (150 g)", "habichuelas (100 g)", "pollo guisado (150 g)", "café"],
        "nutrition": {
            "base_meal": {
                "meal_name": "Arroz, habichuelas y pollo guisado",
                "calories": 680,
                "protein": 45,
                "carbs": 70,
                "healthy_fats": 12,
                "ingredients": ["arroz (150 g)", "habichuelas (100 g)", "pollo guisado (150 g)"],
            },
            "addons": [{
                "meal_name": "café",
                "calories": 50,
                "protein": 1,
                "carbs": 2,
                "healthy_fats": 0,
                "ingredients": ["café"],
            }],
        },
        "created_at": (now - timedelta(minutes=10)).isoformat(),
    }]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda uid: mock_meals)

    messages = [
        HumanMessage(content="Almorcé arroz, habichuelas y pollo guisado"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {"meal_type": "almuerzo", "meal_name": "Arroz, habichuelas y pollo guisado con café"}, "id": "t1"}]),
        ToolMessage(content="¡Éxito! Se ha registrado el consumo como almuerzo en tu diario. [ID_REGISTRO_DIARIO: meal-lunch-102]", tool_call_id="t1"),
        AIMessage(content="Anoté tu almuerzo..."),
        HumanMessage(content="Y de extra un postre de chocolate"),
    ]
    tool_args = {
        "meal_name": "postre de chocolate",
        "calories": 170,
        "protein": 2,
        "carbs": 23,
        "healthy_fats": 10,
        "ingredients": ["postre de chocolate"],
        "meal_type": "almuerzo",
    }

    fused = _intentar_fusionar_alimento_en_comida_existente("u-test", tool_args, messages)
    assert fused is not None
    assert fused["calories"] == 900  # 730 + 170

    nut = fused.get("nutrition") or {}
    base = nut.get("base_meal") or {}
    assert base["calories"] == 680  # Sigue siendo 680, NO 730
    assert base["meal_name"] == "Arroz, habichuelas y pollo guisado"

    addons = nut.get("addons") or []
    assert len(addons) == 2
    assert addons[0]["meal_name"] == "café"
    assert addons[1]["meal_name"] == "postre de chocolate"


def test_build_past_diary_block_desglosa_base_y_extras():
    """El bloque de días anteriores muestra claramente la base vs extras fusionados."""
    hoy = date(2026, 11, 3)
    yesterday_iso = (datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc)).isoformat()
    rows = [{
        "meal_name": "Arroz, habichuelas y pollo guisado con café con leche y postre",
        "meal_type": "almuerzo",
        "calories": 900,
        "protein": 48,
        "consumed_at": yesterday_iso,
        "nutrition": {
            "base_meal": {
                "meal_name": "Arroz, habichuelas y pollo guisado",
                "calories": 680,
                "protein": 45,
            },
            "addons": [{
                "meal_name": "café con leche y postre",
                "calories": 220,
            }],
        },
    }]

    block = build_past_diary_block(rows, hoy, days_back=2)
    assert "Arroz, habichuelas y pollo guisado" in block
    assert "680 kcal base, 45g prot" in block
    assert "[+ extra fusionado: café con leche y postre (+220 kcal)]" in block


def test_extraer_plato_base_helper():
    """Prueba el helper _extraer_plato_base_de_comida."""
    comida_con_base = {
        "meal_name": "Arroz con habichuelas y pollo con postre",
        "calories": 900,
        "protein": 48,
        "nutrition": {
            "base_meal": {
                "meal_name": "Arroz con habichuelas y pollo",
                "calories": 680,
                "protein": 45,
                "carbs": 70,
                "healthy_fats": 12,
                "ingredients": ["arroz", "habichuelas", "pollo"],
            }
        }
    }
    limpio = _extraer_plato_base_de_comida(comida_con_base)
    assert limpio["is_decontaminated"] is True
    assert limpio["meal_name"] == "Arroz con habichuelas y pollo"
    assert limpio["calories"] == 680
    assert limpio["protein"] == 45

    comida_simple = {
        "meal_name": "Mangú con 2 huevos",
        "calories": 450,
        "protein": 20,
        "carbs": 50,
        "healthy_fats": 15,
        "ingredients": ["mangú", "2 huevos"],
    }
    directo = _extraer_plato_base_de_comida(comida_simple)
    assert directo["is_decontaminated"] is False
    assert directo["meal_name"] == "Mangú con 2 huevos"
    assert directo["calories"] == 450


def test_repeat_endpoint_base_only_clona_base_descontaminada(monkeypatch):
    """En routers/diary.py, api_repeat_consumed_meal clona solo la base cuando base_only=True."""
    from routers.diary import api_repeat_consumed_meal, RepeatMealRequest

    captured_kwargs = {}

    def mock_persist(**kwargs):
        captured_kwargs.update(kwargs)
        return {"success": True, "meal_id": "new-meal-id"}

    monkeypatch.setattr("routers.diary._persist_consumed_meal", mock_persist)

    row_data = {
        "meal_name": "Arroz, habichuelas y pollo guisado con café y postre",
        "meal_type": "almuerzo",
        "calories": 900,
        "protein": 48,
        "carbs": 95,
        "healthy_fats": 22,
        "ingredients": ["arroz", "habichuelas", "pollo", "café", "postre"],
        "nutrition": {
            "base_meal": {
                "meal_name": "Arroz, habichuelas y pollo guisado",
                "calories": 680,
                "protein": 45,
                "carbs": 70,
                "healthy_fats": 12,
                "ingredients": ["arroz", "habichuelas", "pollo"],
            },
            "addons": [{"meal_name": "café y postre", "calories": 220}],
        },
    }

    monkeypatch.setattr("routers.diary.execute_sql_query", lambda q, p, **kw: row_data)

    # 1. Con base_only=True (default) -> clona 680 kcal
    req_base = RepeatMealRequest(source_meal_id="11112222-3333-4444-5555-666677778888", base_only=True)
    api_repeat_consumed_meal(req_base, verified_user_id="user-123")
    assert captured_kwargs["meal_name"] == "Arroz, habichuelas y pollo guisado"
    assert captured_kwargs["calories"] == 680
    assert captured_kwargs["protein"] == 45
    assert "café" not in captured_kwargs["ingredients"]

    # 2. Con base_only=False -> clona 900 kcal completo
    captured_kwargs.clear()
    req_full = RepeatMealRequest(source_meal_id="11112222-3333-4444-5555-666677778888", base_only=False)
    api_repeat_consumed_meal(req_full, verified_user_id="user-123")
    assert captured_kwargs["meal_name"] == "Arroz, habichuelas y pollo guisado con café y postre"
    assert captured_kwargs["calories"] == 900
    assert captured_kwargs["protein"] == 48


def test_prompt_directiva_decontaminacion_presente():
    """Verifica que la directiva de descontaminación existe en los prompts del sistema."""
    assert "[P1-CLONE-HABITUAL-DECONTAMINATE · 2026-11-03]" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "DESCONTAMINACIÓN AL REPETIR/CLONAR" in _REGLA_UNA_LLAMADA_POR_COMIDA
