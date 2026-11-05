"""
backend/tests/test_diary_chat_bridge.py
Pruebas unitarias para la sincronización entre el diario de comidas manual / escáner / plan y el Coach conversacional.
"""

from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
import pytest

from diary_chat_bridge import (
    build_sync_turn_texts,
    get_or_resolve_today_session_for_user,
    sync_meal_to_coach_chat,
)
from agent import _format_consumed_meal_line, _ultimo_id_registro_diario_previo
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def test_build_sync_turn_texts_hoy_con_metas_restantes():
    """Para una comida de hoy, el coach confirma macros y calcula lo que le queda al usuario."""
    user_id = "test-uid-1"
    consumed_today = [
        {"calories": 590, "protein": 45, "meal_name": "Mangú con pechuga"}
    ]
    profile = {
        "health_profile": {
            "target_calories": 2300,
            "target_protein": 134,
        }
    }

    with patch("diary_chat_bridge.get_consumed_meals_today", return_value=consumed_today), \
         patch("db.get_user_profile", return_value=profile):
        user_txt, coach_txt = build_sync_turn_texts(
            meal_name="Mangú con pechuga",
            meal_type="desayuno",
            calories=590,
            protein=45,
            days_ago=0,
            user_id=user_id,
            meal_id="uuid-meal-1",
        )

    assert user_txt == "Registré mi desayuno: Mangú con pechuga"
    assert "Anoté tu **desayuno**: Mangú con pechuga" in coach_txt
    assert "**590 kcal** y **45 g de proteína**" in coach_txt
    # 2300 - 590 = 1710 kcal, 134 - 45 = 89 g proteína
    assert "Te quedan **~1710 kcal** y **~89 g de proteína** para tu meta de hoy." in coach_txt


def test_build_sync_turn_texts_hoy_supera_meta():
    """Si el usuario ya superó su meta o no hay targets, reporta el acumulado de hoy."""
    user_id = "test-uid-2"
    consumed_today = [
        {"calories": 2500, "protein": 140, "meal_name": "Cena abundante"}
    ]
    profile = {
        "health_profile": {
            "target_calories": 2000,
            "target_protein": 120,
        }
    }

    with patch("diary_chat_bridge.get_consumed_meals_today", return_value=consumed_today), \
         patch("db.get_user_profile", return_value=profile):
        user_txt, coach_txt = build_sync_turn_texts(
            meal_name="Cena abundante",
            meal_type="cena",
            calories=2500,
            protein=140,
            days_ago=0,
            user_id=user_id,
            meal_id="uuid-meal-2",
        )

    assert user_txt == "Registré mi cena: Cena abundante"
    assert "Anoté tu **cena**: Cena abundante" in coach_txt
    assert "Hoy llevas **~2500 kcal** y **~140 g de proteína** en el diario." in coach_txt


def test_build_sync_turn_texts_ayer():
    """Para una comida de ayer (days_ago=1), aclara que no suma a las macros de hoy."""
    user_txt, coach_txt = build_sync_turn_texts(
        meal_name="4 tostadas integrales con queso cottage",
        meal_type="desayuno",
        calories=700,
        protein=58,
        days_ago=1,
        user_id="test-uid-3",
        meal_id="uuid-meal-3",
    )

    assert user_txt == "Registré mi desayuno de ayer: 4 tostadas integrales con queso cottage"
    assert "Anoté tu **desayuno de ayer**: 4 tostadas integrales con queso cottage" in coach_txt
    assert "**700 kcal** y **58 g de proteína**" in coach_txt
    assert "No suma a las macros de hoy." in coach_txt


def test_build_sync_turn_texts_dias_pasados():
    """Para comidas de hace N días (days_ago > 1), etiqueta la fecha y excluye macros de hoy."""
    user_txt, coach_txt = build_sync_turn_texts(
        meal_name="Sancocho criollo",
        meal_type="almuerzo",
        calories=850,
        protein=50,
        days_ago=3,
        user_id="test-uid-4",
        meal_id="uuid-meal-4",
    )

    assert user_txt == "Registré mi almuerzo de hace 3 días: Sancocho criollo"
    assert "Anoté tu **almuerzo de hace 3 días**: Sancocho criollo" in coach_txt
    assert "No suma a las macros de hoy." in coach_txt


def test_build_sync_turn_texts_suplemento():
    """Un suplemento indica que no sustituye una comida principal."""
    user_txt, coach_txt = build_sync_turn_texts(
        meal_name="1 scoop proteína whey",
        meal_type="suplemento",
        calories=120,
        protein=24,
        days_ago=0,
        user_id="test-uid-5",
        meal_id="uuid-meal-5",
    )

    assert user_txt == "Registré mi suplemento: 1 scoop proteína whey"
    assert "Anoté tu **suplemento**: 1 scoop proteína whey" in coach_txt
    assert "Recuerda que esto no sustituye una comida principal." in coach_txt


def test_get_or_resolve_today_session_for_user_existente_hoy():
    """Si la última sesión del usuario fue creada hoy, se reutiliza."""
    user_id = "test-uid-6"
    ahora_utc = datetime.now(timezone.utc)
    mock_row = {"id": "session-existente-hoy", "created_at": ahora_utc}

    with patch("diary_chat_bridge.execute_sql_query", return_value=mock_row):
        sid = get_or_resolve_today_session_for_user(user_id)
        assert sid == "session-existente-hoy"


def test_get_or_resolve_today_session_for_user_crea_nueva_si_ayer():
    """Si la última sesión es de días pasados, crea una sesión nueva."""
    user_id = "test-uid-7"
    hace_dos_dias = datetime.now(timezone.utc) - timedelta(days=2)
    mock_row = {"id": "session-vieja", "created_at": hace_dos_dias}

    with patch("diary_chat_bridge.execute_sql_query", return_value=mock_row), \
         patch("diary_chat_bridge.get_or_create_session", return_value={"id": "nueva-sesion-id"}):
        sid = get_or_resolve_today_session_for_user(user_id)
        assert sid != "session-vieja"
        assert len(sid) >= 30  # UUID generado


def test_sync_meal_to_coach_chat_omite_chat_y_guests():
    """Las comidas registradas desde el propio chat o de invitados no se duplican."""
    assert sync_meal_to_coach_chat(
        user_id="test-uid", meal_id="1", meal_name="A", source="chat"
    ) is None

    assert sync_meal_to_coach_chat(
        user_id="guest", meal_id="1", meal_name="A", source="manual"
    ) is None


def test_sync_meal_to_coach_chat_persiste_mensajes():
    """sync_meal_to_coach_chat guarda ambos mensajes en agent_messages con save_message."""
    user_id = "test-uid-8"
    saved_messages = []

    def mock_save_message(session_id, role, content, user_id=None, process_nudge=True):
        saved_messages.append({"session_id": session_id, "role": role, "content": content, "user_id": user_id})

    with patch("diary_chat_bridge.get_or_resolve_today_session_for_user", return_value="sid-1234"), \
         patch("diary_chat_bridge.save_message", side_effect=mock_save_message), \
         patch("diary_chat_bridge._try_update_langgraph_checkpoint"), \
         patch("diary_chat_bridge.get_consumed_meals_today", return_value=[]), \
         patch("db.get_user_profile", return_value={}):

        res = sync_meal_to_coach_chat(
            user_id=user_id,
            meal_id="meal-uuid-8",
            meal_name="Pechuga a la plancha con puré",
            meal_type="almuerzo",
            calories=650,
            protein=48,
            source="manual",
        )

    assert res is not None
    assert res["session_id"] == "sid-1234"
    assert len(saved_messages) == 2
    assert saved_messages[0]["role"] == "user"
    assert "Registré mi almuerzo: Pechuga a la plancha con puré" in saved_messages[0]["content"]
    assert saved_messages[1]["role"] == "model"
    assert "Anoté tu **almuerzo**: Pechuga a la plancha con puré" in saved_messages[1]["content"]


def test_format_consumed_meal_line():
    """_format_consumed_meal_line incluye nombre, calorías, id e ingredientes si existen."""
    m_con_ing = {
        "id": "abc-123",
        "meal_type": "almuerzo",
        "meal_name": "Pollo guisado con moro",
        "calories": 750,
        "ingredients": ["150g pechuga", "200g moro de habichuelas"],
    }
    line = _format_consumed_meal_line(m_con_ing)
    assert "almuerzo: Pollo guisado con moro (750 kcal) [id: abc-123]" in line
    assert "(ingredientes: 150g pechuga, 200g moro de habichuelas)" in line

    m_sin_ing = {
        "meal_name": "Manzana",
        "calories": 80,
    }
    line2 = _format_consumed_meal_line(m_sin_ing)
    assert line2 == "Manzana (80 kcal)"


def test_ultimo_id_registro_diario_previo_fallback():
    """_ultimo_id_registro_diario_previo encuentra el ID desde AIMessage o fallback de DB."""
    # 1. En ToolMessage tradicional previo al último turno del usuario
    msgs1 = [
        HumanMessage(content="comí pollo"),
        ToolMessage(content="registrado [ID_REGISTRO_DIARIO: c0ffee01-1234-5678-abcd-123456789001]", tool_call_id="1"),
        HumanMessage(content="cámbialo"),
    ]
    assert _ultimo_id_registro_diario_previo(msgs1) == "c0ffee01-1234-5678-abcd-123456789001"

    # 2. En AIMessage previo al último turno del usuario
    msgs2 = [
        HumanMessage(content="anota"),
        AIMessage(content="Anoté [ID_REGISTRO_DIARIO: c0ffee02-1234-5678-abcd-123456789002] tu plato"),
        HumanMessage(content="corrige"),
    ]
    assert _ultimo_id_registro_diario_previo(msgs2) == "c0ffee02-1234-5678-abcd-123456789002"

    # 3. Fallback a get_consumed_meals_today cuando no hay tag en mensajes
    msgs3 = [HumanMessage(content="cámbiale la pechuga por salmón")]
    today_meals = [{"id": "c0ffee03-1234-5678-abcd-123456789003", "created_at": "2026-11-05T10:00:00Z"}]
    with patch("db_facts.get_consumed_meals_today", return_value=today_meals):
        assert _ultimo_id_registro_diario_previo(msgs3, user_id="user-123") == "c0ffee03-1234-5678-abcd-123456789003"


def test_persist_consumed_meal_triggers_bridge_for_manual_not_chat():
    """_persist_consumed_meal invoca sync_meal_to_coach_chat cuando source es manual pero no cuando es chat."""
    from fastapi import BackgroundTasks
    from routers.diary import _persist_consumed_meal

    with patch("routers.diary.log_consumed_meal", return_value="c0ffee99-1234-5678-abcd-123456789099"), \
         patch("routers.diary.nevera_activa", return_value=False), \
         patch("routers.diary.trigger_incremental_learning"), \
         patch("diary_chat_bridge.sync_meal_to_coach_chat") as mock_bridge:

        # 1. Registro manual dispara bridge
        _persist_consumed_meal(
            user_id="user-xyz",
            meal_name="Mangú con salami",
            meal_type="desayuno",
            calories=600,
            protein=25,
            carbs=70,
            healthy_fats=15,
            ingredients=["plátano", "salami"],
            days_ago=0,
            background_tasks=BackgroundTasks(),
            source="manual",
            session_id="custom-session-id",
        )
        assert mock_bridge.called
        kwargs = mock_bridge.call_args.kwargs
        assert kwargs["meal_id"] == "c0ffee99-1234-5678-abcd-123456789099"
        assert kwargs["meal_name"] == "Mangú con salami"
        assert kwargs["calories"] == 600
        assert kwargs["session_id"] == "custom-session-id"

        # 2. Registro desde chat NO debe disparar bridge
        mock_bridge.reset_mock()
        _persist_consumed_meal(
            user_id="user-xyz",
            meal_name="Mangú con salami",
            meal_type="desayuno",
            calories=600,
            protein=25,
            carbs=70,
            healthy_fats=15,
            ingredients=["plátano", "salami"],
            days_ago=0,
            background_tasks=BackgroundTasks(),
            source="chat",
        )
        assert not mock_bridge.called

