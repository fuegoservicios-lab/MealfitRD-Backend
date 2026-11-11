"""[BB-054 / STALE-EXTRA-DESACOPLED · 2026-11-11]
Test suite para el desacople temporal de adiciones implícitas ('extra ...', 'además ...') post-TTL.

Verifica que:
1. `_es_adicion_implicita_sin_slot` detecta adiciones implícitas y descarta anclajes explícitos o correcciones de ingredientes.
2. `_resolver_slot_post_ttl` asigna 'merienda', 'almuerzo', 'cena' o 'snack' según la hora local y las calorías.
3. En `log_consumed_meal`: si entra un alimento con `meal_type='desayuno'` al mediodía/tarde (>= 12:00 PM)
   y ya hay un desayuno registrado hoy, `_rescue_stale_desayuno_after_noon` lo reclasifica a 'merienda' (o 'almuerzo')
   e inyecta la directiva para que el asistente no diga que lo sumó al desayuno.
4. Caso exacto Moisés (10 nov, 01:32 p. m.): `extra yogurt ratio protein...` no se suma al desayuno de las 10:54 a. m.,
   sino que se anota como merienda independiente.
5. Si el LLM llama a `correct_consumed_meal` sobre una comida de hace horas ante un `extra ...`,
   se redirige a `log_consumed_meal` con el delta de macros y el nuevo slot.
6. Adiciones inmediatas dentro del TTL (<= 30 min) y anclajes explícitos ('al desayuno súmale...')
   mantienen su comportamiento legítimo de fusión sin regresiones.
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
    _es_anclaje_explicito_de_slot,
    _es_adicion_implicita_sin_slot,
    _comida_dentro_de_ttl_fusion,
    _intentar_fusionar_alimento_en_comida_existente,
    _ajustar_args_fusion_correct_consumed_meal,
    execute_tools,
)
import db_facts
import tools


def test_helper_es_adicion_implicita_sin_slot():
    """Valida la detección precisa de adiciones implícitas vs anclajes o aclaraciones."""
    # Adiciones implícitas (deben dar True)
    assert _es_adicion_implicita_sin_slot("extra yogurt ratio protein de strawberry cheese cake") is True
    assert _es_adicion_implicita_sin_slot("Extra lata de atun genova") is True
    assert _es_adicion_implicita_sin_slot("de extra un guineo") is True
    assert _es_adicion_implicita_sin_slot("Además un café negro") is True
    assert _es_adicion_implicita_sin_slot("También dos galletas oreo") is True
    assert _es_adicion_implicita_sin_slot("Y una manzana") is True
    assert _es_adicion_implicita_sin_slot("se me quedó una tostada") is True
    assert _es_adicion_implicita_sin_slot("olvidé poner una cucharada de miel") is True
    assert _es_adicion_implicita_sin_slot("faltó el aguacate") is True

    # Anclajes explícitos de slot (deben dar False porque sí tienen slot explícito)
    assert _es_adicion_implicita_sin_slot("Al desayuno súmale un café") is False
    assert _es_adicion_implicita_sin_slot("En el almuerzo también comí aguacate") is False
    assert _es_adicion_implicita_sin_slot("Fue parte del desayuno") is False
    assert _es_adicion_implicita_sin_slot("Agrégalo a la cena") is False
    assert _es_adicion_implicita_sin_slot("Se me quedó del almuerzo una galleta") is False

    # Correcciones de ingredientes o preguntas (deben dar False)
    assert _es_adicion_implicita_sin_slot("El pan integral tiene mantequilla de mani y queso cottage") is False
    assert _es_adicion_implicita_sin_slot("Eran 2 huevos, no 4") is False
    assert _es_adicion_implicita_sin_slot("¿Y con qué lo puedo acompañar?") is False


def test_resolver_slot_post_ttl(monkeypatch):
    """Valida la asignación adecuada del slot post-TTL según la hora local y calorías."""
    uid = "test-u1"

    # 1. Caso Moisés a la 1:32 p. m. (13.53h) con yogurt de 170 kcal -> 'merienda'
    assert tools._resolver_slot_post_ttl(uid, calories=170, hora_local=13.53) == "merienda"

    # 2. A la 1:32 p. m. (13.53h) con comida sustancial (600 kcal) y sin almuerzo previo -> 'almuerzo'
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda u: 240)
    monkeypatch.setattr("db.execute_sql_query", lambda *args, **kwargs: {"total": 0})
    assert tools._resolver_slot_post_ttl(uid, calories=600, hora_local=13.53) == "almuerzo"

    # 3. A la 1:32 p. m. (13.53h) con 600 kcal pero YA tiene almuerzo registrado -> 'merienda'
    monkeypatch.setattr("db.execute_sql_query", lambda *args, **kwargs: {"total": 1})
    assert tools._resolver_slot_post_ttl(uid, calories=600, hora_local=13.53) == "merienda"

    # 4. A las 4:30 p. m. (16.5h) con 250 kcal -> 'merienda'
    assert tools._resolver_slot_post_ttl(uid, calories=250, hora_local=16.5) == "merienda"

    # 5. A las 8:30 p. m. (20.5h) con 80 kcal -> 'merienda'
    assert tools._resolver_slot_post_ttl(uid, calories=80, hora_local=20.5) == "merienda"

    # 6. A las 11:30 p. m. (23.5h) con 82 kcal -> 'snack'
    assert tools._resolver_slot_post_ttl(uid, calories=82, hora_local=23.5) == "snack"


def test_rescue_stale_desayuno_after_noon(monkeypatch):
    """Valida que _rescue_stale_desayuno_after_noon rescata desayunos tardíos al mediodía."""
    uid = "test-u2"

    # Caso A: Son las 10:30 a. m. (hora < 12.0) -> No se reclasifica (sigue desayuno)
    monkeypatch.setattr(tools, "_hora_local_float", lambda u: 10.5)
    slot, rescatado = tools._rescue_stale_desayuno_after_noon(uid, "desayuno", calories=170, days_ago=0)
    assert slot == "desayuno"
    assert rescatado is False

    # Caso B: Son las 01:32 p. m. (13.53h) y ya hay 1 desayuno hoy -> reclasifica a 'merienda'
    monkeypatch.setattr(tools, "_hora_local_float", lambda u: 13.53)
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda u: 240)
    monkeypatch.setattr("db.execute_sql_query", lambda sql, *args, **kwargs: {"total": 1} if "desayuno" in sql else {"total": 0})

    slot, rescatado = tools._rescue_stale_desayuno_after_noon(uid, "desayuno", calories=170, days_ago=0)
    assert slot == "merienda"
    assert rescatado is True

    # Caso C: Son las 01:32 p. m. y entra plato fuerte (650 kcal) con desayuno previo pero sin almuerzo -> 'almuerzo'
    slot, rescatado = tools._rescue_stale_desayuno_after_noon(uid, "desayuno", calories=650, days_ago=0)
    assert slot == "almuerzo"
    assert rescatado is True


def test_case_moises_log_consumed_meal_desacopla_extra_yogurt(monkeypatch):
    """Simula el flujo completo de Moisés en execute_tools con log_consumed_meal.
    Desayuno registrado a las 10:54 a. m.; a la 1:32 p. m. el usuario dice:
    'extra yogurt ratio protein de strawberry cheese cake' y el LLM pasa meal_type='desayuno'.
    Verifica que execute_tools reclasifica a 'merienda'."""
    uid = "moises-test-uid"
    now = datetime(2026, 11, 10, 17, 32, 34, tzinfo=timezone.utc)  # 01:32:34 p. m. RD (UTC-4)

    # Desayuno creado a las 10:54 a. m. RD = 14:54 UTC (~2h 38m antes)
    ts_desayuno = (now - timedelta(hours=2, minutes=38)).isoformat()
    mock_meals = [{
        "id": "desayuno-moises-id",
        "meal_name": "4 huevos enteros con pan integral tostado, jamón cocido y aguacate",
        "meal_type": "desayuno",
        "calories": 768,
        "protein": 52,
        "carbs": 40,
        "healthy_fats": 35,
        "created_at": ts_desayuno,
    }]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda u, **kw: mock_meals)
    monkeypatch.setattr(tools, "_hora_local_float", lambda u: 13.53)
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda u: 240)
    monkeypatch.setattr("db.execute_sql_query", lambda sql, *args, **kwargs: {"total": 1} if "desayuno" in sql else {"total": 0})

    # Mock db_log_consumed_meal
    captured_logs = []
    def mock_db_log(user_id, meal_name, calories, protein, carbs, healthy_fats, ingredients=None, meal_type=None, **kwargs):
        captured_logs.append({
            "meal_name": meal_name,
            "calories": calories,
            "protein": protein,
            "meal_type": meal_type,
        })
        return "new-yogurt-meal-id"
    monkeypatch.setattr(tools, "db_log_consumed_meal", mock_db_log)
    monkeypatch.setattr(tools, "nevera_activa", lambda u: False)

    messages = [
        HumanMessage(content="extra yogurt ratio protein de strawberry cheese cake"),
        AIMessage(content="", tool_calls=[{
            "name": "log_consumed_meal",
            "args": {
                "meal_name": "Yogurt Ratio Protein sabor strawberry cheesecake",
                "calories": 170,
                "protein": 25,
                "carbs": 6,
                "healthy_fats": 4,
                "meal_type": "desayuno",
            },
            "id": "tc-yogurt-1",
        }]),
    ]

    state = {
        "user_id": uid,
        "messages": messages,
        "turn_plate_photos": [],
        "tz_offset": 240,
    }

    result_state = execute_tools(state)
    tool_msgs = [m for m in result_state.get("messages", []) if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1
    content = str(tool_msgs[0].content)

    # 1. Se guardó como 'merienda' en la base de datos
    assert len(captured_logs) == 1
    assert captured_logs[0]["meal_type"] == "merienda"
    assert captured_logs[0]["calories"] == 170
    assert captured_logs[0]["protein"] == 25

    # 2. El aviso interno prohibe explícitamente decir que se sumó al desayuno
    assert "como merienda" in content
    assert "NUNCA digas que lo sumaste o agregaste al desayuno" in content


def test_case_moises_correct_consumed_meal_desacopla_extra_yogurt(monkeypatch):
    """Si el LLM hubiese emitido correct_consumed_meal sobre el desayuno de las 10:54 a. m.
    ante 'extra yogurt ratio protein...', execute_tools debe interceptarlo,
    evitar la sobreescritura del desayuno viejo y redirigir a log_consumed_meal con el delta."""
    uid = "moises-test-uid"
    now = datetime(2026, 11, 10, 17, 32, 34, tzinfo=timezone.utc)
    ts_desayuno = (now - timedelta(hours=2, minutes=38)).isoformat()

    mock_target = {
        "id": "desayuno-moises-id",
        "meal_name": "4 huevos enteros hervidos con pan integral",
        "meal_type": "desayuno",
        "calories": 768,
        "protein": 52,
        "carbs": 40,
        "healthy_fats": 35,
        "created_at": ts_desayuno,
    }
    monkeypatch.setattr(tools, "_leer_fila_diario", lambda u, mid: mock_target)
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda u, **kw: [mock_target])
    monkeypatch.setattr(tools, "_hora_local_float", lambda u: 13.53)
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda u: 240)
    monkeypatch.setattr("db.execute_sql_query", lambda sql, *args, **kwargs: {"total": 1} if "desayuno" in sql else {"total": 0})

    captured_logs = []
    def mock_db_log(user_id, meal_name, calories, protein, carbs, healthy_fats, ingredients=None, meal_type=None, **kwargs):
        captured_logs.append({
            "meal_name": meal_name,
            "calories": calories,
            "protein": protein,
            "meal_type": meal_type,
        })
        return "new-yogurt-meal-id"
    monkeypatch.setattr(tools, "db_log_consumed_meal", mock_db_log)
    monkeypatch.setattr(tools, "nevera_activa", lambda u: False)

    messages = [
        HumanMessage(content="extra yogurt ratio protein de strawberry cheese cake"),
        AIMessage(content="", tool_calls=[{
            "name": "correct_consumed_meal",
            "args": {
                "meal_id": "desayuno-moises-id",
                "meal_name": "4 huevos enteros con pan integral y yogurt ratio protein",
                "calories": 938,   # 768 + 170
                "protein": 77,     # 52 + 25
                "carbs": 46,
                "healthy_fats": 39,
                "meal_type": "desayuno",
            },
            "id": "tc-yogurt-correct-1",
        }]),
    ]

    state = {
        "user_id": uid,
        "messages": messages,
        "turn_plate_photos": [],
        "tz_offset": 240,
    }

    result_state = execute_tools(state)
    tool_msgs = [m for m in result_state.get("messages", []) if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1

    # Verificamos que se ejecutó log_consumed_meal con el delta (170 kcal, 25 g P) y como 'merienda'
    assert len(captured_logs) == 1
    assert captured_logs[0]["meal_type"] == "merienda"
    assert captured_logs[0]["calories"] == 170
    assert captured_logs[0]["protein"] == 25


def test_adicion_inmediata_dentro_de_ttl_sigue_fusionandose(monkeypatch):
    """Aclaración inmediata al minuto (10:55 a. m. tras desayuno de 10:54 a. m.):
    'extra mantequilla de mani' dentro de TTL (<= 30 min) SÍ se fusiona en el desayuno."""
    now = datetime.now(timezone.utc)
    mock_meals = [{
        "id": "meal-des-fresh",
        "meal_name": "4 huevos y pan",
        "meal_type": "desayuno",
        "calories": 573,
        "protein": 40,
        "created_at": (now - timedelta(minutes=1)).isoformat(),  # 1 min atrás
    }]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda uid: mock_meals)

    messages = [
        HumanMessage(content="Desayuné 4 huevos y pan"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {"meal_type": "desayuno", "meal_name": "4 huevos y pan"}, "id": "t1"}]),
        ToolMessage(content="Registrado como desayuno. [ID_REGISTRO_DIARIO: meal-des-fresh]", tool_call_id="t1"),
        AIMessage(content="Anoté tu desayuno..."),
        HumanMessage(content="extra mantequilla de maní y queso cottage"),
    ]
    tool_args = {"meal_name": "mantequilla de maní y queso cottage", "calories": 195, "protein": 12, "meal_type": "desayuno"}

    fused = _intentar_fusionar_alimento_en_comida_existente("u-test", tool_args, messages)
    assert fused is not None
    assert fused["meal_id"] == "meal-des-fresh"
    assert fused["calories"] == 573 + 195


def test_anclaje_explicito_horas_despues_si_se_fusiona(monkeypatch):
    """A las 3 horas, si el usuario dice 'En el desayuno súmale un yogurt',
    SÍ se fusiona porque nombró explícitamente el slot ('desayuno')."""
    now = datetime.now(timezone.utc)
    mock_meals = [{
        "id": "meal-des-old",
        "meal_name": "4 huevos y pan",
        "meal_type": "desayuno",
        "calories": 573,
        "protein": 40,
        "created_at": (now - timedelta(hours=3)).isoformat(),
    }]
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda uid: mock_meals)

    messages = [
        HumanMessage(content="En el desayuno súmale un yogurt ratio protein"),
    ]
    tool_args = {"meal_name": "yogurt ratio protein", "calories": 170, "protein": 25, "meal_type": "desayuno"}

    fused = _intentar_fusionar_alimento_en_comida_existente("u-test", tool_args, messages)
    assert fused is not None
    assert fused["meal_id"] == "meal-des-old"
    assert fused["calories"] == 573 + 170
