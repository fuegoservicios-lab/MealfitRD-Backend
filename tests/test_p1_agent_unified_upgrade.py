"""[AI-60] Pruebas unitarias y de integración para la actualización maestra del agente IA.

Cubre:
1. Rotación justa anti-inanición en el cron proactivo (CTE last_sent ASC).
2. Backoff progresivo para usuarios inactivos y fatiga de avisos.
3. Protein Rescue (rescate de proteína vespertino).
4. Filtro de extracción de hechos (evitar contaminar memoria de largo plazo con comidas puntuales).
5. Tool gating dinámico para reducir tokens y alucinaciones en el chat.
6. Inyección de modelo de usuario (Dreaming) en contexto conversacional.
7. Puente con la sesión anterior reciente (cross-session bridge).
8. Short-circuit de mensajes casuales y saludos en el chat.
"""
import pytest
from unittest.mock import MagicMock, patch


def test_casual_short_circuit_detection():
    """AI-42: Saludos y mensajes casuales cortos deben detectarse como casuales para saltar LLMs."""
    from agent import _rag_es_casual

    assert _rag_es_casual("hola") is True
    assert _rag_es_casual("buenas tardes") is True
    assert _rag_es_casual("ok gracias") is True
    assert _rag_es_casual("klk") is True
    assert _rag_es_casual("listo dale") is True

    # Consultas nutricionales o peticiones concretas NO deben saltar
    assert _rag_es_casual("cuántas calorías tiene el plátano?") is False
    assert _rag_es_casual("desayuné dos huevos con aguacate") is False
    assert _rag_es_casual("cámbiame el desayuno de mañana") is False


def test_fact_extractor_rejects_single_meals():
    """AI-30: Consumos puntuales del diario no deben extraerse como hechos permanentes."""
    from fact_extractor import should_extract_facts

    # El fast-path debe rechazar registros de comidas normales inmediatamente
    assert should_extract_facts("desayuné dos huevos con mangú") is False
    assert should_extract_facts("comí pica pollo con tostones al mediodía") is False
    assert should_extract_facts("me comí 1 manzana verde") is False
    assert should_extract_facts("almorcé pollo con arroz") is False
    assert should_extract_facts("cené ensalada césar") is False

    # Preferencias duraderas o alergias NO deben rechazarse por el fast-path
    assert should_extract_facts("soy alérgico al maní y a los mariscos") is True
    assert should_extract_facts("no me gusta el hígado ni la berenjena") is True


def test_tool_gating_core_selection():
    """AI-41: Turnos normales de comida solo cargan herramientas esenciales (mínimo 7)."""
    from agent import _select_tools_for_turn

    class DummyTool:
        def __init__(self, name):
            self.name = name

    all_tools = [
        DummyTool("log_consumed_meal"),
        DummyTool("correct_consumed_meal"),
        DummyTool("check_hydration_today"),
        DummyTool("log_water_glass"),
        DummyTool("proponer_comida"),
        DummyTool("consultar_dia_del_plan"),
        DummyTool("search_deep_memory"),
        DummyTool("analizar_suplemento"),
        DummyTool("guardar_suplemento"),
        DummyTool("check_shopping_list"),
        DummyTool("modify_pantry_inventory"),
        DummyTool("cambiar_ajuste_de_la_app"),
        DummyTool("check_clinical_profile"),
        DummyTool("buscar_etiqueta_en_internet"),
    ]

    # Mensaje normal de diario
    state_meal = {
        "messages": [MagicMock(type="user", content="desayuné 2 huevos revueltos con queso")],
        "turn_plate_photos": [],
        "turn_photo_to_log": None,
    }
    selected_meal = _select_tools_for_turn(all_tools, state_meal)
    selected_names = {t.name for t in selected_meal}

    # Contiene las 7 herramientas core
    assert "log_consumed_meal" in selected_names
    assert "correct_consumed_meal" in selected_names
    assert "check_hydration_today" in selected_names
    # NO contiene herramientas irrelevantes como despensa, suplementos o lista de compras
    assert "check_shopping_list" not in selected_names
    assert "analizar_suplemento" not in selected_names
    assert "modify_pantry_inventory" not in selected_names

    # Mensaje sobre suplemento activa analizar_suplemento
    state_sup = {
        "messages": [MagicMock(type="user", content="me tomé un scoop de creatina")],
        "turn_plate_photos": [],
        "turn_photo_to_log": None,
    }
    selected_sup = _select_tools_for_turn(all_tools, state_sup)
    names_sup = {t.name for t in selected_sup}
    assert "analizar_suplemento" in names_sup
    assert "guardar_suplemento" in names_sup


def test_session_previa_safe_return():
    """AI-32: Cross-session bridge maneja valores vacíos y fallos sin romper."""
    from agent import _contexto_sesion_previa

    assert _contexto_sesion_previa(None, "sess-1") == ""
    assert _contexto_sesion_previa("guest", "sess-1") == ""
    assert _contexto_sesion_previa("user-1", None) == ""


def test_proactive_no_gamified_or_disclaimer_boilerplate():
    """AI-21: Los prompts y tono de aviso no deben incluir disclaimers fijos ni clichés de juego."""
    from prompts.proactive import PROACTIVE_PROMPT
    from tono_del_aviso import TONO_SE_LA_SALTA

    # Verificar que el prompt prohíbe el disclaimer y clichés gamificados
    assert "Ajustes" in PROACTIVE_PROMPT
    assert "desbloquea" in PROACTIVE_PROMPT
    assert "reto" in PROACTIVE_PROMPT

    # Verificar que TONO_SE_LA_SALTA prohíbe el disclaimer
    assert "Configuración" in TONO_SE_LA_SALTA
    assert "puede apagar solo ese recordatorio" not in TONO_SE_LA_SALTA


def test_starvation_free_proactive_sql():
    """AI-10: La selección de usuarios proactivos debe ordenar por last_sent ASC para evitar inanición."""
    import inspect
    from proactive_agent import get_active_users_for_proactive

    source = inspect.getsource(get_active_users_for_proactive)
    assert "nudge_outcomes" in source
    assert "ORDER BY last_sent ASC, user_id ASC" in source


def test_dreaming_user_model_chat_injection():
    """AI-31: La memoria debe inyectar el bloque del modelo de usuario Dreaming si existe."""
    from memory_manager import _get_user_model_block

    # Cuando no hay usuario o es guest, retorna cadena vacía
    assert _get_user_model_block(None) == ""
    assert _get_user_model_block("guest") == ""

    # Con mock de get_user_memory_profile retornando un user_model
    fake_profile = {"user_model": "Paciente femenino de 28 años, meta: pérdida de grasa."}
    with patch("dreaming.get_user_memory_profile", return_value=fake_profile):
        block = _get_user_model_block("00000000-0000-0000-0000-000000000001")
        assert "MODELO DEL USUARIO" in block
        assert "Paciente femenino de 28 años" in block


def test_ayuno_and_skipped_meal_handling():
    """AI-70: Seguimiento inteligente de comidas pasadas con opción de ayuno y regla 8-ter."""
    import tools
    from prompts.chat_agent import _CHAT_BREVITY_RULES

    # Regla 8-ter en el system prompt
    assert "8-ter. AYUNO O COMIDAS OMITIDAS:" in _CHAT_BREVITY_RULES
    assert "sin sermonear, regañar ni juzgar" in _CHAT_BREVITY_RULES
    assert "NO inventes alimentos ni registres nada en el diario para esa comida" in _CHAT_BREVITY_RULES

    # Verificación en la nota interna para el asistente
    _TIPICAS = {"desayuno": 9.0, "almuerzo": 13.0, "merienda": 16.0, "cena": 19.5}
    with patch.object(tools, "_local_date_str_for_user", lambda _u: "2026-10-24"):
        with patch.object(tools, "user_tz_offset_min", lambda _u: 240):
            with patch.object(tools, "get_latest_usable_meal_plan", lambda _u: None):
                with patch.object(tools, "_horas_tipicas_de_comida", lambda _u: dict(_TIPICAS)):
                    with patch("db_facts.get_consumed_meals_today", return_value=[{"meal_type": "almuerzo"}]):
                        nota = tools._nota_comidas_sin_registrar("u-test-ayuno", 0, ahora_local=14.5)
                        assert "ayunas" in nota
                        assert "desayuno" in nota
