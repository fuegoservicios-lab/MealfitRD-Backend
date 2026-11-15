"""Tests para reemplazo in-place de avisos proactivos previos sin respuesta."""
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import db_chat
import db_facts
import proactive_agent
import routers.chat as chat_router


def test_prune_reemplaza_avisos_sin_mensajes_del_usuario():
    rows = [
        {"id": "m-desayuno", "role": "model"},
        {"id": "m-merienda", "role": "model"},
    ]
    with patch.object(db_chat, "connection_pool", object()), \
         patch.object(db_chat, "execute_sql_query", return_value=rows), \
         patch.object(db_chat, "execute_sql_write", return_value=[{"id": "m-desayuno"}, {"id": "m-merienda"}]) as w:
        eliminados = db_chat.prune_unanswered_proactive_messages("ses-1")

    assert eliminados == 2
    assert w.call_count == 1
    args = w.call_args[0][1]
    assert args[0] == "ses-1"
    assert set(args[1]) == {"m-desayuno", "m-merienda"}


def test_prune_conserva_respuesta_real_y_solo_poda_avisos_posteriores_sin_respuesta():
    rows = [
        {"id": "u-1", "role": "user"},
        {"id": "m-reply", "role": "model"},
        {"id": "m-nudge-old", "role": "model"},
    ]
    with patch.object(db_chat, "connection_pool", object()), \
         patch.object(db_chat, "execute_sql_query", return_value=rows), \
         patch.object(db_chat, "execute_sql_write", return_value=[{"id": "m-nudge-old"}]) as w:
        eliminados = db_chat.prune_unanswered_proactive_messages("ses-2")

    assert eliminados == 1
    assert w.call_args[0][1][1] == ["m-nudge-old"]


def test_prune_no_toca_conversacion_normal_sin_avisos_encadenados():
    rows = [
        {"id": "u-1", "role": "user"},
        {"id": "m-reply", "role": "model"},
    ]
    with patch.object(db_chat, "connection_pool", object()), \
         patch.object(db_chat, "execute_sql_query", return_value=rows), \
         patch.object(db_chat, "execute_sql_write") as w:
        eliminados = db_chat.prune_unanswered_proactive_messages("ses-3")

    assert eliminados == 0
    assert w.call_count == 0


def test_prune_fail_open_si_falla_db():
    with patch.object(db_chat, "connection_pool", object()), \
         patch.object(db_chat, "execute_sql_query", side_effect=RuntimeError("db down")):
        assert db_chat.prune_unanswered_proactive_messages("ses-err") == 0


def test_proactive_agent_poda_antes_de_guardar_y_conserva_firma():
    src = Path(proactive_agent.__file__).read_text(encoding="utf-8")
    save_pos = src.find('save_message(session_id, "model", content)')
    prune_pos = src.find("prune_unanswered_proactive_messages(session_id)")
    assert save_pos != -1
    assert prune_pos != -1
    assert prune_pos < save_pos
    assert save_pos - prune_pos < 250


def test_reconcile_311pm_elimina_desayuno_y_conserva_almuerzo_pendiente():
    """A las 3:11 PM (19:11 UTC en RD UTC-4) sin almuerzo registrado: elimina desayuno viejo y deja solo almuerzo."""
    rows = [
        {
            "id": "m-desayuno",
            "role": "model",
            "content": "¿Te provoca desayunar una arepa rellena de carne de res y aguacate?",
            "created_at": "2026-11-15T12:45:00+00:00",  # 08:45 RD
        },
        {
            "id": "m-almuerzo",
            "role": "model",
            "content": "Para almorzar, ¿qué te parece un filete de pescado con batata asada?",
            "created_at": "2026-11-15T16:45:00+00:00",  # 12:45 RD
        },
    ]
    ahora_311pm_utc = datetime(2026, 11, 15, 19, 11, 0, tzinfo=timezone.utc)
    with patch.object(db_chat, "connection_pool", object()), \
         patch.object(db_chat, "execute_sql_query", return_value=rows), \
         patch.object(db_chat, "execute_sql_write", return_value=[{"id": "m-desayuno"}]) as w, \
         patch.object(db_facts, "user_tz_offset_min", return_value=240), \
         patch.object(db_facts, "get_consumed_meals_today", return_value=[]):
        eliminados = db_chat.reconcile_unanswered_proactive_messages("ses-311", user_id="usr-1", now_utc=ahora_311pm_utc)

    assert eliminados == 1
    assert w.call_args[0][1] == ("ses-311", ["m-desayuno"])


def test_reconcile_elimina_todos_cuando_almuerzo_ya_fue_registrado():
    """Cuando el usuario registra el almuerzo a las 3:15 PM, desaparecen desayuno Y almuerzo."""
    rows = [
        {
            "id": "m-desayuno",
            "role": "model",
            "content": "¿Te provoca desayunar una arepa rellena?",
            "created_at": "2026-11-15T12:45:00+00:00",
        },
        {
            "id": "m-almuerzo",
            "role": "model",
            "content": "Para almorzar, ¿qué te parece un filete de pescado?",
            "created_at": "2026-11-15T16:45:00+00:00",
        },
    ]
    ahora_315pm_utc = datetime(2026, 11, 15, 19, 15, 0, tzinfo=timezone.utc)
    consumed = [{"meal_type": "Almuerzo", "consumed_at": "2026-11-15T19:14:00+00:00"}]
    with patch.object(db_chat, "connection_pool", object()), \
         patch.object(db_chat, "execute_sql_query", return_value=rows), \
         patch.object(db_chat, "execute_sql_write", return_value=[{"id": "m-desayuno"}, {"id": "m-almuerzo"}]) as w, \
         patch.object(db_facts, "user_tz_offset_min", return_value=240), \
         patch.object(db_facts, "get_consumed_meals_today", return_value=consumed):
        eliminados = db_chat.reconcile_unanswered_proactive_messages("ses-315", user_id="usr-1", now_utc=ahora_315pm_utc)

    assert eliminados == 2
    assert set(w.call_args[0][1][1]) == {"m-desayuno", "m-almuerzo"}


def test_reconcile_nunca_toca_system_title_ni_respuesta_real():
    rows = [
        {"id": "m-title", "role": "model", "content": "[SYSTEM_TITLE] Mi sesión", "created_at": "2026-11-15T10:00:00+00:00"},
        {"id": "u-1", "role": "user", "content": "Hola", "created_at": "2026-11-15T11:00:00+00:00"},
        {"id": "m-reply", "role": "model", "content": "Hola Angelo", "created_at": "2026-11-15T11:01:00+00:00"},
        {"id": "m-almuerzo", "role": "model", "content": "Para almorzar, ¿pollo guisado?", "created_at": "2026-11-15T16:30:00+00:00"},
    ]
    ahora_cena_utc = datetime(2026, 11, 15, 23, 30, 0, tzinfo=timezone.utc)  # 19:30 RD -> almuerzo expirado
    with patch.object(db_chat, "connection_pool", object()), \
         patch.object(db_chat, "execute_sql_query", return_value=rows), \
         patch.object(db_chat, "execute_sql_write", return_value=[{"id": "m-almuerzo"}]) as w, \
         patch.object(db_facts, "user_tz_offset_min", return_value=240), \
         patch.object(db_facts, "get_consumed_meals_today", return_value=[]):
        eliminados = db_chat.reconcile_unanswered_proactive_messages("ses-real", user_id="usr-1", now_utc=ahora_cena_utc)

    assert eliminados == 1
    assert w.call_args[0][1] == ("ses-real", ["m-almuerzo"])


def test_api_history_concilia_antes_de_leer_mensajes():
    src = Path(chat_router.__file__).read_text(encoding="utf-8")
    rec_pos = src.find("reconcile_unanswered_proactive_messages(session_id")
    get_pos = src.find("messages = get_session_messages(session_id)")
    assert rec_pos != -1 and get_pos != -1 and rec_pos < get_pos
