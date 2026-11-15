"""Tests para reemplazo in-place de avisos proactivos previos sin respuesta."""
from pathlib import Path
from unittest.mock import patch

import db_chat
import proactive_agent


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
