# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-692 · 2026-09-28] «Bioboros te respondió»: push al terminar la respuesta del coach."""
from __future__ import annotations

import io
import os

import aviso_respuesta_chat as arc
import push_i18n

_UID = "61a13831-2a70-4437-a084-0d3e09b653e4"


def test_cuerpo_sin_markdown_y_en_una_linea():
    t = "Con eso cierras **flojo** de proteína:\n\n- **2 huevos** con [maduro](https://x) [UI_ACTION: OPEN_DIARY]"
    assert arc.cuerpo_de(t) == "Con eso cierras flojo de proteína: 2 huevos con maduro"


def test_cuerpo_cortado_en_palabra():
    c = arc.cuerpo_de("palabra " * 60)
    assert len(c) <= arc.MAX_CUERPO + 1 and c.endswith("…") and not c.endswith(" …")


def test_envia_con_solo_si_no_mira(monkeypatch):
    enviados, tareas = [], []
    import utils_push, bg_executor
    monkeypatch.setattr(utils_push, "send_push_notification", lambda *a, **k: enviados.append((a, k)))
    monkeypatch.setattr(bg_executor, "submit_bg_task", lambda fn, task_name=None: tareas.append(task_name) or fn())
    assert arc.avisar_respuesta(_UID, "Listo, **anotado**.") is True
    (args, kw), = enviados
    assert args == (_UID, arc.TITULO, "Listo, anotado.")
    assert kw["solo_si_no_mira"] is True and kw["url"] == "/dashboard/agent" and kw["tag"] == "chat-respuesta"
    assert tareas == ["aviso_respuesta_chat"]


def test_invitados_vacios_y_knob(monkeypatch):
    assert arc.avisar_respuesta("guest", "hola") is False
    assert arc.avisar_respuesta(None, "hola") is False
    assert arc.avisar_respuesta(_UID, "  ") is False
    monkeypatch.setenv("MEALFIT_CHAT_REPLY_PUSH", "false")
    assert arc.avisar_respuesta(_UID, "hola") is False


def test_nunca_lanza(monkeypatch):
    import bg_executor
    def _boom(*a, **k):
        raise RuntimeError("pool lleno")
    monkeypatch.setattr(bg_executor, "submit_bg_task", _boom)
    assert arc.avisar_respuesta(_UID, "hola") is False


def test_titulo_traducido_a_los_cuatro_idiomas():
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        tr = push_i18n.translate_push_text(arc.TITULO, loc)
        assert tr and tr != arc.TITULO


def test_el_done_del_stream_lo_dispara_con_el_usuario_verificado():
    s = io.open(os.path.join(os.path.dirname(__file__), "..", "routers", "chat.py"), encoding="utf-8").read()
    i = s.index("[P2-CHAT-DONE-PERSIST-LOUD] la respuesta del modelo")
    tramo = s[i:i + 900]
    assert "avisar_respuesta(verified_user_id, response_text)" in tramo


def test_marker():
    import app
    assert "P1-PLAN-LOTE-69" in app._LAST_KNOWN_PFIX   # 692 o posterior del bloque
