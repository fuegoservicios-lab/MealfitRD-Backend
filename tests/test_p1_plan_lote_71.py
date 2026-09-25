# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-71 · 2026-09-16] Al volver a iniciar sesión, el Agente abre el chat de HOY.

El dueño cerró sesión, volvió a entrar y el Agente le abrió un chat en blanco mientras «Recientes · Hoy» listaba su
conversación de esa mañana (en producción: una sola sesión, creada a las 08:26 RD, 6 mensajes, el último a las 12:58).
La regla del día (`P1-AGENT-SESSION-DAY`, frontend) solo leía `mealfit_current_session`, y el logout la borra a
propósito (`P2-CHAT-CACHE-XUSER`). El arreglo vive en el frontend: la sesión que abre la regla se cambia por la del
servidor cuando llega la lista.

Aquí vive el contrato entre los dos repos: los dos campos que lee la regla (`last_activity`, `title_key: 'empty'`), que
la lista sea solo del dueño del token, y que el logout siga borrando la sesión guardada — el arreglo no puede comprarse
aflojando la privacidad.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def test_la_lista_trae_los_dos_campos_que_lee_la_regla(monkeypatch):
    import db_chat

    sesiones = [
        {"id": "s-hoy", "created_at": "2026-09-16 12:26:47.817987+00", "user_id": "u"},
        {"id": "s-vacia", "created_at": "2026-09-16 16:59:00.000000+00", "user_id": "u"},
    ]
    mensajes = [
        {"session_id": "s-hoy", "content": "[SYSTEM_TITLE] Primer saludo", "created_at": "2026-09-16 12:26:47.825039+00", "role": "model"},
        {"session_id": "s-hoy", "content": "hola", "created_at": "2026-09-16 12:26:47.830000+00", "role": "user"},
        {"session_id": "s-hoy", "content": "¡Hola!", "created_at": "2026-09-16 16:58:13.337404+00", "role": "model"},
    ]
    monkeypatch.setattr(db_chat, "execute_sql_query", lambda *a, **k: [dict(m) for m in mensajes])

    out = {s["id"]: s for s in db_chat._process_and_sort_sessions([dict(s) for s in sesiones])}

    # `last_activity` es el ÚLTIMO mensaje, no la creación: el chat de las 08:26 sigue vivo a las 12:58.
    assert out["s-hoy"]["last_activity"] == "2026-09-16 16:58:13.337404+00"
    assert out["s-hoy"]["title_key"] is None and out["s-hoy"]["title"] == "Primer saludo"
    # Una sesión sin mensajes llega marcada, y la regla del frontend no la toma por «tu chat de hoy».
    assert out["s-vacia"]["title_key"] == "empty"
    assert out["s-vacia"]["last_activity"] == "2026-09-16 16:59:00.000000+00"


def test_la_lista_es_solo_del_dueno_del_token():
    # La adopción se fía de la lista: si otro usuario pudiera pedirla, abriría su chat.
    fuente = _src("routers/chat.py")
    cuerpo = fuente.split('@router.get("/sessions/{user_id}")', 1)[1].split("@router.", 1)[0]
    assert "if not verified_user_id or verified_user_id != user_id:" in cuerpo
    assert 'raise HTTPException(status_code=403, detail="Prohibido.")' in cuerpo


def test_la_regla_del_frontend_adopta_solo_la_sesion_que_abrio_ella():
    regla = _front("src/utils/chatSessionDay.js")
    assert "export const SESSION_AUTO_KEY = 'mealfit_current_session_auto';" in regla
    assert "export const sesionDelDiaAAdoptar = " in regla
    assert "export const sesionDeHoyEnServidor = " in regla
    # La marca nace con la sesión automática y muere con cualquier actividad o elección a mano.
    resolver = regla.split("export const resolverSesionDelDia", 1)[1].split("export const marcarActividad", 1)[0]
    assert "safeLocalStorageSet(SESSION_AUTO_KEY, id);" in resolver
    actividad = regla.split("export const marcarActividad", 1)[1].split("export const diaLocalDe", 1)[0]
    assert "safeLocalStorageRemove(SESSION_AUTO_KEY);" in actividad
    assert "s.title_key === 'empty'" in regla
    assert "s.last_activity || s.created_at" in regla


def test_el_agente_consulta_la_regla_con_la_primera_lista_del_servidor():
    agente = _front("src/pages/AgentPage.jsx")
    assert "P1-PLAN-LOTE-71" in agente
    assert "if (!isGuest && offset === 0) adoptarSesionDelDia(data.sessions || []);" in agente
    adoptar = agente.split("const adoptarSesionDelDia = useStableCallback(", 1)[1].split("const fetchChatSessions", 1)[0]
    # Nada del usuario en juego: turno, mensajes o borrador.
    for guarda in ("isTurnActiveRef.current", "messagesRef.current", "draftSnapshotRef.current"):
        assert guarda in adoptar, guarda
    assert "sesionDelDiaAAdoptar({ sesiones: sesionesDelServidor, actual: currentSessionIdRef.current })" in adoptar
    assert "setCurrentSessionId(deHoy);" in adoptar
    # Por el envoltorio, que marca la actividad y quita la marca automática.
    # [P1-PLAN-LOTE-226] un chat de otro día elegido en Recientes conserva SU día (`dia`); el resto, hoy.
    assert re.search(r"const setCurrentSessionId = \(id, dia\) => \{(?:\s*//[^\n]*\n)+\s*marcarActividad\(id, dia \|\| undefined\);", agente)


def test_el_logout_sigue_borrando_la_sesion_guardada():
    contexto = _front("src/context/AssessmentContext.jsx")
    assert "safeLocalStorageRemove(CHAT_CURRENT_SESSION_KEY);" in contexto


def test_la_prueba_que_monta_el_agente_existe():
    prueba = _front("src/__tests__/Agent.chat_del_dia_tras_login.test.jsx")
    assert "sin la clave local (el logout la borra), recupera del servidor el chat de hoy" in prueba
    assert "un «Nuevo chat» elegido a mano hoy se respeta al volver al Agente" in prueba


def test_marcador_y_documento():
    assert "P1-PLAN-LOTE-71" in _src("app.py")
    doc = _src("docs/chat_sesion_del_dia.md")
    assert "P1-PLAN-LOTE-71" in doc and "P1-AGENT-SESSION-DAY" in doc
