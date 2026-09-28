# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-760 · 2026-09-28] El coach termina su respuesta aunque el usuario salga de la app.

Caso vivo (28-sep 12:25 UTC): «Cómo estás?» → el usuario salió de la app → `SSE abortado por cliente kind=GeneratorExit
chunk_observed=False` a los 24 s → «No llegó la respuesta del coach». La generación vivía dentro del stream HTTP.
"""
from __future__ import annotations

import io
import os
import threading
import time

import pytest

import turno_desacoplado as td

_BACKEND = os.path.join(os.path.dirname(__file__), "..")


def _src(rel):
    return io.open(os.path.join(_BACKEND, rel), encoding="utf-8").read()


def _turno(registro, n=5, pausa=0.02):
    """Un turno de mentira: emite n trozos y, como el de verdad, «guarda» la respuesta al llegar al final."""
    try:
        for i in range(n):
            time.sleep(pausa)
            yield f"data: {i}\n\n"
        registro.append("guardada")
    except GeneratorExit:
        registro.append("cortada")
        raise
    finally:
        registro.append("finally")


def _esperar(cond, tope=5.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > tope:
            return False
        time.sleep(0.01)
    return True


def test_quien_lee_recibe_todo():
    reg = []
    assert list(td.desacoplar(_turno(reg), "s-todo")) == [f"data: {i}\n\n" for i in range(5)]
    assert _esperar(lambda: "finally" in reg) and reg == ["guardada", "finally"]


def test_si_el_cliente_se_va_el_turno_termina_y_se_guarda():
    """El corazón del lote: el lector abandona tras el primer trozo (la app se cerró) y la respuesta se guarda igual."""
    reg = []
    lector = td.desacoplar(_turno(reg, n=6, pausa=0.03), "s-se-fue")
    assert next(lector) == "data: 0\n\n"
    lector.close()   # lo que hace Starlette cuando el teléfono corta la conexión
    assert _esperar(lambda: "finally" in reg)
    assert reg == ["guardada", "finally"], "sin el hilo propio el turno moría con la conexión ('cortada')"


def test_detener_corta_el_turno_y_su_finally_corre():
    reg = []
    lector = td.desacoplar(_turno(reg, n=50, pausa=0.02), "s-detener")
    next(lector)
    assert td.en_curso("s-detener")
    assert td.detener("s-detener") is True
    assert _esperar(lambda: "finally" in reg)
    assert "cortada" in reg and "guardada" not in reg
    assert not td.en_curso("s-detener")
    assert td.detener("s-detener") is False   # ya no hay turno


def test_detener_sin_turno_o_sin_chat():
    assert td.detener("no-existe") is False
    assert td.detener("") is False and td.detener(None) is False


def test_dos_turnos_seguidos_del_mismo_chat_no_se_pisan_la_marca():
    reg1, reg2 = [], []
    l1 = td.desacoplar(_turno(reg1, n=2, pausa=0.01), "s-mismo")
    list(l1)
    assert _esperar(lambda: "finally" in reg1)
    l2 = td.desacoplar(_turno(reg2, n=40, pausa=0.02), "s-mismo")
    next(l2)
    assert td.en_curso("s-mismo")   # el final del primero no borró la marca del segundo
    td.detener("s-mismo")
    assert _esperar(lambda: "finally" in reg2)


def test_un_error_dentro_del_turno_no_cuelga_al_lector():
    def _malo():
        yield "data: a\n\n"
        raise RuntimeError("boom")
    assert list(td.desacoplar(_malo(), "s-error")) == ["data: a\n\n"]
    assert not td.en_curso("s-error")


def test_el_stream_del_chat_usa_el_hilo_propio_y_el_knob_lo_revierte():
    s = _src("routers/chat.py")
    assert "_flujo = desacoplar(event_generator(), session_id) if _turno_desacoplado() else event_generator()" in s
    assert 'return StreamingResponse(_flujo, media_type="text/event-stream")' in s


def test_knob(monkeypatch):
    assert td.activo() is True
    monkeypatch.setenv("MEALFIT_CHAT_TURN_DETACHED", "false")
    assert td.activo() is False


@pytest.fixture
def cliente(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routers.chat as rc
    import db_chat
    app = FastAPI()
    app.include_router(rc.router)   # el router ya trae su prefijo /api/chat
    app.dependency_overrides[rc.get_verified_user_id] = lambda: "u-1"
    app.dependency_overrides[rc._CHAT_STOP_LIMITER] = lambda: None
    monkeypatch.setattr(db_chat, "get_session_owner", lambda sid: {"s-ajeno": "u-2"}.get(sid, "u-1"))
    return TestClient(app)


def test_endpoint_stop(cliente):
    assert cliente.post("/api/chat/stop", json={}).status_code == 400
    assert cliente.post("/api/chat/stop", json={"session_id": "s-ajeno"}).status_code == 403
    assert cliente.post("/api/chat/stop", json={"session_id": "s-libre"}).json() == {"stopped": False}
    reg = []
    lector = td.desacoplar(_turno(reg, n=50, pausa=0.02), "s-vivo")
    next(lector)
    assert cliente.post("/api/chat/stop", json={"session_id": "s-vivo"}).json() == {"stopped": True}
    assert _esperar(lambda: "finally" in reg) and "cortada" in reg


def test_marker():
    import re
    import app
    m = re.search(r"LOTE-(\d+)", app._LAST_KNOWN_PFIX)
    assert m and int(m.group(1)) >= 760, app._LAST_KNOWN_PFIX
