"""[P1-PLAN-LOTE-905 · 2026-09-29] Modo voz con GPT-Live-1: su voz y sus oídos, NUESTRO coach como cerebro.

El dueño quiere probarlo él mismo, con un tope DURO de US$0,60. Aquí, sin red: el canal lateral es falso y el coach
también; lo que se prueba es el contrato (quién puede, el tope, qué se le devuelve a GPT-Live-1 y cuándo se cierra).
"""
from __future__ import annotations

import asyncio
import json
import queue

import pytest

import coach_live


@pytest.fixture
def habilitado(monkeypatch):
    monkeypatch.setenv("MEALFIT_COACH_LIVE_USUARIOS", "u-duenio, u-otro")
    monkeypatch.delenv("MEALFIT_COACH_LIVE_TOPE_USD", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    monkeypatch.setattr(coach_live, "gastado_usd", lambda: 0.0)


# ── 1. Quién puede y el tope ──────────────────────────────────────────────────────────────────────────────────

def test_solo_las_cuentas_habilitadas(habilitado, monkeypatch):
    assert coach_live.disponible_para("u-duenio") and coach_live.disponible_para("u-otro")
    assert not coach_live.disponible_para("u-cualquiera") and not coach_live.disponible_para(None)
    monkeypatch.setenv("MEALFIT_COACH_LIVE_USUARIOS", "")
    assert not coach_live.disponible_para("u-duenio"), "vacío = apagado para todos"


def test_sin_presupuesto_no_abre_ni_llama_a_openai(habilitado, monkeypatch):
    monkeypatch.setattr(coach_live, "gastado_usd", lambda: 0.60)
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("sin presupuesto no se llama a OpenAI"))
    with pytest.raises(coach_live.LiveNoDisponible) as e:
        coach_live.crear_sesion("u-duenio", "sdp", "s1")
    assert e.value.motivo == "presupuesto"


def test_el_tope_por_defecto_es_el_que_autorizo_el_duenio(monkeypatch):
    monkeypatch.delenv("MEALFIT_COACH_LIVE_TOPE_USD", raising=False)
    assert coach_live.tope_usd() == 0.60
    assert abs(coach_live.costo_usd(60) - 0.05) < 1e-9, "US$0,05 por minuto, por segundo"


def test_crear_sesion_pide_delegacion_al_cliente_y_no_expone_la_clave(habilitado, monkeypatch):
    import httpx
    visto = {}

    class _R:
        status_code = 201

        def json(self):
            return {"session": {"id": "live_1"}, "transport": {"type": "webrtc", "sdp": "respuesta"}}

    def _post(url, headers=None, json=None, timeout=None):
        visto.update(url=url, headers=headers, cuerpo=json)
        return _R()

    monkeypatch.setattr(httpx, "post", _post)
    monkeypatch.setattr(coach_live.threading, "Thread", lambda *a, **k: type("T", (), {"start": lambda self: None})())
    live_id, sdp = coach_live.crear_sesion("u-duenio", "oferta", "s1")
    assert (live_id, sdp) == ("live_1", "respuesta")
    assert visto["url"] == "https://api.openai.com/v1/live/sessions"
    c = visto["cuerpo"]
    assert c["session"]["model"] == "gpt-live-1"
    assert c["session"]["delegation"] == {"type": "client"}, "el cerebro es NUESTRO coach"
    assert c["transport"] == {"type": "webrtc", "sdp": "oferta"}
    assert "Delegation policy" in c["session"]["instructions"]
    assert coach_live.sesion_de("live_1", "u-duenio") is not None
    assert coach_live.sesion_de("live_1", "u-otro") is None, "la sesión es de quien la abrió"


# ── 2. El canal lateral ───────────────────────────────────────────────────────────────────────────────────────

class _WS:
    def __init__(self, eventos):
        self.q = queue.Queue()
        for e in eventos:
            self.q.put(json.dumps(e))
        self.enviados = []

    def recv(self, timeout=None):
        try:
            return self.q.get(timeout=0.01)
        except queue.Empty:
            raise TimeoutError

    def send(self, texto):
        self.enviados.append(json.loads(texto))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _correr(monkeypatch, eventos, turno=None):
    ws = _WS(eventos)
    import websockets.sync.client as wsc
    monkeypatch.setattr(wsc, "connect", lambda *a, **k: ws)
    usos = []
    monkeypatch.setattr(coach_live, "registrar_uso", lambda uid, lid, seg, motivo: usos.append((uid, lid, seg, motivo)))
    monkeypatch.setattr(coach_live, "correr_turno_del_coach",
                        turno or (lambda s, dicho: (f"Listo, anoté {dicho}. [UI_ACTION: REFRESH_INVENTORY]",
                                                    {"ajustes_de_app": {"hidratacion": True}})))
    hilos = []
    real_thread = coach_live.threading.Thread

    def _hilo(target=None, args=(), **k):
        class _T:
            def start(self_):
                target(*args)
        hilos.append(target)
        return _T()

    monkeypatch.setattr(coach_live.threading, "Thread", _hilo)
    s = coach_live.SesionLive(live_id="live_x", user_id="u-duenio", chat_session_id="s1")
    coach_live._canal_lateral(s, "sk")
    monkeypatch.setattr(coach_live.threading, "Thread", real_thread)
    return s, ws, usos


def test_una_delegacion_corre_el_coach_con_lo_que_dijo_y_devuelve_su_respuesta(habilitado, monkeypatch):
    s, ws, usos = _correr(monkeypatch, [
        {"type": "session.started"},
        {"type": "session.input_transcript.delta", "delta": "me comí dos huevos "},
        {"type": "session.input_transcript.delta", "delta": "con pan"},
        {"type": "session.delegation.created", "delegation": {"id": "d1", "type": "delegation", "target": "client"}},
        {"type": "session.closed", "reason": "close_requested", "usage": {"seconds": 42}},
    ])
    comentario = [e for e in ws.enviados if e["type"] == "session.commentary.append"]
    assert len(comentario) == 1
    assert comentario[0]["delegation_id"] == "d1"
    assert comentario[0]["content"] == "Listo, anoté me comí dos huevos con pan.", "sin etiquetas UI_ACTION"
    assert s.novedades[0]["oido"] == "me comí dos huevos con pan"
    assert s.novedades[0]["ajustes_de_app"] == {"hidratacion": True}
    assert usos == [("u-duenio", "live_x", 42.0, "close_requested")]
    assert s.cerrada


def test_al_llegar_al_tope_el_servidor_cierra_la_sesion(habilitado, monkeypatch):
    monkeypatch.setattr(coach_live, "gastado_usd", lambda: 0.0)
    s, ws, usos = _correr(monkeypatch, [
        {"type": "session.usage.updated", "usage": {"seconds": 30}},
        {"type": "session.usage.updated", "usage": {"seconds": 301}},
        {"type": "session.closed", "reason": "close_requested", "usage": {"seconds": 302}},
    ])
    assert [e["type"] for e in ws.enviados] == ["session.close"]
    assert usos[0][2] == 302.0


def test_el_gasto_de_antes_cuenta_para_el_tope(habilitado, monkeypatch):
    s0 = coach_live.SesionLive(live_id="l", user_id="u", chat_session_id="s", gastado_antes_usd=0.58)
    ws = _WS([{"type": "session.usage.updated", "usage": {"seconds": 30}}, {"type": "session.closed", "usage": {"seconds": 30}}])
    import websockets.sync.client as wsc
    monkeypatch.setattr(wsc, "connect", lambda *a, **k: ws)
    monkeypatch.setattr(coach_live, "registrar_uso", lambda *a: None)
    coach_live._canal_lateral(s0, "sk")
    assert ws.enviados and ws.enviados[0]["type"] == "session.close", "0,58 + 0,025 ≥ 0,60"


def test_si_el_coach_falla_no_se_queda_mudo(habilitado, monkeypatch):
    def _boom(s, dicho):
        raise RuntimeError("DeepSeek caído")
    s, ws, _ = _correr(monkeypatch, [
        {"type": "session.input_transcript.delta", "delta": "hola"},
        {"type": "session.delegation.created", "delegation": {"id": "d1", "target": "client"}},
        {"type": "session.closed", "usage": {"seconds": 5}},
    ], turno=_boom)
    c = [e for e in ws.enviados if e["type"] == "session.commentary.append"][0]
    assert "no pude" in c["content"]


def test_el_gasto_va_a_llm_usage_events_con_su_precio_por_minuto(monkeypatch):
    import db
    visto = {}
    monkeypatch.setattr(db, "execute_sql_write", lambda sql, params: visto.update(sql=sql, params=params))
    coach_live.registrar_uso("u", "live_1", 120, "close_requested")
    assert "INSERT INTO llm_usage_events" in visto["sql"]
    assert visto["params"][1:3] == ("gpt-live-1", "coach_live_voice")
    assert visto["params"][5] == 100_000, "2 min × US$0,05 = US$0,10"
    src = open(coach_live.__file__, encoding="utf-8").read()
    assert "log_api_usage" not in src, "la voz no quema créditos del plan"


# ── 3. Los endpoints ──────────────────────────────────────────────────────────────────────────────────────────

def test_sin_sesion_o_sin_habilitar_no_hay_live(habilitado):
    from fastapi import HTTPException
    from routers import chat
    assert asyncio.run(chat.api_chat_live_disponible("u-cualquiera")) == {"disponible": False}
    with pytest.raises(HTTPException) as e:
        asyncio.run(chat.api_chat_live_sesion({"sdp": "x", "session_id": "s"}, None, None))
    assert e.value.status_code == 401


def test_las_novedades_son_solo_del_duenio_de_la_sesion(habilitado):
    from fastapi import HTTPException
    from routers import chat
    s = coach_live.SesionLive(live_id="live_n", user_id="u-duenio", chat_session_id="s1")
    s.novedades = [{"n": 1, "oido": "a"}, {"n": 2, "oido": "b"}]
    coach_live.SESIONES["live_n"] = s
    r = asyncio.run(chat.api_chat_live_novedades("live_n", 1, "u-duenio", None))
    assert [n["n"] for n in r["novedades"]] == [2]
    with pytest.raises(HTTPException):
        asyncio.run(chat.api_chat_live_novedades("live_n", 0, "u-otro", None))
