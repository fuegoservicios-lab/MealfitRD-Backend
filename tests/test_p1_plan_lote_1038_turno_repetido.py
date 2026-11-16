"""[P1-PLAN-LOTE-1038 · A-23 · 2026-11-16] Mismo turno dos veces = una sola respuesta y una sola escritura.

El cliente reutiliza el `client_message_id` al reintentar. El servidor, si ese mensaje ya estaba guardado, repite la
respuesta ya dada (sin modelo, sin herramientas, sin guardar, sin cobrar) en vez de ejecutar el turno otra vez.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import coach_turno_repetido as ctr  # noqa: E402

UID = "11111111-2222-3333-4444-555555555555"
SID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
CMID = "99999999-8888-7777-6666-555555555555"


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _obj(evento: str) -> dict:
    return json.loads(evento[len("data: "):])


# ───────────────────────────────────────────────────────────── la lectura
def test_la_consulta_busca_el_primer_mensaje_posterior_al_del_usuario():
    sql = " ".join(ctr._SQL_RESPUESTA.split())
    assert "u.session_id = %s AND u.client_message_id = %s AND u.role = 'user'" in sql
    assert "m.created_at > u.created_at" in sql and "ORDER BY m.created_at ASC LIMIT 1" in sql
    assert "NOT LIKE '[SYSTEM_TITLE]%%'" in sql, "el título de la sesión no es la respuesta"


@pytest.mark.parametrize("fila,esperado", [
    ({"role": "model", "content": "Anotado: 2 huevos."}, "Anotado: 2 huevos."),
    ({"role": "user", "content": "y otra cosa"}, None),          # otro mensaje del usuario en medio: no se repite
    ({"role": "model", "content": "   "}, None),
    (None, None),
    ([], None),
])
def test_respuesta_ya_dada(fila, esperado):
    vistos = []

    def q(sql, params=None, **k):
        vistos.append((params, k))
        return fila
    assert ctr.respuesta_ya_dada(SID, CMID, execute_sql_query=q) == esperado
    assert vistos == [((SID, CMID), {"fetch_one": True})]


def test_sin_identificadores_no_consulta_y_un_error_de_lectura_falla_abierto(monkeypatch):
    def no(*a, **k):
        raise AssertionError("no debe consultar")
    assert ctr.respuesta_ya_dada(SID, None, execute_sql_query=no) is None
    assert ctr.respuesta_ya_dada("", CMID, execute_sql_query=no) is None
    import fail_open as fo
    contados = []
    monkeypatch.setattr(fo, "fail_open", lambda sitio, exc, **k: contados.append(sitio))

    def rota(*a, **k):
        raise RuntimeError("db")
    assert ctr.respuesta_ya_dada(SID, CMID, execute_sql_query=rota) is None
    assert contados == ["chat_turno_repetido"]


# ───────────────────────────────────────────────────────────── la decisión
def test_con_respuesta_se_repite_marcada():
    eventos = ctr.resolver(SID, CMID, corr="c-1", turno_vivo=lambda s: True,
                           execute_sql_query=lambda *a, **k: {"role": "model", "content": "Anotado."})
    chunk, done = [_obj(e) for e in eventos]
    assert chunk == {"type": "chunk", "text": "Anotado.", "replayed": True}
    assert done["type"] == "done" and done["response"] == "Anotado." and done["replayed"] is True and done["corr"] == "c-1"
    # la misma forma que el `done` de un turno corto: el cliente no encuentra claves que falten
    for clave in ("updated_fields", "new_plan", "coherence_warnings", "pantry_modified_at", "pantry_depleted_items",
                  "ajustes_de_app", "diary_corrected_meal_ids"):
        assert clave in done
    assert ctr.es_repeticion(done) and not ctr.es_repeticion({"type": "done"})


def test_sin_respuesta_y_turno_vivo_avisa_en_vez_de_lanzar_otro():
    eventos = ctr.resolver(SID, CMID, turno_vivo=lambda s: True, execute_sql_query=lambda *a, **k: None)
    assert [_obj(e) for e in eventos] == [{"type": "error", "code": "turn_in_progress", "message": ctr.MENSAJE_EN_CURSO}]


def test_sin_respuesta_y_sin_turno_vivo_el_turno_se_ejecuta():
    assert ctr.resolver(SID, CMID, turno_vivo=lambda s: False, execute_sql_query=lambda *a, **k: None) is None

    def revienta(s):
        raise RuntimeError("x")
    assert ctr.resolver(SID, CMID, turno_vivo=revienta, execute_sql_query=lambda *a, **k: None) is None


def test_apagado_por_knob(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_TURNO_REPETIDO", "0")

    def no(*a, **k):
        raise AssertionError("apagado no consulta")
    assert ctr.resolver(SID, CMID, turno_vivo=lambda s: True, execute_sql_query=no) is None


# ───────────────────────────────────────────────────────────── el guardado dice si era un reintento
def test_el_guardado_marca_el_reintento(monkeypatch):
    import db_chat
    insertados = []
    monkeypatch.setattr(db_chat, "connection_pool", object())
    monkeypatch.setattr(db_chat, "_owned_attachment_metadata", lambda *a, **k: [])
    monkeypatch.setattr(db_chat, "_procesar_respuesta_a_nudge", lambda *a, **k: None)
    monkeypatch.setattr(db_chat, "execute_sql_transaction", lambda pasos: insertados.append(pasos[0][1][0]))

    # primera vez: la fila que queda es la que se acaba de insertar
    monkeypatch.setattr(db_chat, "execute_sql_query", lambda *a, **k: {"id": insertados[-1]})
    info = {}
    db_chat.save_message_with_attachments(SID, "hola", UID, [], client_message_id=CMID, info=info)
    assert info == {"repetido": False}

    # reintento: el ON CONFLICT no insertó nada y la fila que hay es la de antes
    monkeypatch.setattr(db_chat, "execute_sql_query", lambda *a, **k: {"id": "fila-de-antes"})
    info = {}
    assert db_chat.save_message_with_attachments(SID, "hola", UID, [], client_message_id=CMID, info=info) == "fila-de-antes"
    assert info == {"repetido": True}
    # sin `info` sigue funcionando como siempre
    assert db_chat.save_message_with_attachments(SID, "hola", UID, [], client_message_id=CMID) == "fila-de-antes"


# ───────────────────────────────────────────────────────────── el router
@pytest.fixture
def router_falso(monkeypatch):
    import routers.chat as chat
    estado = {"cobros": [], "fondo": [], "guardados": [], "turnos": 0, "repetido": False}

    def guardar_usuario(sid, prompt, uid, adjuntos, client_message_id=None, vision_items=None, info=None):
        if info is not None:
            info["repetido"] = estado["repetido"]
        return "id-usuario"

    def turno(**kw):
        estado["turnos"] += 1
        yield _sse({"type": "chunk", "text": "Anotado."})
        yield _sse({"type": "done", "response": "Anotado."})

    monkeypatch.setenv("MEALFIT_CHAT_TURNO_EN_SERIE_S", "5")
    monkeypatch.setattr(chat, "StreamingResponse", lambda flujo, media_type=None: flujo)
    monkeypatch.setattr(chat, "get_or_create_session", lambda *a, **k: None)
    monkeypatch.setattr(chat, "save_message_with_attachments", guardar_usuario)
    monkeypatch.setattr(chat, "save_message", lambda sid, rol, texto, **k: estado["guardados"].append((rol, texto)))
    monkeypatch.setattr(chat, "merge_form_data_with_profile", lambda uid, fd: fd or {})
    monkeypatch.setattr(chat, "get_user_profile", lambda uid: {"plan_tier": "gratis"})
    monkeypatch.setattr(chat, "get_latest_meal_plan", lambda uid: None)
    monkeypatch.setattr(chat, "rechazar_si_menor", lambda *a, **k: None)
    monkeypatch.setattr(chat, "submit_bg_task", lambda fn, *a, task_name=None, **k: estado["fondo"].append(task_name))
    monkeypatch.setattr(chat, "log_api_usage", lambda uid, kind: estado["cobros"].append((uid, kind)))
    monkeypatch.setattr(chat, "_resolve_chat_local_time", lambda d, tz, uid: (d, tz))
    monkeypatch.setattr(chat, "chat_with_agent_stream", turno)
    import db_chat
    monkeypatch.setattr(db_chat, "get_session_owner", lambda sid: None)
    import turno_desacoplado
    monkeypatch.setattr(turno_desacoplado, "activo", lambda: False)
    import decision_triage
    monkeypatch.setattr(decision_triage, "iniciar_turno", lambda *a, **k: None)
    import aviso_respuesta_chat
    monkeypatch.setattr(aviso_respuesta_chat, "avisar_respuesta", lambda *a, **k: None)
    return chat, estado


def _enviar(chat):
    cuerpo = {"session_id": SID, "prompt": "me comí 2 huevos", "client_message_id": CMID}
    return [_obj(e) for e in chat.api_chat_stream(None, data=cuerpo, verified_user_id=UID, _ia=None)]


def test_el_reintento_repite_la_respuesta_sin_ejecutar_el_turno(router_falso, monkeypatch):
    chat, estado = router_falso
    primero = _enviar(chat)
    assert estado["turnos"] == 1 and estado["cobros"] == [(UID, "llm_chat")]
    assert ("model", "Anotado.") in estado["guardados"]

    estado["repetido"] = True                                  # el mismo client_message_id otra vez
    vistos = []
    monkeypatch.setattr(ctr, "respuesta_ya_dada", lambda sid, cmid, **k: vistos.append((sid, cmid)) or "Anotado.")
    segundo = _enviar(chat)

    assert vistos == [(SID, CMID)]
    assert estado["turnos"] == 1, "el modelo (y sus herramientas) no corren dos veces por el mismo mensaje"
    assert estado["cobros"] == [(UID, "llm_chat")], "el reintento no gasta otro mensaje de la cuota"
    assert estado["guardados"].count(("model", "Anotado.")) == 1, "ni deja una segunda respuesta en el historial"
    assert estado["fondo"].count("chat_sse_bg_tasks") == 1
    assert [e["type"] for e in segundo] == ["chunk", "done"] and all(e["replayed"] is True for e in segundo)
    assert segundo[1]["response"] == primero[1]["response"]


def test_un_mensaje_nuevo_no_consulta_nada(router_falso, monkeypatch):
    chat, estado = router_falso

    def no(*a, **k):
        raise AssertionError("un mensaje nuevo no paga la consulta de repetición")
    monkeypatch.setattr(ctr, "respuesta_ya_dada", no)
    assert [e["type"] for e in _enviar(chat)] == ["chunk", "done"]
    assert estado["turnos"] == 1


def test_reintento_con_el_primer_turno_aun_vivo_avisa_y_no_lanza_otro(router_falso, monkeypatch):
    chat, estado = router_falso
    estado["repetido"] = True
    monkeypatch.setattr(ctr, "respuesta_ya_dada", lambda *a, **k: None)
    import agent
    monkeypatch.setattr(agent, "is_turn_active", lambda sid: True)
    salida = _enviar(chat)
    assert salida == [{"type": "error", "code": "turn_in_progress", "message": ctr.MENSAJE_EN_CURSO}]
    assert estado["turnos"] == 0 and estado["cobros"] == []


def test_reintento_de_un_turno_que_murio_sin_responder_se_ejecuta(router_falso, monkeypatch):
    chat, estado = router_falso
    estado["repetido"] = True
    monkeypatch.setattr(ctr, "respuesta_ya_dada", lambda *a, **k: None)
    import agent
    monkeypatch.setattr(agent, "is_turn_active", lambda sid: False)
    assert [e["type"] for e in _enviar(chat)] == ["chunk", "done"]
    assert estado["turnos"] == 1 and estado["cobros"] == [(UID, "llm_chat")]


def test_el_router_decide_despues_de_esperar_al_turno_anterior():
    fuente = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
    serie = fuente.index('entrar_en_serie(session_id)')
    repetido = fuente.index('__import__("coach_turno_repetido").resolver(')
    turno = fuente.index("for chunk in chat_with_agent_stream(")
    assert serie < repetido < turno


# ───────────────────────────────────────────────────────────── el modelo repite la misma llamada en un turno
def test_la_telemetria_marca_la_llamada_repetida_sin_guardar_los_argumentos():
    from chat_phase_metrics import ChatTurnPhaseTracker
    import coach_tool_loop
    t = ChatTurnPhaseTracker(user_id=UID, session_id=SID)
    f1 = coach_tool_loop._firma({"name": "log_water_glass", "args": {"glasses": 1}})
    f2 = coach_tool_loop._firma({"name": "log_water_glass", "args": {"glasses": 2}})
    t.record_tool("log_water_glass", 10, firma=f1)
    t.record_tool("log_water_glass", 10, firma=f2)     # otros argumentos: no es una repetición
    t.record_tool("log_water_glass", 10, firma=f1)     # idéntica a la primera
    t.record_tool("check_hydration_today", 5)          # sin firma: como siempre
    assert [e.get("repeated", False) for e in t.tools] == [False, False, True, False]
    assert all(set(e) <= {"name", "duration_ms", "guard_blocked", "repeated"} for e in t.tools), "ni argumentos ni huella en la fila"
    fuente = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert 'firma=__import__("coach_tool_loop")._firma({"name": tool_name, "args": tool_args})' in fuente


# ───────────────────────────────────────────────────────────── el corte visto desde el teléfono
def test_registra_el_corte_del_cliente_sin_texto(monkeypatch):
    import db_core
    import chat_phase_metrics
    escritos = []
    monkeypatch.setattr(chat_phase_metrics, "phase_metrics_enabled", lambda: True)
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p=None, **k: escritos.append(p) or True)
    r = ctr.registrar_corte_del_cliente(user_id=UID, session_id=SID, payload={
        "event": "stream_cut", "outcome": "rescued", "attempts": 3, "waited_ms": 4200, "platform": "ios",
        "prompt": "no debe guardarse", "corr": "abc-123"})
    assert r == {"recorded": True, "outcome": "rescued"}
    (uid, sid, nodo, dur, meta), = escritos
    assert (sid, nodo, dur) == (SID, "chat_client_stream_cut", 4200)
    assert json.loads(meta) == {"corr": "abc-123", "outcome": "rescued", "attempts": 3, "waited_ms": 4200, "platform": "ios"}


@pytest.mark.parametrize("payload", [{}, {"outcome": "otra-cosa"}, {"outcome": None}, None])
def test_un_resultado_desconocido_no_se_guarda(payload, monkeypatch):
    import db_core
    monkeypatch.setattr(db_core, "execute_sql_write", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no escribe")))
    assert ctr.registrar_corte_del_cliente(user_id=UID, session_id=SID, payload=payload)["recorded"] is False


def test_valores_raros_y_base_caida_no_rompen(monkeypatch):
    import db_core
    import chat_phase_metrics
    import fail_open as fo
    monkeypatch.setattr(chat_phase_metrics, "phase_metrics_enabled", lambda: True)
    contados, escritos = [], []
    monkeypatch.setattr(fo, "fail_open", lambda sitio, exc, **k: contados.append(sitio))
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p=None, **k: escritos.append(p) or True)
    ctr.registrar_corte_del_cliente(user_id=None, session_id=SID, payload={"outcome": "dead", "attempts": True, "waited_ms": "x"})
    assert json.loads(escritos[0][4])["attempts"] == 0 and escritos[0][3] == 0
    ctr.registrar_corte_del_cliente(user_id=None, session_id=SID, payload={"outcome": "exhausted", "attempts": 9999})
    assert json.loads(escritos[1][4])["attempts"] == 100
    monkeypatch.setattr(db_core, "execute_sql_write", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db")))
    assert ctr.registrar_corte_del_cliente(user_id=UID, session_id=SID, payload={"outcome": "dead"}) == {"recorded": False, "outcome": "dead"}
    assert contados == ["chat_client_stream_cut_emit"]


def test_el_endpoint_de_metricas_enruta_el_corte():
    fuente = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
    i = fuente.index('@router.post("/client-metrics"')
    cuerpo = fuente[i:fuente.index("@router.post(", i + 10)]
    assert 'data.get("event") == "stream_cut"' in cuerpo
    assert "registrar_corte_del_cliente as record_client_chat_perf" in cuerpo
