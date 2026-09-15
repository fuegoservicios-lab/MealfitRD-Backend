"""[Auditoría del coach · 2026-09-14] Núcleo del agente de chat (`agent.py`, `memory_manager.py`).

Cada bloque cierra un hallazgo verificado leyendo el código:

- P1-CHAT-ORPHAN-TOOLCALL-SANITIZE — un tool_call sin ToolMessage (tool que lanza, cliente
  que corta, purga por número) dejaba la sesión inservible: el proveedor rechaza ese
  historial en cada turno siguiente.
- P1-CHAT-CB-CLIENT-ERRORS — un 4xx de UNA sesión rota abría el breaker compartido de todos.
- P1-DIARY-NUDGE-VISIBLE — el nudge del diario era un SystemMessage que `call_model`
  descartaba: el reintento repetía la misma entrada.
- P0-CHAT-ALLERGY-FRONTEND-TRUTH — el navegador recibía la lista nueva de alergias SOLA
  (antes de ejecutar la tool) y la sincronización del Dashboard borraba las previas.
- P2-CHAT-SINGLE-ERROR-EVENT — dos eventos `error` por timeout, con `str(e)` crudo.
- P2-CHAT-STATE-TURN-RESET — avisos y «agotados» de turnos viejos reenviados en cada `done`.
- P2-CHAT-NONSTREAM-TIMEOUT-REAL — el `with ThreadPoolExecutor` esperaba al grafo.
- P1-CHAT-STREAM-TOOLCALL-CHUNKS / P1-CHAT-STREAM-NODE-FILTER — el guard de deliberación
  nunca veía la tool_call y el texto interno de las tools llegaba al usuario.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import agent as A  # noqa: E402
import memory_manager as MM  # noqa: E402

_AGENT_SRC = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_ROUTER_SRC = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")


def _ai_tc(*ids, name="log_consumed_meal", content=""):
    return AIMessage(content=content, tool_calls=[
        {"name": name, "args": {}, "id": i, "type": "tool_call"} for i in ids
    ])


def _assert_historial_valido(msgs):
    """Invariante del proveedor: cada tool_call tiene su ToolMessage justo detrás, y ningún
    ToolMessage aparece sin su tool_call."""
    abiertos = []
    for m in msgs:
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in abiertos, f"ToolMessage huérfano {m.tool_call_id}"
            abiertos.remove(m.tool_call_id)
            continue
        assert not abiertos, f"tool_calls sin respuesta: {abiertos}"
        if isinstance(m, AIMessage):
            abiertos = [tc["id"] for tc in (m.tool_calls or [])]
    assert not abiertos


# ───────────────────────── P1-CHAT-ORPHAN-TOOLCALL-SANITIZE ─────────────────────────

def test_tool_call_sin_respuesta_recibe_una_sintetica_honesta():
    msgs = [HumanMessage(content="cené pan"), _ai_tc("c1"), HumanMessage(content="¿y?")]
    out = A._sanitize_tool_call_history(msgs)
    _assert_historial_valido(out)
    sint = [m for m in out if isinstance(m, ToolMessage)]
    assert len(sint) == 1 and "TOOL_INTERRUPTED" in sint[0].content


def test_tool_message_huerfano_se_descarta():
    """El caso de la purga: sobrevive el ToolMessage, no su AIMessage."""
    msgs = [ToolMessage(content="ok", tool_call_id="viejo"), HumanMessage(content="hola")]
    out = A._sanitize_tool_call_history(msgs)
    _assert_historial_valido(out)
    assert not any(isinstance(m, ToolMessage) for m in out)


def test_historial_sano_no_se_toca():
    msgs = [
        HumanMessage(content="a"), _ai_tc("c1", "c2"),
        ToolMessage(content="r1", tool_call_id="c1"), ToolMessage(content="r2", tool_call_id="c2"),
        AIMessage(content="listo"),
    ]
    out = A._sanitize_tool_call_history(msgs)
    assert [type(m) for m in out] == [type(m) for m in msgs]
    assert [getattr(m, "content", None) for m in out] == [m.content for m in msgs]


def test_respuesta_parcial_completa_solo_lo_que_falta():
    msgs = [HumanMessage(content="a"), _ai_tc("c1", "c2"), ToolMessage(content="r1", tool_call_id="c1")]
    out = A._sanitize_tool_call_history(msgs)
    _assert_historial_valido(out)
    assert [m.content for m in out if isinstance(m, ToolMessage)][0] == "r1"


def test_call_model_usa_el_payload_saneado():
    i = _AGENT_SRC.index("def call_model(")
    body = _AGENT_SRC[i:_AGENT_SRC.index("def _form_field_updates_after_write(")]
    assert "_llm_messages_from_state(messages, sys_prompt)" in body


class _ToolQueRevienta:
    name = "check_hydration_today"

    def invoke(self, args):
        raise RuntimeError("SELECT ... secreto interno")


def test_una_tool_que_lanza_responde_con_tool_error_y_no_tumba_el_turno(monkeypatch):
    monkeypatch.setattr(A, "agent_tools", [_ToolQueRevienta()])
    state = {
        "messages": [HumanMessage(content="¿cuánta agua?"),
                     _ai_tc("c9", name="check_hydration_today")],
        "user_id": "u-1", "session_id": "s-1",
    }
    out = A.execute_tools(state)
    tms = out["messages"]
    assert len(tms) == 1 and isinstance(tms[0], ToolMessage) and tms[0].tool_call_id == "c9"
    assert "TOOL_ERROR" in tms[0].content
    assert "secreto interno" not in tms[0].content, "el detalle interno llegó al modelo"


# ───────────────────────── purga alineada a turnos ─────────────────────────

def test_la_purga_nunca_empieza_por_un_tool_message():
    msgs = [HumanMessage(content="h0"), _ai_tc("c1"), ToolMessage(content="r", tool_call_id="c1"),
            AIMessage(content="ok"), HumanMessage(content="h1"), AIMessage(content="ok1")]
    # keep_recent=4 cortaría justo en el ToolMessage (índice 2)
    corte = MM._purge_cut_index(msgs, 4)
    assert isinstance(msgs[corte], HumanMessage) or corte == 0
    _assert_historial_valido(msgs[corte:])


def test_la_purga_sin_humano_disponible_no_purga():
    msgs = [_ai_tc("c1"), ToolMessage(content="r", tool_call_id="c1"), AIMessage(content="x")]
    assert MM._purge_cut_index(msgs, 1) == 0


# ───────────────────────── P1-CHAT-CB-CLIENT-ERRORS ─────────────────────────

class _Exc(Exception):
    def __init__(self, status):
        super().__init__("x")
        self.status_code = status


@pytest.mark.parametrize("status,esperado", [
    (400, True), (404, True), (422, True), (408, False), (429, False), (500, False), (None, False),
])
def test_clasificacion_de_errores_de_cliente(status, esperado):
    assert A._is_client_request_error(_Exc(status)) is esperado


def test_el_4xx_se_descarta_antes_de_contar_en_el_breaker():
    i = _AGENT_SRC.index("def call_model(")
    body = _AGENT_SRC[i:_AGENT_SRC.index("def _form_field_updates_after_write(")]
    assert body.index("_is_client_request_error(_invoke_exc)") < body.index("_cb.record_failure()")


# ───────────────────────── P1-DIARY-NUDGE-VISIBLE ─────────────────────────

def test_el_nudge_de_este_turno_llega_al_modelo_y_los_viejos_no():
    viejo = SystemMessage(content="exigencia caducada")
    msgs = [HumanMessage(content="t1"), AIMessage(content="r1"), viejo,
            HumanMessage(content="cené pan"), AIMessage(content="Cena registrada."),
            SystemMessage(content="ALTO. No llamaste a log_consumed_meal")]
    payload = A._llm_messages_from_state(msgs, "SYS")
    textos = [m.content for m in payload]
    assert textos[0] == "SYS" and isinstance(payload[0], SystemMessage)
    assert any("ALTO. No llamaste" in t for t in textos), "el nudge no llega al modelo"
    assert not any("exigencia caducada" in t for t in textos)
    # solo hay UN system: el prompt (el nudge viaja con rol de usuario)
    assert sum(isinstance(m, SystemMessage) for m in payload) == 1


# ───────────────────────── P0-CHAT-ALLERGY-FRONTEND-TRUTH ─────────────────────────

def test_alergias_se_propagan_con_el_valor_fusionado_de_la_base(monkeypatch):
    monkeypatch.setattr(A, "get_user_profile", lambda uid: {
        "health_profile": {"allergies": ["Mariscos", "Frutos Secos", "Lacteos"]}})
    assert A._form_field_updates_after_write("allergies", ["Lacteos"], "u1") == {
        "allergies": ["Mariscos", "Frutos Secos", "Lacteos"]}


def test_si_la_relectura_falla_se_envia_la_nueva_y_el_frontend_une(monkeypatch):
    def _boom(uid):
        raise RuntimeError("db")
    monkeypatch.setattr(A, "get_user_profile", _boom)
    assert A._form_field_updates_after_write("allergies", ["Lacteos"], "u1") == {"allergies": ["Lacteos"]}


def test_select_viaja_con_el_valor_canonico_guardado_no_con_el_crudo(monkeypatch):
    """«ganar peso» crudo en el formulario lo escribiría el PATCH del Dashboard encima de
    `gain_muscle` (la tool canoniza desde P1-CHAT-TOOLS-AUDIT)."""
    monkeypatch.setattr(A, "get_user_profile", lambda uid: {"health_profile": {"mainGoal": "gain_muscle"}})
    assert A._form_field_updates_after_write("mainGoal", "ganar peso", "u1") == {"mainGoal": "gain_muscle"}


def test_el_peso_arrastra_su_unidad(monkeypatch):
    monkeypatch.setattr(A, "get_user_profile", lambda uid: {
        "health_profile": {"weight": "176.4", "weightUnit": "lb"}})
    assert A._form_field_updates_after_write("weight", "80", "u1") == {"weight": "176.4", "weightUnit": "lb"}


def test_invitado_canoniza_sin_leer_la_base(monkeypatch):
    monkeypatch.setattr(A, "get_user_profile", lambda uid: pytest.fail("un invitado no tiene perfil"))
    assert A._form_field_updates_after_write("mainGoal", "ganar peso", None, raw_value="ganar peso") == {
        "mainGoal": "gain_muscle"}


class _UFF:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, args):
        self.calls.append(dict(args))
        return self.result


def _estado_uff(field, value, uid="u-1"):
    return {
        "messages": [HumanMessage(content="x"), AIMessage(content="", tool_calls=[{
            "name": "update_form_field", "args": {"field": field, "new_value": value},
            "id": "t1", "type": "tool_call"}])],
        "user_id": uid, "session_id": "s-1", "updated_fields": {},
    }


def test_valor_rechazado_por_la_tool_no_llega_al_formulario(monkeypatch):
    monkeypatch.setattr(A, "update_form_field", _UFF("No pude guardar: 'keto' no es una dieta válida."))
    out = A.execute_tools(_estado_uff("dietType", "keto"))
    assert "dietType" not in out["updated_fields"], (
        "un valor que la tool RECHAZÓ llegó al formulario — la sincronización del Dashboard lo persistiría"
    )


def test_alergia_aceptada_viaja_con_la_union(monkeypatch):
    monkeypatch.setattr(A, "update_form_field", _UFF("¡Éxito! El campo 'allergies' ha sido actualizado."))
    monkeypatch.setattr(A, "get_user_profile", lambda uid: {
        "health_profile": {"allergies": ["Mariscos", "Lacteos"]}})
    out = A.execute_tools(_estado_uff("allergies", "Lacteos"))
    assert out["updated_fields"]["allergies"] == ["Mariscos", "Lacteos"]


def test_invitado_valida_con_la_tool_como_guest_y_llena_el_formulario(monkeypatch):
    fake = _UFF("¡Éxito! El campo 'weight' ha sido actualizado a '80' (solo en el formulario).")
    monkeypatch.setattr(A, "update_form_field", fake)
    out = A.execute_tools(_estado_uff("weight", "80", uid="guest"))
    assert fake.calls and fake.calls[0]["user_id"] == "guest", (
        "la tool debe recibir 'guest' para validar sin escribir (no el session_id)"
    )
    assert out["updated_fields"]["weight"] == "80"


def test_invitado_con_valor_rechazado_no_llena_el_formulario(monkeypatch):
    monkeypatch.setattr(A, "update_form_field", _UFF("No actualicé 'age': una edad de 300 años..."))
    out = A.execute_tools(_estado_uff("age", "300", uid="guest"))
    assert "age" not in out["updated_fields"]


# ───────────────────────── eventos de error y estado del turno ─────────────────────────

def test_el_evento_de_error_no_lleva_detalle_interno():
    p = A._chat_stream_error_payload(RuntimeError("psycopg: relation user_facts SELECT ..."))
    assert p["type"] == "error" and p["code"] == "internal"
    assert "psycopg" not in p["message"]
    assert A._chat_stream_error_payload(TimeoutError("x"))["code"] == "timeout"


def test_un_solo_evento_error_por_turno():
    assert "if not _error_emitted:" in _AGENT_SRC
    assert "{'type': 'error', 'message': str(e)}" not in _AGENT_SRC
    assert "'message': str(e)" not in _ROUTER_SRC
    assert "if not _error_seen:" in _ROUTER_SRC


@pytest.mark.parametrize("campo", [
    '"coherence_warnings": []', '"pantry_modified_at": None',
    '"pantry_depleted_items": None', '"diary_claim_retried": False',
])
def test_los_campos_del_turno_se_reinician_en_inputs(campo):
    i = _AGENT_SRC.index("existing_state = chat_graph_app.get_state(config)\n    \n    inputs = {")
    assert campo in _AGENT_SRC[i:i + 1600]


def test_el_timeout_no_stream_no_espera_al_grafo():
    i = _AGENT_SRC.index("_graph_timeout_s = _chat_graph_total_timeout_s()")
    ventana = _AGENT_SRC[i:i + 2500]
    assert "_ex.shutdown(wait=False)" in ventana
    assert "with concurrent.futures.ThreadPoolExecutor(" not in ventana


def test_el_stream_detecta_chunks_de_tool_y_filtra_el_nodo():
    assert 'getattr(msg_chunk, "tool_call_chunks", None)' in _AGENT_SRC
    assert '_node != "call_model"' in _AGENT_SRC
    # el progress sale una vez por mensaje, no por chunk
    assert "_progress_keys.add(_progress_key)" in _AGENT_SRC


def test_dias_pasados_anclan_en_el_huso_del_usuario():
    """[P2-CHAT-PAST-DAYS-USER-TZ · 2026-09-14] 23:30 UTC del 27-jul es el 27 en RD
    (UTC-4) pero ya el 28 en Madrid (UTC+2, offset JS = -120)."""
    from datetime import date
    from chat_history_context import resolve_day_dates
    plan = {"days": [{"day_name": "x", "meals": []}], "grocery_start_date": "2026-07-27T23:30:00+00:00"}
    hoy = date(2026, 7, 30)
    assert resolve_day_dates(plan, hoy)[0]["date"] == date(2026, 7, 27)
    assert resolve_day_dates(plan, hoy, -120)[0]["date"] == date(2026, 7, 28)
    i = _AGENT_SRC.index("out = build_past_plan_days_block(")
    assert "tz_offset_mins=tz_offset_mins" in _AGENT_SRC[i:i + 200]


def test_la_inactividad_a_posteriori_ya_no_aborta_un_turno_vivo():
    i = _AGENT_SRC.index("if _gap_since_last > _stream_inactivity_budget:")
    bloque = _AGENT_SRC[i:i + 1400]
    assert "raise TimeoutError" not in bloque.split("_last_event_at = _now")[0]
