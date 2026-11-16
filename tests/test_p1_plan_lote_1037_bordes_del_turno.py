"""[P1-PLAN-LOTE-1037 · A-34 · 2026-11-15] Casos de borde del turno del coach: un caso por cada uno, con su decisión.

1. tope diario × cuota mensual: el turno que el tope contesta sin modelo no se cobra ni lanza trabajo de fondo;
2. invitados: sin caché de contexto ni tope de gasto, y con un tope de turnos por IP y día;
3. cambio de día y de huso a mitad de sesión: el «hoy» es de cada turno y lo que manda el cliente se valida;
4. dos turnos a la vez en la misma sesión: van en serie.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import coach_bordes_del_turno as cbt  # noqa: E402

_ROUTER = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
UID = "11111111-2222-3333-4444-555555555555"


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# ───────────────────────────────────────────────────────────── 1. tope diario × cuota mensual
def test_la_marca_de_turno_sin_modelo():
    assert cbt.turno_sin_modelo({"type": "done", "budget_cap": True}) is True
    assert cbt.turno_sin_modelo({"type": "done"}) is False
    assert cbt.turno_sin_modelo({"type": "done", "budget_cap": "true"}) is False     # solo el booleano del servidor
    assert cbt.turno_sin_modelo(None) is False


def test_el_agente_marca_los_dos_eventos_del_tope(monkeypatch):
    """El turno bloqueado por el tope emite su texto fijo con `budget_cap` en el chunk Y en el done."""
    import agent
    import chat_hard_turn
    monkeypatch.setattr(chat_hard_turn, "check_user_daily_chat_budget",
                        lambda uid, **k: {"status": "hard_100", "block_turn": True, "allow_escalation": False})
    eventos = [json.loads(e[6:]) for e in agent.chat_with_agent_stream(session_id="s-1037", prompt="hola", user_id=UID)]
    assert [e["type"] for e in eventos] == ["chunk", "done"]
    assert all(e.get("budget_cap") is True for e in eventos)
    assert eventos[1]["response"] == eventos[0]["text"] and eventos[0]["text"].strip()
    assert agent.is_turn_active("s-1037") is False


@pytest.fixture
def router_falso(monkeypatch):
    """El endpoint del stream con todo lo de fuera simulado: devuelve (chat, cobros, fondo, guardados)."""
    import routers.chat as chat
    cobros, fondo, guardados = [], [], []
    monkeypatch.setenv("MEALFIT_CHAT_TURNO_EN_SERIE_S", "5")
    monkeypatch.setattr(chat, "StreamingResponse", lambda flujo, media_type=None: flujo)
    monkeypatch.setattr(chat, "get_or_create_session", lambda *a, **k: None)
    monkeypatch.setattr(chat, "save_message", lambda sid, rol, texto, **k: guardados.append((rol, texto)))
    monkeypatch.setattr(chat, "merge_form_data_with_profile", lambda uid, fd: fd or {})
    monkeypatch.setattr(chat, "get_user_profile", lambda uid: {"plan_tier": "gratis"})
    monkeypatch.setattr(chat, "get_latest_meal_plan", lambda uid: None)
    monkeypatch.setattr(chat, "rechazar_si_menor", lambda *a, **k: None)
    monkeypatch.setattr(chat, "submit_bg_task", lambda fn, *a, task_name=None, **k: fondo.append(task_name))
    monkeypatch.setattr(chat, "log_api_usage", lambda uid, kind: cobros.append((uid, kind)))
    monkeypatch.setattr(chat, "_resolve_chat_local_time", lambda d, tz, uid: (d, tz))
    import db_chat
    monkeypatch.setattr(db_chat, "get_session_owner", lambda sid: None)
    import turno_desacoplado
    monkeypatch.setattr(turno_desacoplado, "activo", lambda: False)
    import decision_triage
    monkeypatch.setattr(decision_triage, "iniciar_turno", lambda *a, **k: None)
    import aviso_respuesta_chat
    monkeypatch.setattr(aviso_respuesta_chat, "avisar_respuesta", lambda *a, **k: None)
    return chat, cobros, fondo, guardados


def _correr(chat, eventos, monkeypatch, session_id="s-1037", data=None, recibido=None):
    def falso(**kw):
        if recibido is not None:
            recibido.update(kw)
        yield from eventos
    monkeypatch.setattr(chat, "chat_with_agent_stream", falso)
    cuerpo = {"session_id": session_id, "prompt": "hola", **(data or {})}
    return list(chat.api_chat_stream(None, data=cuerpo, verified_user_id=UID, _ia=None))


def test_el_turno_del_tope_no_se_cobra_ni_lanza_trabajo_de_fondo(router_falso, monkeypatch):
    chat, cobros, fondo, guardados = router_falso
    texto = "Llegaste al tope de hoy."
    salida = _correr(chat, [_sse({"type": "chunk", "text": texto, "budget_cap": True}),
                            _sse({"type": "done", "response": texto, "budget_cap": True})], monkeypatch)
    assert len(salida) == 2                                    # el usuario SÍ recibe el aviso
    assert ("model", texto) in guardados                       # y queda en el historial
    assert cobros == [], "un turno sin modelo no puede gastar un mensaje de la cuota mensual"
    assert "chat_sse_bg_tasks" not in fondo, "sin modelo no hay hechos que extraer (y esa extracción gasta IA)"


def test_el_turno_normal_se_sigue_cobrando_una_vez(router_falso, monkeypatch):
    chat, cobros, fondo, _ = router_falso
    _correr(chat, [_sse({"type": "chunk", "text": "Hola"}), _sse({"type": "done", "response": "Hola"})], monkeypatch)
    assert cobros == [(UID, "llm_chat")]
    assert "chat_sse_bg_tasks" in fondo


def test_un_done_normal_sin_chunks_tambien_se_cobra(router_falso, monkeypatch):
    chat, cobros, _, _ = router_falso
    _correr(chat, [_sse({"type": "done", "response": "Hola"})], monkeypatch)
    assert cobros == [(UID, "llm_chat")]


# ───────────────────────────────────────────────────────────── 2. invitados
def test_invitado_sin_cache_de_contexto_ni_tope_de_gasto(monkeypatch):
    import agent
    import chat_hard_turn
    monkeypatch.setenv("MEALFIT_CHAT_CTX_CACHE_TTL_S", "30")
    agent.invalidate_chat_user_ctx_cache(None)
    veces = []
    for _ in range(2):
        agent._ctx_cache_get_or_load("guest", "pantry", (), lambda: veces.append(1) or "x")
    assert len(veces) == 2, "el invitado no comparte caché: cada turno lee de nuevo"
    assert agent._CHAT_USER_CTX_CACHE == {}

    def no_debe_consultar(*a, **k):
        raise AssertionError("el tope de gasto no consulta la base para un invitado")
    info = chat_hard_turn.check_user_daily_chat_budget("guest", execute_sql_query=no_debe_consultar)
    assert info["block_turn"] is False


def test_invitado_tope_de_turnos_por_ip_y_dia(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_GUEST_TURNS_PER_DAY", "3")
    monkeypatch.setattr(cbt, "_TURNOS_DE_INVITADO", {})
    t0 = 1_000_000.0
    assert [cbt.invitado_puede("1.1.1.1", t0 + i) for i in range(5)] == [True, True, True, False, False]
    assert cbt.invitado_puede("2.2.2.2", t0) is True, "cada IP tiene su cupo"
    assert len(cbt._TURNOS_DE_INVITADO["1.1.1.1"]) == 3, "el rechazado no se anota: insistir no alarga el castigo"
    assert cbt.invitado_puede("1.1.1.1", t0 + 86_400 + 1) is True, "a las 24 h del primero vuelve a caber uno"
    assert cbt.invitado_puede(None, t0) is True                # sin IP no se puede contar


def test_invitado_en_su_tope_recibe_429_y_la_cuenta_no(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_GUEST_TURNS_PER_DAY", "1")
    monkeypatch.setattr(cbt, "_TURNOS_DE_INVITADO", {})
    cbt.exigir_cupo_de_invitado("9.9.9.9", None)
    with pytest.raises(HTTPException) as exc:
        cbt.exigir_cupo_de_invitado("9.9.9.9", None)
    assert exc.value.status_code == 429 and exc.value.headers["X-Coach-Guest-Cap"] == "1"
    for _ in range(5):
        cbt.exigir_cupo_de_invitado("9.9.9.9", UID)           # con cuenta manda su cuota, no este tope
    monkeypatch.setenv("MEALFIT_CHAT_GUEST_TURNS_PER_DAY", "0")
    cbt.exigir_cupo_de_invitado("9.9.9.9", None)               # 0 = sin tope


def test_en_las_pruebas_el_tope_de_invitado_esta_apagado_salvo_que_se_pida(monkeypatch):
    monkeypatch.delenv("MEALFIT_CHAT_GUEST_TURNS_PER_DAY", raising=False)
    assert cbt.tope_de_invitado() == 0, "una suite que hace cientos de turnos de invitado no puede toparse con el 429"
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert cbt.tope_de_invitado() == 40


def test_los_dos_endpoints_del_chat_exigen_el_cupo_de_invitado():
    assert '@router.post("/stream", dependencies=[Depends(_CHAT_STREAM_LIMITER), Depends(_cupo_de_invitado)])' in _ROUTER
    assert '@router.post("", dependencies=[Depends(_CHAT_STREAM_LIMITER), Depends(_cupo_de_invitado)])' in _ROUTER


# ───────────────────────────────────────────────────────────── 3. cambio de día y de huso
MEDIODIA = datetime(2026, 11, 15, 16, 0, tzinfo=timezone.utc)      # 12:00 en RD (240)


@pytest.mark.parametrize("crudo,esperado", [
    (240, 240), (0, 0), (-540, -540), ("240", 240), (240.0, 240), (840, 840), (-840, -840),
    (841, None), (1731686400, None), (True, None), ("mañana", None), (None, None), ({}, None), (float("nan"), None),
])
def test_huso_del_cliente(crudo, esperado):
    assert cbt.huso_del_cliente(crudo) == esperado


@pytest.mark.parametrize("fecha,vale", [
    ("2026-11-15", True), ("2026-11-14", True), ("2026-11-16", True),     # hoy y un día de margen (viaje, rollover 04:00)
    ("2026-11-13", False), ("2026-11-17", False), ("2020-01-01", False),  # pestaña vieja o cliente roto
    ("2026-13-40", False), ("15/11/2026", False), ("", False), (20261115, False), (None, False),
])
def test_fecha_del_cliente(fecha, vale):
    assert cbt.fecha_del_cliente(fecha, 240, MEDIODIA) == (fecha if vale else None)


def test_cambio_de_dia_a_mitad_de_sesion():
    """El «hoy» es de CADA turno: el mismo chat, antes y después del rollover de las 04:00, lleva fechas distintas."""
    antes = datetime(2026, 11, 16, 7, 30, tzinfo=timezone.utc)      # 03:30 en RD: sigue siendo el día 15
    despues = datetime(2026, 11, 16, 8, 30, tzinfo=timezone.utc)    # 04:30 en RD: ya es el 16
    assert cbt.fecha_y_huso_del_turno("2026-11-15", 240, antes) == ("2026-11-15", 240)
    assert cbt.fecha_y_huso_del_turno("2026-11-16", 240, despues) == ("2026-11-16", 240)
    # la pestaña quedó abierta dos días: su fecha se descarta y la deriva el servidor
    assert cbt.fecha_y_huso_del_turno("2026-11-13", 240, despues) == (None, 240)


def test_cambio_de_huso_a_mitad_de_sesion():
    """Viaja de Santo Domingo a Madrid con el chat abierto: manda su huso nuevo y su fecha nueva, y valen."""
    noche = datetime(2026, 11, 15, 23, 30, tzinfo=timezone.utc)     # 19:30 en RD (día 15), 00:30 en Madrid
    assert cbt.fecha_y_huso_del_turno("2026-11-15", 240, noche) == ("2026-11-15", 240)
    assert cbt.fecha_y_huso_del_turno("2026-11-15", -60, noche) == ("2026-11-15", -60)
    assert cbt.fecha_y_huso_del_turno("2026-11-16", -60, noche) == ("2026-11-16", -60)
    # huso imposible: se descarta el huso (lo pone el perfil) y la fecha se juzga con el huso por defecto
    assert cbt.fecha_y_huso_del_turno("2026-11-15", 99999, noche) == ("2026-11-15", None)


def test_el_router_sanea_antes_de_rellenar_y_el_turno_recibe_lo_saneado(router_falso, monkeypatch):
    chat, _, _, _ = router_falso
    recibido = {}
    _correr(chat, [_sse({"type": "done", "response": "ok"})], monkeypatch,
            data={"local_date": "2020-01-01", "tz_offset": 99999}, recibido=recibido)
    assert recibido["local_date"] is None and recibido["tz_offset"] is None
    # y en el código, el saneado va ANTES del relleno desde el perfil, en los dos endpoints
    assert _ROUTER.count("fecha_y_huso_del_turno(local_date, tz_offset)") == 2
    for trozo in _ROUTER.split("fecha_y_huso_del_turno(local_date, tz_offset)")[1:]:
        assert "_resolve_chat_local_time(local_date, tz_offset, verified_user_id)" in trozo[:400]


def test_la_cache_de_contexto_lleva_el_dia_en_la_clave():
    """Lo que depende del día (diario de hoy, agua) no puede servirse del día anterior tras el cambio."""
    assert '"consumed_today",\n                (self.local_date, self.tz_diario),' in _AGENT
    assert '"consumed_today",\n                        (local_date, _tz_diario),' in _AGENT
    assert '"hydration", (local_date_str,)' in _AGENT


# ───────────────────────────────────────────────────────────── 4. dos turnos a la vez en la misma sesión
def test_dos_turnos_de_la_misma_sesion_van_en_serie():
    orden, dentro = [], threading.Event()

    def turno(nombre, espera):
        paso = cbt.entrar_en_serie("s-serie", espera_s=5)
        try:
            orden.append(f"{nombre}:entra")
            dentro.set()
            time.sleep(espera)
            orden.append(f"{nombre}:sale")
        finally:
            paso.salir()

    a = threading.Thread(target=turno, args=("A", 0.25))
    a.start()
    assert dentro.wait(2)
    b = threading.Thread(target=turno, args=("B", 0.0))
    b.start()
    a.join(5)
    b.join(5)
    assert orden == ["A:entra", "A:sale", "B:entra", "B:sale"]
    assert cbt._SESIONES == {}, "no queda nada apuntado cuando no hay turnos"


def test_sesiones_distintas_no_se_esperan():
    uno = cbt.entrar_en_serie("s-uno", espera_s=5)
    t0 = time.monotonic()
    otro = cbt.entrar_en_serie("s-otro", espera_s=5)
    assert otro.tomado and time.monotonic() - t0 < 0.5
    uno.salir()
    otro.salir()
    otro.salir()                                               # idempotente
    assert cbt._SESIONES == {}


def test_si_el_turno_anterior_no_termina_el_segundo_sigue_y_queda_contado(monkeypatch):
    import fail_open as fo
    vistos = []
    monkeypatch.setattr(fo, "fail_open", lambda sitio, exc, **k: vistos.append(sitio))
    colgado = cbt.entrar_en_serie("s-colgada", espera_s=5)
    segundo = cbt.entrar_en_serie("s-colgada", espera_s=0.1)
    assert segundo.tomado is False and vistos == ["chat_turno_en_serie"]
    segundo.salir()                                            # no suelta lo que no tomó
    tercero_no_entra = cbt._SESIONES["s-colgada"][0].acquire(blocking=False)
    assert tercero_no_entra is False
    colgado.salir()
    assert cbt._SESIONES == {}


def test_apagado_por_knob_y_sin_sesion(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_TURNO_EN_SERIE_S", "0")
    a, b = cbt.entrar_en_serie("s-off"), cbt.entrar_en_serie("s-off")
    assert a.tomado is False and b.tomado is False and cbt._SESIONES == {}
    a.salir()
    b.salir()
    assert cbt.entrar_en_serie(None, espera_s=5).tomado is False


def test_el_router_suelta_el_turno_aunque_el_cliente_corte(router_falso, monkeypatch):
    chat, _, _, _ = router_falso

    def turno(**kw):
        yield _sse({"type": "chunk", "text": "Ho"})
        yield _sse({"type": "chunk", "text": "la"})
    monkeypatch.setattr(chat, "chat_with_agent_stream", turno)
    flujo = chat.api_chat_stream(None, data={"session_id": "s-corte", "prompt": "hola"}, verified_user_id=UID, _ia=None)
    next(flujo)
    assert "s-corte" in cbt._SESIONES
    flujo.close()                                              # el cliente se fue a mitad
    assert cbt._SESIONES == {}
    # y el segundo envío de la misma sesión entra sin esperar
    assert len(_correr(chat, [_sse({"type": "done", "response": "ok"})], monkeypatch, session_id="s-corte")) == 1
