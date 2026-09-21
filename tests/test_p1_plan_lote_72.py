# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-72 · 2026-09-16] El recordatorio de una comida no se pierde porque otra comida se registró tarde.

El dueño recibió a las 10:30 el recordatorio del desayuno («Ya son pasadas las 10:30 AM y tu desayuno sigue pendiente de
registro») y lo contestó a las 12:58 registrando el desayuno. El del almuerzo, que tocaba a las 14:30, no llegó nunca.
Forense de solo lectura: un solo aviso en `nudge_outcomes` (Desayuno, 14:30Z) y el desayuno en `consumed_meals` con
`consumed_at` = 16:58Z, la hora del REGISTRO.

La cadena:
  1. `get_avg_meal_hour` promedia `consumed_at`: con ese único registro, «el dueño desayuna a las 12:58».
  2. El aviso del desayuno pasó a 12:58 + 1,5 h = 14:28 → la hora 14, la MISMA que el del almuerzo (13:00 + 1,5).
  3. El bucle toma la PRIMERA comida que coincide con la hora (desayuno), ve que ya está registrada y `continue`:
     salta la hora entera sin mirar el almuerzo.
  4. Y mañana el aviso del desayuno llegaría a las 14:30 — y a las 10:30 no llegaría nada.

Arreglo: (a) un registro fuera de la franja de su comida (un desayuno a las 12:58) no mueve la hora del aviso; (b) se
consideran todas las comidas que tocan y se avisa la que falte; (c) un aviso que no pudo salir en su hora (conversación
en curso, IA caída, reinicio) se reintenta durante unas horas, sin repetir uno ya enviado hoy.
"""
from __future__ import annotations

import re
from datetime import datetime as _dt_real, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

UID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
SID = "11111111-2222-4333-8444-555555555555"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _fila(hora: str) -> dict:
    h, m = hora.split(":")
    return {"hr": int(h), "mn": int(m)}


@pytest.fixture
def dia(monkeypatch):
    """El día del dueño con todas las dependencias de `run_proactive_checks` simuladas. Lo que decide el disparo
    —`get_avg_meal_hour` con su SQL, la hora local, el bucle— es el código real."""
    import db_facts
    import proactive_agent as pa
    import utils_push

    estado = {
        "ahora": _dt_real(2026, 9, 16, 18, 30, 5, tzinfo=timezone.utc),   # 14:30 en RD
        "registros": {"Desayuno": ["12:58"]},                             # hora LOCAL de cada consumed_at
        "comidas": [{"meal_type": "desayuno", "meal_name": "3 huevos revueltos con empanada pequeña de pollo"}],
        "avisos_hoy": ["Desayuno"],
        "mensajes": [
            {"role": "model", "created_at": "2026-09-16 16:58:13.337404+00"},
            {"role": "user", "created_at": "2026-09-16 16:57:56.25274+00"},
            {"role": "model", "created_at": "2026-09-16 14:30:18.74015+00"},
        ],
        "guardados": [],
        "avisos_nuevos": [],
        "push": [],
        "prompts": [],
    }

    class _Reloj(_dt_real):
        @classmethod
        def now(cls, tz=None):
            return estado["ahora"] if tz else estado["ahora"].replace(tzinfo=None)

    def _sql_avg(query, params, fetch_all=False):
        comida = str(params[3]).strip("%")
        return [_fila(h) for h in estado["registros"].get(comida, [])]

    def _sql_pa(query, params=None, fetch_all=False, fetch_one=False):
        if "nudge_outcomes" in query:
            return [{"nudge_type": t} for t in estado["avisos_hoy"]]
        return []

    class _LLM:
        def __init__(self, *a, **k):
            pass

        def invoke(self, prompt):
            estado["prompts"].append(prompt)
            m = re.search(r"aún no ha registrado su (.+?)\.", prompt)
            comida = m.group(1) if m else "resumen"

            class _R:
                content = f"Recordatorio: {comida}"
            return _R()

    def _log(user_id, nudge_type, **k):
        estado["avisos_nuevos"].append(nudge_type)

    monkeypatch.setattr(db_facts, "execute_sql_query", _sql_avg)
    monkeypatch.setattr(db_facts, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(db_facts, "connection_pool", object())
    monkeypatch.setattr(pa, "datetime", _Reloj)
    monkeypatch.setattr(pa, "get_active_users_for_proactive", lambda: [{"id": SID, "user_id": UID}])
    monkeypatch.setattr(pa, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(pa, "get_daily_nudge_count", lambda _u: len(estado["avisos_hoy"]))
    monkeypatch.setattr(pa, "get_nudge_response_rate", lambda _u, nudge_type=None: (1.0, 0))
    monkeypatch.setattr(pa, "get_recent_messages", lambda _s, limit=5: list(estado["mensajes"])[:limit])
    monkeypatch.setattr(pa, "get_user_profile", lambda _u: {
        "health_profile": {"goals": ["ganar músculo"], "dietTypes": ["balanceada"]}, "locale": "es-DO"})
    monkeypatch.setattr(pa, "get_consumed_meals_today", lambda _u, date_str=None, tz_offset_mins=None: list(estado["comidas"]))
    monkeypatch.setattr(pa, "_usuario_en_modo_contador", lambda _u: True)
    monkeypatch.setattr(pa, "get_embedding", lambda _t: None)
    monkeypatch.setattr(pa, "get_best_nudge_style", lambda _u: "gamificado")
    monkeypatch.setattr(pa, "execute_sql_query", _sql_pa)
    monkeypatch.setattr(pa, "ChatGLM", _LLM)
    monkeypatch.setattr(pa, "save_message", lambda s, r, c: estado["guardados"].append((s, r, c)))
    monkeypatch.setattr(pa, "log_nudge_outcome", _log)
    monkeypatch.setattr(utils_push, "send_push_notification", lambda **k: estado["push"].append(k) or True)
    monkeypatch.delenv("MEALFIT_PROACTIVE_MAX_NUDGES_PER_DAY", raising=False)
    monkeypatch.delenv("MEALFIT_PROACTIVE_NUDGE_RETRY_HOURS", raising=False)
    estado["correr"] = pa.run_proactive_checks
    return estado


def _utc(dia_mes: int, hora: int, minuto: int = 30) -> _dt_real:
    return _dt_real(2026, 9, dia_mes, hora, minuto, 5, tzinfo=timezone.utc)


# ── El caso del dueño ─────────────────────────────────────────────────────────────────────────────

def test_el_almuerzo_de_las_1430_sale_aunque_el_desayuno_se_registrara_a_las_1258(dia):
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Almuerzo"], (
        "el recordatorio del almuerzo se pierde: el desayuno registrado tarde le roba la hora"
    )
    assert dia["guardados"] == [(SID, "model", "Recordatorio: Almuerzo")]
    assert "12:45 PM" in dia["prompts"][0], "[P1-PLAN-LOTE-150] su hora, ya no 1,5 h después"


def test_manana_el_desayuno_se_recuerda_a_su_hora_y_no_a_las_1430(dia):
    dia.update(ahora=_utc(17, 14), comidas=[], avisos_hoy=[], mensajes=[])
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Desayuno"], "un registro a las 12:58 movió el aviso del desayuno a la tarde"
    # 10:30 sigue dentro de su ventana de reintento (8-10); la hora que dice el mensaje es la suya: 8:45.
    assert "8:45 AM" in dia["prompts"][0]

    dia.update(ahora=_utc(17, 18), avisos_hoy=[], avisos_nuevos=[])
    dia["comidas"] = [{"meal_type": "almuerzo", "meal_name": "arroz con habichuelas"}]
    dia["correr"]()
    assert dia["avisos_nuevos"] == [], "a las 14:30 no se pregunta por el desayuno"


def test_un_registro_en_su_franja_si_personaliza_la_hora(dia):
    # Desayuna a las 7:05 de verdad: su aviso es a las 6:50, no el de las 8:45 de quien no tiene historial.
    dia.update(ahora=_utc(17, 12), registros={"Desayuno": ["07:00", "07:10"]}, comidas=[], avisos_hoy=[], mensajes=[])
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Desayuno"]
    assert "6:50 AM" in dia["prompts"][0]


def test_dos_comidas_en_la_misma_hora_se_avisa_la_que_falta(dia):
    # Desayuno ~11:45 y almuerzo ~12:15, ambos reales: los dos avisos caen en la hora 13.
    dia.update(ahora=_utc(16, 17), registros={"Desayuno": ["11:40", "11:50"], "Almuerzo": ["12:10", "12:20"]},
               avisos_hoy=[], mensajes=[])
    dia["comidas"] = [{"meal_type": "desayuno", "meal_name": "mangú"}]
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Almuerzo"]


def test_si_faltan_las_dos_se_avisa_la_mas_reciente(dia):
    dia.update(ahora=_utc(16, 17), registros={"Desayuno": ["11:40", "11:50"], "Almuerzo": ["12:10", "12:20"]},
               comidas=[], avisos_hoy=[], mensajes=[])
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Almuerzo"], "a la 1:30 PM se pregunta por el almuerzo, no por el desayuno"


# ── Un aviso bloqueado se reintenta, sin repetirse ────────────────────────────────────────────────

def test_un_aviso_bloqueado_por_la_conversacion_se_reintenta_la_hora_siguiente(dia):
    # [P1-PLAN-LOTE-150] 13:30 local en vez de 15:30: con la antelación, a las 15:30 la pendiente ya es la
    # MERIENDA y el escenario dejaría de hablar del almuerzo. 13:30: el coach respondió hace 20 min →
    # anti-spam. 14:30: ya puede salir, y el almuerzo sigue pendiente.
    dia.update(ahora=_utc(16, 17))
    dia["mensajes"] = [{"role": "model", "created_at": "2026-09-16 17:10:00+00"}] + dia["mensajes"]
    dia["correr"]()
    assert dia["avisos_nuevos"] == []
    dia.update(ahora=_utc(16, 18))
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Almuerzo"]
    assert "12:45 PM" in dia["prompts"][0], "el aviso dice la hora en que tocaba"


def test_un_aviso_ya_enviado_hoy_no_se_repite(dia, monkeypatch):
    monkeypatch.setenv("MEALFIT_PROACTIVE_MAX_NUDGES_PER_DAY", "4")
    # [P1-PLAN-LOTE-150] 13:30: a las 15:30 ya tocaría la merienda y el caso dejaría de ser «no se repite».
    dia.update(ahora=_utc(16, 17), avisos_hoy=["Desayuno", "Almuerzo"])
    dia["correr"]()
    assert dia["avisos_nuevos"] == []


def test_pasada_la_ventana_toca_la_siguiente_comida(dia, monkeypatch):
    # 17:30: el almuerzo (12:45) ya no se reintenta; toca la merienda (16:00 − 15 min = 15:45, ventana 15-17).
    monkeypatch.setenv("MEALFIT_PROACTIVE_MAX_NUDGES_PER_DAY", "4")
    dia.update(ahora=_utc(16, 21))
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Merienda"]


def test_con_la_ventana_en_1_la_conducta_es_la_de_antes(dia, monkeypatch):
    monkeypatch.setenv("MEALFIT_PROACTIVE_NUDGE_RETRY_HOURS", "1")
    dia.update(ahora=_utc(16, 20))
    dia["correr"]()
    assert dia["avisos_nuevos"] == []


def test_con_el_tope_de_cuatro_la_merienda_llega_tras_desayuno_y_almuerzo(dia):
    # Decisión del dueño (16-sep): un aviso por comida. Con el 2 de antes, este caso se quedaba sin merienda.
    dia.update(ahora=_utc(16, 21), avisos_hoy=["Desayuno", "Almuerzo"])
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Merienda"]


def test_el_tope_de_cuatro_corta_el_quinto(dia):
    # Sin registrar nada en todo el día y con los cuatro avisos dados, el Resumen de las 23:00 ya no sale.
    dia.update(ahora=_utc(17, 3), comidas=[], mensajes=[],
               avisos_hoy=["Desayuno", "Almuerzo", "Merienda", "Cena"])
    dia["correr"]()
    assert dia["avisos_nuevos"] == []


def test_el_tope_diario_es_un_knob(dia, monkeypatch):
    monkeypatch.setenv("MEALFIT_PROACTIVE_MAX_NUDGES_PER_DAY", "2")
    dia.update(ahora=_utc(16, 21), avisos_hoy=["Desayuno", "Almuerzo"])
    dia["correr"]()
    assert dia["avisos_nuevos"] == []


def test_el_resumen_de_las_23_no_cambia(dia):
    dia.update(ahora=_utc(17, 3), comidas=[], avisos_hoy=[], mensajes=[])
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Resumen del día"]


# ── La franja de cada comida ──────────────────────────────────────────────────────────────────────

def _avg(monkeypatch, horas, comida, ventana):
    import db_facts
    monkeypatch.setattr(db_facts, "execute_sql_query", lambda q, p, fetch_all=False: [_fila(h) for h in horas])
    monkeypatch.setattr(db_facts, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(db_facts, "connection_pool", object())
    return db_facts.get_avg_meal_hour("u-1", comida, ventana=ventana)


def test_la_franja_descarta_los_registros_tardios(monkeypatch):
    import proactive_agent as pa
    ventana = pa.FRANJA_DE_COMIDA["Desayuno"]
    assert _avg(monkeypatch, ["12:58"], "Desayuno", ventana) is None
    assert _avg(monkeypatch, ["07:00", "12:58", "08:00"], "Desayuno", ventana) == pytest.approx(7.5, abs=0.01)


def test_la_franja_de_la_cena_cruza_la_medianoche(monkeypatch):
    import proactive_agent as pa
    ventana = pa.FRANJA_DE_COMIDA["Cena"]
    assert _avg(monkeypatch, ["23:30", "00:30"], "Cena", ventana) == pytest.approx(0.0, abs=0.01)
    assert _avg(monkeypatch, ["15:00"], "Cena", ventana) is None


def test_sin_franja_la_media_no_cambia(monkeypatch):
    # El contrato viejo (lo usan los tests de signo y de media circular) queda intacto.
    assert _avg(monkeypatch, ["12:58"], "Desayuno", None) == pytest.approx(12.97, abs=0.01)


def test_las_franjas_cubren_las_cuatro_comidas_y_sus_horas_por_defecto():
    import proactive_agent as pa
    for comida, hora in {"Desayuno": 9.0, "Almuerzo": 13.0, "Merienda": 16.0, "Cena": 19.5}.items():
        desde, hasta = pa.FRANJA_DE_COMIDA[comida]
        dentro = desde <= hora < hasta if desde <= hasta else (hora >= desde or hora < hasta)
        assert dentro, f"la hora por defecto de {comida} cae fuera de su propia franja"


# ── Anclas ────────────────────────────────────────────────────────────────────────────────────────

def test_el_consumidor_pasa_la_franja():
    fuente = _src("proactive_agent.py")
    assert "get_avg_meal_hour(user_id, meal, ventana=FRANJA_DE_COMIDA.get(meal))" in fuente
    assert "P1-PLAN-LOTE-72" in fuente


def test_el_tope_por_defecto_es_cuatro():
    import proactive_agent as pa
    assert pa._max_avisos_por_dia() == 4


def test_marcador_y_documento():
    assert "P1-PLAN-LOTE-72" in _src("app.py")
    doc = _src("docs/recordatorios_de_comida.md")
    assert "P1-PLAN-LOTE-72" in doc
    for knob in ("MEALFIT_PROACTIVE_MAX_NUDGES_PER_DAY", "MEALFIT_PROACTIVE_NUDGE_RETRY_HOURS"):
        assert knob in doc and knob in _src("proactive_agent.py"), knob
