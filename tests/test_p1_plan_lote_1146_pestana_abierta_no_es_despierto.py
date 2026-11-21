# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1146 · 2026-11-21] «Si dejo una pestaña abierta mientras duermo, ¿qué pasa?» (el dueño).

La presencia (`presencia:<uid>`) la marca CUALQUIER petición con sesión, y una pestaña abierta sondea sola toda la
noche: con ella en la cadena de actividad nunca había un hueco de sueño y el día seguía extendido hasta el mediodía
(el desayuno de las 8 caía en el día anterior). Ahora solo cuentan los ACTOS (registrar, escribir al coach), y un
desayuno de ahora estando «despierto» empieza el día nuevo.
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest

import dia_extendido as de

RD = 240
UID = "11111111-1111-1111-1111-111111111111"
_BACKEND = Path(__file__).resolve().parent.parent


def _u(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _encendido(monkeypatch):
    monkeypatch.setenv("MEALFIT_DIA_TERMINA_AL_DORMIR", "1")
    monkeypatch.setattr("db.get_user_profile", lambda uid: {"health_profile": {}}, raising=False)
    de.olvidar(UID)


def test_la_presencia_no_entra_en_la_actividad():
    vistos = []

    def _q(sql, params, fetch_all=False, fetch_one=False):
        vistos.append(sql)
        return [] if fetch_all else {}
    de.estado(UID, _u("2026-11-21T08:22:00"), tz_offset_min=RD, execute_sql_query=_q, usar_cache=False)
    actividad = next(s for s in vistos if "consumed_meals" in s)
    assert "presencia" not in actividad and "app_kv_store" not in actividad
    assert "agent_messages" in actividad


def test_un_desayuno_despierto_empieza_el_dia(monkeypatch):
    escrito = {}
    estados = iter([{"extendido": True, "motivo": "despierto"}])
    monkeypatch.setattr(de, "estado", lambda *a, **k: next(estados, {"extendido": False}))
    monkeypatch.setattr(de, "fijar", lambda uid, accion, ahora=None: escrito.update(uid=uid, accion=accion))
    assert de.hora_para_registro(UID, meal_type="desayuno") is None
    assert escrito == {"uid": UID, "accion": "empezar"}


@pytest.mark.parametrize("motivo,tipo", [("pedido", "desayuno"), ("corte_personal", "desayuno"), ("despierto", "merienda")])
def test_lo_que_el_eligio_y_otras_franjas_no_se_tocan(monkeypatch, motivo, tipo):
    monkeypatch.setattr(de, "estado", lambda *a, **k: {"extendido": True, "motivo": motivo,
                                                        "fin": "2026-11-21T08:00:00+00:00"})
    monkeypatch.setattr(de, "fijar", lambda *a, **k: pytest.fail("no debía empezar el día"))
    assert de.hora_para_registro(UID, meal_type=tipo) == "2026-11-21T07:59:00+00:00"


def test_cableado():
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert 'desayuno_empieza_el_dia(user_id, _meal_type)' in tools
    assert 'hora_para_registro(user_id, meal_type=meal_type)' in (_BACKEND / "db_facts.py").read_text(encoding="utf-8")
