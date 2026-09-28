# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-693 · 2026-09-28] El aviso de comida ya no interroga a quien SÍ come esa comida.

Auditoría del chat: «Ya casi te toca almorzar… ¿Qué te está fallando con el almuerzo?» al dueño, que registró el almuerzo
7 de 9 días avisados. La señal era la tasa de RESPUESTA al aviso y el tono contradecía la regla del 413. Replay sobre
producción (0 IA): el tono «se la salta» pasa de 10 casos a 1 (la merienda del dueño: 0 de 11 días registrada).
"""
from __future__ import annotations

import io
import os

import tono_del_aviso as t

_BACKEND = os.path.join(os.path.dirname(__file__), "..")


def _src(rel):
    return io.open(os.path.join(_BACKEND, rel), encoding="utf-8").read()


def test_umbral_y_minimo_de_dias():
    assert t.se_la_salta(1.0, 11) is True
    assert t.se_la_salta(0.7, 5) is True
    assert t.se_la_salta(0.25, 8) is False      # el almuerzo del dueño: 2 de 8 días sin registrar
    assert t.se_la_salta(1.0, 4) is False       # con 3-4 días no se juzga a nadie


def test_tasa_desde_la_base(monkeypatch):
    import db
    visto = {}

    def _q(sql, params, fetch_one=False):
        visto["sql"], visto["params"] = sql, params
        return {"dias": 11, "saltados": 11}
    monkeypatch.setattr(db, "execute_sql_query", _q)
    assert t.tasa_de_salto("u-1", "Merienda", 240) == (1.0, 11)
    assert visto["params"] == (240, 240, "u-1", "Merienda", 240, 240)
    # la fecha es la LOCAL y el día de hoy no cuenta (todavía puede registrarla)
    assert "consumed_at - make_interval(mins => %s))::date = d.dia" in visto["sql"]
    assert "< (NOW() - make_interval(mins => %s))::date" in visto["sql"]
    # misma regla que `_comida_ya_registrada`: por tipo o por nombre
    assert "c.meal_type" in visto["sql"] and "c.meal_name" in visto["sql"]


def test_sin_datos_o_con_error_no_se_juzga(monkeypatch):
    import db
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: {"dias": 0, "saltados": 0})
    assert t.tasa_de_salto("u", "Cena") == (0.0, 0)

    def _boom(*a, **k):
        raise RuntimeError("db caída")
    monkeypatch.setattr(db, "execute_sql_query", _boom)
    assert t.tasa_de_salto("u", "Cena") == (0.0, 0)


def test_los_tonos_no_interrogan_regla_del_413():
    prompt = _src("prompts/proactive.py")
    assert "Nada de interrogatorios" in prompt
    for tono in (t.TONO_SE_LA_SALTA, t.TONO_POCA_RESPUESTA):
        assert "Pregúntale" not in tono and "pregúntale si hay" not in tono


def test_el_cron_usa_la_senal_nueva_y_no_la_vieja():
    s = _src("proactive_agent.py")
    assert "Pregúntale qué está fallando particularmente" not in s
    assert "pregúntale si hay algún obstáculo" not in s
    assert "if se_la_salta(*tasa_de_salto(user_id, meal, _user_tz_off)):" in s
    assert "meal_rate < 0.30 and meal_total >= 3" not in s


def test_el_prompt_no_da_la_frase_que_se_copiaba():
    prompt = _src("prompts/proactive.py")
    assert "es la hora que tiene puesta para este aviso" not in prompt
    assert "No hables de la hora del aviso ni del recordatorio" in prompt
