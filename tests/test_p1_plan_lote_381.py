# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-381 · 2026-09-26] El anti doble toque también protege lo registrado para AYER.

La auditoría de «Registrar comida»: el dedup de `log_consumed_meal` buscaba `consumed_at > NOW() - 60 s`, pero una
comida de «Ayer» (days_ago ≥ 1) nace con `consumed_at` en el pasado: esa condición no la ve nunca, y un doble toque o
el reintento tras un timeout (el servidor sí guardó) la duplicaba. La ventana corta es de CUÁNDO SE ESCRIBIÓ la fila
(`created_at`); y para no confundir el café de ayer con el de hoy registrados seguidos, además el mismo día de consumo
(±12 h del `consumed_at` que se va a insertar).
Tooltip-anchor: P1-PLAN-LOTE-381
"""
from __future__ import annotations

import inspect


def _sql_del_dedup(monkeypatch, **kw):
    import db_facts
    vistos = []

    def fake_query(sql, params, fetch_one=False):
        vistos.append((sql, params))
        return {"id": "x"}   # «ya existe»

    monkeypatch.setattr(db_facts, "execute_sql_query", fake_query)
    r = db_facts.log_consumed_meal("u1", "Café", 50, 1, 5, 2, meal_type="desayuno", **kw)
    return r, vistos


def test_el_dedup_mira_cuando_se_escribio_la_fila_no_cuando_se_comio(monkeypatch):
    r, vistos = _sql_del_dedup(monkeypatch, consumed_at_override="2026-09-25T12:00:00+00:00")
    assert r == "deduped"
    sql, params = vistos[0]
    assert "created_at > NOW() - make_interval(secs => %s)" in sql
    assert "consumed_at > NOW()" not in sql
    # el mismo día de consumo que la fila que se va a insertar (no el café de hoy contra el de ayer)
    assert "consumed_at BETWEEN %s::timestamptz - interval '12 hours' AND %s::timestamptz + interval '12 hours'" in sql
    assert params[-2:] == ("2026-09-25T12:00:00+00:00", "2026-09-25T12:00:00+00:00")


def test_ancla():
    import db_facts
    assert "P1-PLAN-LOTE-381" in inspect.getsource(db_facts.log_consumed_meal)
