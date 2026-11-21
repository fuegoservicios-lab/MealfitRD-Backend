# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1140 · 2026-11-21] Lo que contesta el RESPALDO del coach no cuenta contra los topes del usuario.

El 13-nov el coach corrió en `gemini-4-hydrogen` (~10 ¢ por mensaje frente a ~0,15 ¢ de Luna) y una cuenta gratis
quedó frenada por 6 mensajes. Hydrogen siguió siendo el RESPALDO: una caída de Luna repetía la historia. Decisión del
dueño (21-nov): respaldo barato (`gemini-3.8-flash`, en el `.env` del VPS) Y el gasto del respaldo no cuenta para el
usuario (lo absorbe el dueño, acotado por `coach_respaldo_tope` a US$ 0,50/h).
"""
from pathlib import Path

import coach_tope_diario as ctd
import topes_por_nivel as tpn

_BACKEND = Path(__file__).resolve().parent.parent
_FILTRO = "COALESCE(metadata->>'respaldo', '') <> 'true'"


def test_la_fila_del_respaldo_lleva_la_marca():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i = src.index("[P1-PLAN-LOTE-1140] turno del RESPALDO")
    trozo = src[i - 200:i + 400]
    assert "if _primary_exc is not None:" in trozo
    assert '"respaldo": True' in trozo
    assert trozo.index('"respaldo": True') < trozo.index("_emit_llm_usage_event_best_effort(")


def test_los_tres_topes_excluyen_el_respaldo():
    assert ctd.SIN_RESPALDO_SQL == _FILTRO
    assert "{SIN_RESPALDO_SQL}" in (_BACKEND / "coach_tope_diario.py").read_text(encoding="utf-8")
    assert _FILTRO in (_BACKEND / "chat_hard_turn.py").read_text(encoding="utf-8")
    assert _FILTRO in (_BACKEND / "topes_por_nivel.py").read_text(encoding="utf-8")


def test_gasto_mes_usd_manda_el_filtro(monkeypatch):
    visto = {}

    def _q(sql, params):
        visto["sql"] = sql
        return {"m": 250_000}

    monkeypatch.setattr(tpn, "_q", _q)
    assert tpn.gasto_mes_usd("u-1") == 0.25
    assert _FILTRO in visto["sql"]


def test_libre_desde_manda_el_filtro():
    visto = {}

    def _exec(sql, params, fetch_all=False):
        visto["sql"] = sql
        return []

    ctd.libre_desde("u-1", 0.25, execute_sql_query=_exec)
    assert _FILTRO in visto["sql"]
