# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1045 · D6] El respaldo del coach no puede gastar más de US$ 0,50 por hora (decisión del dueño)."""
from pathlib import Path

import pytest

import coach_respaldo_tope as t

_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _limpio(monkeypatch):
    t._cache.clear()
    monkeypatch.delenv("MEALFIT_CHAT_FALLBACK_MAX_USD_PER_HOUR", raising=False)
    alertas = []
    monkeypatch.setattr(t, "_alertar", lambda m, g, tope: alertas.append((m, g, tope)))
    return alertas


def test_el_tope_por_defecto_es_50_centavos():
    assert t.tope_usd() == 0.50


def test_bajo_el_tope_se_usa(monkeypatch, _limpio):
    monkeypatch.setattr(t, "_gasto_sql", lambda m: 0.49)
    assert t.permitir("gemini-4-hydrogen") is True
    assert _limpio == []


def test_en_el_tope_se_corta_y_avisa(monkeypatch, _limpio):
    monkeypatch.setattr(t, "_gasto_sql", lambda m: 0.50)
    assert t.permitir("gemini-4-hydrogen") is False
    assert _limpio == [("gemini-4-hydrogen", 0.50, 0.50)]


def test_base_caida_no_corta(monkeypatch, _limpio):
    def roto(m):
        raise RuntimeError("db")
    monkeypatch.setattr(t, "_gasto_sql", roto)
    assert t.permitir("gemini-4-hydrogen") is True


def test_tope_cero_es_sin_tope(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_FALLBACK_MAX_USD_PER_HOUR", "0")
    monkeypatch.setattr(t, "_gasto_sql", lambda m: 99.0)
    assert t.permitir("gemini-4-hydrogen") is True


def test_la_lectura_se_cachea_60_s(monkeypatch):
    n = []
    monkeypatch.setattr(t, "_gasto_sql", lambda m: n.append(1) or 0.1)
    t.permitir("x"); t.permitir("x")
    assert len(n) == 1


def test_agent_lo_consulta_antes_de_usar_el_respaldo():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i = src.index('if not __import__("coach_respaldo_tope").permitir(_fallback_model):')
    j = src.index("chat_llm = _build_chat_fallback_llm(_fallback_model)")
    assert i < j and "raise _primary_exc" in src[i:j]
