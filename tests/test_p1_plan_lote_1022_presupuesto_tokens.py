"""[P1-PLAN-LOTE-1022 · A-21] Una llamada del coach por encima de 40 mil tokens de entrada se avisa (una vez por hora)."""
from __future__ import annotations

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import coach_presupuesto_tokens as cpt  # noqa: E402


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    cpt.reset_para_tests()
    monkeypatch.delenv("MEALFIT_CHAT_TURN_TOKEN_ALERT", raising=False)
    escritas = []
    monkeypatch.setattr(cpt, "_abrir_alerta", lambda *a, **k: escritas.append(a))
    yield escritas
    cpt.reset_para_tests()


def test_bajo_el_umbral_no_avisa(limpio):
    assert cpt.vigilar(33_000, modelo="gpt-6-luna") is False
    assert cpt.vigilar(None) is False and limpio == []


def test_sobre_el_umbral_avisa_una_vez_por_hora_y_cuenta_todos(limpio, caplog):
    with caplog.at_level(logging.WARNING):
        assert cpt.vigilar(52_000, modelo="gpt-6-luna") is True
        assert cpt.vigilar(61_000, modelo="gpt-6-luna") is True
    assert len(limpio) == 1                                  # una alerta por hora, no por llamada
    assert caplog.text.count("[A-21 TOKENS]") == 1
    assert cpt.snapshot() == {"excesos": 2, "maximo": 61_000}


def test_el_umbral_es_un_knob_y_cero_lo_apaga(limpio, monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_TURN_TOKEN_ALERT", "0")
    assert cpt.vigilar(500_000) is False
    monkeypatch.setenv("MEALFIT_CHAT_TURN_TOKEN_ALERT", "20000")
    assert cpt.vigilar(25_000) is True


def test_el_tracker_marca_la_llamada_y_nunca_rompe(limpio, monkeypatch):
    from chat_phase_metrics import ChatTurnPhaseTracker

    trk = ChatTurnPhaseTracker(user_id="u", session_id="s", corr="c1022")
    trk.record_llm_call(model="gpt-6-luna", duration_ms=1200, input_tokens=45_000, cached_tokens=40_000)
    trk.record_llm_call(model="gpt-6-luna", duration_ms=900, input_tokens=30_000)
    calls = trk.to_metadata()["llm_calls"]
    assert calls[0].get("sobre_presupuesto") is True and "sobre_presupuesto" not in calls[1]
    monkeypatch.setattr(cpt, "vigilar", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    trk.record_llm_call(model="gpt-6-luna", duration_ms=900, input_tokens=99_000)   # no lanza
    assert len(trk.to_metadata()["llm_calls"]) == 3
