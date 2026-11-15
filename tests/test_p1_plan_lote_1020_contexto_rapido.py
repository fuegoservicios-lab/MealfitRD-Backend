"""[P1-PLAN-LOTE-1020 · A-11] Fase de contexto sin router de memoria y sin sentimiento cuando no hay señal afectiva."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import coach_contexto_rapido as ccr  # noqa: E402

_UID = "u-dueno"


class _Tracker:
    def __init__(self):
        self.arms = {}

    def set_arm(self, f, b):
        self.arms[f] = b


def _sin_router(p):
    return {"skip": False, "query": p}


@pytest.fixture
def canario(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_CANARY_USER_IDS", _UID)
    monkeypatch.delenv("MEALFIT_CHAT_CANARY_CONTEXTO_RAPIDO", raising=False)


def test_canario_por_defecto_solo_para_las_cuentas_canario(canario):
    t = _Tracker()
    out = ccr.aplicar(t, "u-otro", "qué puedo cenar hoy?", True, True, True, None, _sin_router)
    assert out == (True, True, None) and t.arms == {"contexto_rapido": "off"}
    out = ccr.aplicar(t, _UID, "qué puedo cenar hoy?", True, True, True, None, _sin_router)
    assert out == (False, False, {"skip": False, "query": "qué puedo cenar hoy?"})
    assert t.arms == {"contexto_rapido": "canary"}


def test_apagado_no_cambia_nada(canario, monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_CANARY_CONTEXTO_RAPIDO", "off")
    assert ccr.aplicar(None, _UID, "qué puedo cenar hoy?", True, True, True, None, _sin_router) == (True, True, None)


@pytest.mark.parametrize("texto", [
    "estoy frustrado, no bajo de peso", "me siento culpable por lo que comí", "logré bajar 2 libras!!",
    "hoy fue un día horrible 😢", "I feel so tired and frustrated",
])
def test_con_senal_afectiva_o_en_otro_idioma_el_sentimiento_se_clasifica(canario, texto):
    do_sent, do_router, _ = ccr.aplicar(None, _UID, texto, True, True, True, None, _sin_router)
    assert do_sent is True and do_router is False


def test_sin_memoria_no_hay_busqueda(canario):
    assert ccr.aplicar(None, _UID, "qué puedo cenar hoy?", False, True, False, None, _sin_router) == (False, False, None)


def test_un_error_deja_las_decisiones_como_venian(canario):
    def revienta(p):
        raise RuntimeError("x")
    assert ccr.aplicar(None, _UID, "qué puedo cenar hoy?", True, True, True, "prev", revienta) == (True, True, "prev")



def test_enganchado_antes_del_pool_y_en_el_camino_sin_stream():
    s = (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")
    i = s.index("_ctx_rapido_aplicar(\n")
    assert i < s.index("_pre_ex.submit(classify_sentiment, prompt)") and i > s.index("[SENTIMENT FASTPATH]")
    # [P1-PLAN-LOTE-1025] el camino sin stream (y su `router_del_turno`) ya no existe: decide `_ctx_rapido_aplicar`
    assert "_f_rag = _pre_ex.submit(rag_query_router, prompt) if _do_rag_router else None" in s
    assert s.count("rag_query_router(prompt)") == 0
