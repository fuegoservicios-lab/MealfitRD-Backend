# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-996 · 2026-11-13] Dos fallos medidos en un turno del coach con Gemini 4 Hydrogen (55 s, $0,35):

  1. `log_consumed_meal` esperó 32 s: la resolución de ingredientes intentó construir la caché semántica desde
     cero dentro de la petición (354 alimentos, lotes de 3, 3 s entre lotes ⇒ ~6 min) con un plazo de 30 s, así
     que abortaba siempre. Una petición ya no la construye: lanza el calentador y sigue por el fast-path.
  2. En streaming Gemini manda el uso ACUMULADO en cada trozo y LangChain lo suma: `llm_usage_events` registraba
     2-6× los tokens reales. Solo el trozo con `finish_reason` conserva el uso.

tooltip-anchor: P1-PLAN-LOTE-996
"""
from __future__ import annotations

import io
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gemini_hydrogen as gh
import shopping_calculator as sc

_BACKEND = Path(__file__).resolve().parent.parent


import calentador_semantico as cs


def _reset():
    sc._semantic_cache = None
    sc._semantic_cache_failed_until = 0.0
    cs._vivo = False


def _aislado(master):
    """Lo que otras pruebas de la batería dejan cambiado (kill-switch, lock, caché) no decide aquí."""
    return (patch.object(sc, "_semantic_cache_disabled", return_value=False),
            patch.object(sc, "get_master_ingredients", return_value=master),
            patch.object(sc, "_semantic_cache", None),
            patch.object(sc, "_semantic_cache_failed_until", 0.0),
            patch.object(sc, "_semantic_cache_lock", threading.Lock()))


def test_la_peticion_no_construye_la_cache_y_lanza_el_calentador(monkeypatch):
    _reset()
    monkeypatch.delenv("MEALFIT_EMBED_INIT_EN_PETICION", raising=False)
    master = [{"name": "Pollo", "category": "Proteína", "aliases": []}]
    emb = MagicMock()
    d, m, c, f, lk = _aislado(master)
    with d, m, c, f, lk,          patch.object(sc, "_try_load_embed_vectors_from_redis", return_value=None),          patch("embeddings_provider.get_embeddings_client", return_value=emb),          patch.object(cs, "lanzar", return_value=True) as lanzar:
        assert sc.get_semantic_cache() is None
    lanzar.assert_called_once()
    emb.embed_documents.assert_not_called()


def test_con_redis_la_peticion_sigue_sirviendo_la_cache():
    _reset()
    master = [{"name": "Pollo", "category": "Proteína", "aliases": []}]
    d, m, c, f, lk = _aislado(master)
    with d, m, c, f, lk,          patch("embeddings_provider.get_embeddings_client", return_value=MagicMock()),          patch.object(sc, "_try_load_embed_vectors_from_redis", return_value=[[0.1, 0.2]]),          patch.object(cs, "lanzar") as lanzar:
        assert sc.get_semantic_cache() is not None
    lanzar.assert_not_called()


def test_un_solo_calentador_a_la_vez():
    cs._vivo = True
    assert cs.lanzar(lambda: None) is False
    cs._vivo = False


def test_el_codigo_nuevo_vive_en_su_modulo():
    src = io.open(_BACKEND / "shopping_calculator.py", encoding="utf-8").read()
    assert "_cs.lanzar(lambda: get_semantic_cache(deadline_s=EMBED_WARM_DEADLINE_S))" in src


def _gen(uso):
    from langchain_core.messages import AIMessageChunk
    return SimpleNamespace(message=AIMessageChunk(content="", usage_metadata=uso))


_USO = {"input_tokens": 1512, "output_tokens": 1, "total_tokens": 1513}


def test_el_trozo_intermedio_pierde_el_uso_acumulado():
    g = _gen(dict(_USO))
    gh.uso_solo_al_final({"choices": [{"delta": {"content": "Ho"}, "finish_reason": None}]}, g)
    assert g.message.usage_metadata is None


def test_el_trozo_final_lo_conserva():
    g = _gen(dict(_USO))
    gh.uso_solo_al_final({"choices": [{"delta": {}, "finish_reason": "stop"}]}, g)
    assert g.message.usage_metadata["input_tokens"] == 1512


def test_un_trozo_solo_de_uso_lo_conserva():
    g = _gen(dict(_USO))
    gh.uso_solo_al_final({"choices": [], "usage": {"prompt_tokens": 1512}}, g)
    assert g.message.usage_metadata is not None


def test_el_cliente_aplica_el_ajuste_solo_a_gemini():
    src = io.open(_BACKEND / "llm_provider.py", encoding="utf-8").read()
    assert "_gh.uso_solo_al_final(chunk, generacion)" in src


def test_marker():
    assert '_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-99' in io.open(_BACKEND / "app.py", encoding="utf-8").read()   # 996 o posterior


# ── foto + etiqueta corta: sin sentimiento ni memoria antes del primer token ─────────────────────────
import pytest


@pytest.mark.parametrize("texto, esperado", [
    ("Desayuno", True), ("", True), ("cena de hoy", True),
    ("¿esto engorda mucho?", False), ("esto es lo que comí hoy en casa de mi abuela", False),
])
def test_foto_con_etiqueta_corta(texto, esperado):
    import agent
    assert agent._foto_con_etiqueta_corta(texto) is esperado


def test_el_stream_salta_sentimiento_y_rag_con_foto_y_etiqueta():
    src = io.open(_BACKEND / "agent.py", encoding="utf-8").read()
    i = src.index("elif vision and _foto_con_etiqueta_corta(prompt):")
    bloque = src[i:i + 700]
    assert "_do_sentiment = False" in bloque and "_do_rag_router = False" in bloque and '{"skip": True}' in bloque
