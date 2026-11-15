"""[P1-PLAN-LOTE-1012 · A-06 + A-03] Medición completa del turno y despliegue por canario.

La verificación contra la base real (las 3 consultas nuevas de `chat_phase_metrics`) se hizo a mano en el VPS antes del
commit y está anotada en la memoria del lote; estas pruebas fijan el resto.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── tracker ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _tracker():
    from chat_phase_metrics import ChatTurnPhaseTracker
    return ChatTurnPhaseTracker(user_id="u-1", session_id="s-1", corr="abc12345")


def test_el_tracker_guarda_primera_senal_bloques_brazo_y_primer_token():
    t = _tracker()
    t.mark_first_progress()
    t.mark_first_progress()   # solo cuenta la primera
    t.record_context_block("pantry", 120.4)
    t.record_context_block("consumed_today", 35)
    t.set_arm("cache_v2", "canary")
    t.record_llm_call(model="gpt-6-luna", duration_ms=1900, input_tokens=30000, cached_tokens=0, output_tokens=170, ttft_ms=640)
    t.record_llm_call(model="gpt-6-luna", duration_ms=1300)
    meta = t.to_metadata()
    assert isinstance(meta["first_progress_ms"], int) and meta["first_progress_ms"] >= 0
    assert meta["context_blocks_ms"] == {"pantry": 120, "consumed_today": 35}
    assert meta["arm"] == {"cache_v2": "canary"}
    assert meta["llm_calls"][0]["ttft_ms"] == 640
    assert "ttft_ms" not in meta["llm_calls"][1]


def test_los_bloques_y_los_brazos_tienen_tope():
    t = _tracker()
    for i in range(60):
        t.record_context_block(f"b{i}", 1)
        t.set_arm(f"f{i}", "on")
    meta = t.to_metadata()
    assert len(meta["context_blocks_ms"]) == 24 and len(meta["arm"]) == 12


# ── canario ──────────────────────────────────────────────────────────────────────────────────────────────────────
def test_canario_off_canary_on(monkeypatch):
    import coach_canario as cc

    uid_dueno, uid_otro = "61A13831-2a70-4437-a084-0d3e09b653e4", "a9a658db-0000-0000-0000-000000000000"
    monkeypatch.setenv("MEALFIT_CHAT_CANARY_USER_IDS", " 61a13831-2a70-4437-a084-0d3e09b653e4 , x ")
    monkeypatch.delenv("MEALFIT_CHAT_CANARY_CACHE_V2", raising=False)
    assert cc.modo("cache_v2") == "off"
    assert cc.brazo(uid_dueno, "cache_v2") == "off" and not cc.activa(uid_dueno, "cache_v2")

    monkeypatch.setenv("MEALFIT_CHAT_CANARY_CACHE_V2", "canary")
    assert cc.brazo(uid_dueno, "cache_v2") == "canary" and cc.activa(uid_dueno, "cache_v2")
    assert cc.brazo(uid_otro, "cache_v2") == "off"
    assert cc.brazo(None, "cache_v2") == "off"

    monkeypatch.setenv("MEALFIT_CHAT_CANARY_CACHE_V2", "on")
    assert cc.brazo(uid_otro, "cache_v2") == "on" and cc.brazo(None, "cache_v2") == "on"

    monkeypatch.setenv("MEALFIT_CHAT_CANARY_CACHE_V2", "valor-raro")   # valor inválido ⇒ el default, que es apagado
    assert cc.modo("cache_v2") == "off"


def test_canario_funcion_desconocida_falla_alto():
    import coach_canario as cc

    with pytest.raises(KeyError):
        cc.modo("no_existe")


def test_registrar_brazo_anota_en_el_tracker(monkeypatch):
    import coach_canario as cc

    monkeypatch.setenv("MEALFIT_CHAT_CANARY_USER_IDS", "u-1")
    monkeypatch.setenv("MEALFIT_CHAT_CANARY_UNA_LLAMADA", "canary")
    t = _tracker()
    assert cc.registrar_brazo(t, "u-1", "una_llamada") is True
    assert cc.registrar_brazo(t, "u-2", "una_llamada") is False
    assert t.to_metadata()["arm"] == {"una_llamada": "off"}   # la última escritura gana; el brazo del turno es de UN usuario
    assert cc.registrar_brazo(None, "u-1", "una_llamada") is True   # sin tracker no rompe


def test_las_funciones_del_canario_tienen_knob_registrado():
    import coach_canario as cc
    from knobs import get_knobs_registry_snapshot

    for f in cc.FUNCIONES:
        cc.modo(f)
    cc.usuarios_canario()
    reg = get_knobs_registry_snapshot()
    nombres = set(reg) if isinstance(reg, dict) else {r.get("name") for r in reg}
    for k in ("MEALFIT_CHAT_CANARY_USER_IDS", "MEALFIT_CHAT_CANARY_CACHE_V2", "MEALFIT_CHAT_CANARY_UNA_LLAMADA"):
        assert k in nombres, k


# ── prefetcher: tiempo por bloque ────────────────────────────────────────────────────────────────────────────────
def test_el_prefetcher_cronometra_cada_tarea_y_las_manda_al_tracker():
    import concurrent.futures
    import time as _t

    import agent
    from chat_phase_metrics import start_turn_tracker, release_turn_tracker

    pf = agent._ChatContextPrefetcher("s-pf", "u-pf", None, "2026-11-14", 240, True, True)
    trk, tok = start_turn_tracker(user_id="u-pf", session_id="s-pf", corr="pf000001")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            pf._futures["lento"] = pf._sub("lento", ex, lambda: (_t.sleep(0.05), "v")[1])
            pf._futures["falla"] = pf._sub("falla", ex, lambda: (_ for _ in ()).throw(RuntimeError("x")))
            pf.resolve_all()   # el error de una tarea ya se tragaba en resolve_all; aquí solo importa que se cronometre
        assert pf.blocks_ms["lento"] >= 45
        assert "falla" in pf.blocks_ms
        meta = trk.to_metadata()
        assert meta["context_blocks_ms"]["lento"] >= 45
    finally:
        release_turn_tracker(trk, tok)


# ── primer token ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_el_stream_anota_el_primer_token_en_el_sumidero(monkeypatch):
    import time as _t

    import llm_provider
    from langchain_openai.chat_models.base import BaseChatOpenAI

    def _falso(self, *a, **k):
        _t.sleep(0.05)
        yield "trozo-1"
        yield "trozo-2"

    monkeypatch.setattr(BaseChatOpenAI, "_stream", _falso)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    llm = llm_provider.ChatGLM(model="gpt-6-luna", temperature=0.7, timeout=5)
    sink: dict = {}
    llm_provider.set_ttft_sink(sink)
    try:
        assert list(llm._stream([])) == ["trozo-1", "trozo-2"]
    finally:
        llm_provider.set_ttft_sink(None)
    assert sink["ttft_ms"] >= 45
    # sin sumidero activo no pasa nada
    assert list(llm._stream([])) == ["trozo-1", "trozo-2"]


# ── contrato del código ──────────────────────────────────────────────────────────────────────────────────────────
def _src(nombre):
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / nombre).read_text(encoding="utf-8")


def test_call_model_abre_y_cierra_el_sumidero_y_pasa_el_primer_token():
    s = _src("agent.py")
    assert "_set_ttft_sink(_ttft_sink)" in s and "_set_ttft_sink(None)" in s
    assert 'ttft_ms=_ttft_sink.get("ttft_ms")' in s


def test_el_stream_marca_la_primera_senal_antes_del_primer_evento():
    s = _src("agent.py")
    i = s.index("_pt.mark_first_progress()")
    assert "'phase': 'analizando'" in s[i:i + 400]


def test_el_panel_devuelve_cache_y_brazos():
    s = _src("routers/system.py")
    assert 'summary["cache"] = query_chat_cache_summary(window_h)' in s
    assert 'summary["by_arm"] = query_chat_by_arm(window_h)' in s


def test_las_consultas_nuevas_usan_columnas_reales():
    """`llm_usage_events` tiene cost_usd_micros y node (no endpoint ni estimated_cost_micros): el error de los lotes 1010."""
    s = _src("chat_phase_metrics.py")
    i = s.index("def query_chat_cache_summary")
    cuerpo = s[i:]
    assert "cost_usd_micros" in cuerpo and "node = 'chat_call_model'" in cuerpo
    for fantasma in ("estimated_cost_micros", "endpoint"):
        assert fantasma not in cuerpo
