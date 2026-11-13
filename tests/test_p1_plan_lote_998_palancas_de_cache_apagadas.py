# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-998 · 2026-11-13] Revisión del lote 997 (eficiencia de Gemini 4 Hydrogen).

Lo que se dejó ENCENDIDO del 997 (baja latencia sin tocar el comportamiento): lecturas en paralelo antes del
modelo, atajo sin RAG para registros del presente, caché de embeddings, deduplicación por turno y telemetría de
tokens cacheados/razonamiento.

Lo que se dejó APAGADO por defecto (interruptor para A/B contra `cached_tokens` de `llm_usage_events`):
  * `MEALFIT_CHAT_PROMPT_COMPACT_ENABLED`: el manual «compacto» (12 k) NO deduplica — el legacy (63 k) casi no repite
    frases (0 de 429) y el compacto descartaba 42 de 54 reglas con etiqueta.
  * `MEALFIT_GEMINI_CROSS_USER_CACHE_ANCHOR`: pasa el manual de rol sistema a rol usuario sin beneficio medido
    (la sonda del 13-nov: 6 variantes, 0 tokens cacheados en todas).
  * `MEALFIT_CHAT_SPLIT_VOLATILE_SYS_PROMPT`: movía el bloque clínico/alergias al mensaje de usuario.
  * `MEALFIT_CHAT_TOOL_GATING_ON_GEMINI` vuelve a True (catálogo recortado por turno, como antes).

tooltip-anchor: P1-PLAN-LOTE-998
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _limpiar(monkeypatch):
    for k in ("MEALFIT_CHAT_PROMPT_COMPACT_ENABLED", "MEALFIT_GEMINI_CROSS_USER_CACHE_ANCHOR",
              "MEALFIT_CHAT_SPLIT_VOLATILE_SYS_PROMPT", "MEALFIT_CHAT_TOOL_GATING_ON_GEMINI",
              "MEALFIT_CHAT_TOOL_GATING_ENABLED", "MEALFIT_CHAT_PREFETCH_PARALLEL",
              "MEALFIT_CHAT_FASTPATH_PRESENT_LOGS", "MEALFIT_QUERY_EMBED_CACHE_TTL_S"):
        monkeypatch.delenv(k, raising=False)


def test_las_palancas_de_cache_sin_evidencia_salen_apagadas(monkeypatch):
    import agent
    import gemini_hydrogen as gh
    _limpiar(monkeypatch)
    assert agent._chat_prompt_compact_enabled() is False
    assert agent._chat_split_volatile_sys_prompt() is False
    assert gh.cache_anchor_activo() is False
    assert agent._tool_gating_enabled_for_model("gemini-4-hydrogen") is True


def test_lo_que_baja_la_latencia_sigue_encendido(monkeypatch):
    import agent
    _limpiar(monkeypatch)
    assert agent._chat_prefetch_enabled() is True
    assert agent._chat_fastpath_present_logs() is True


def test_el_compacto_solo_puede_encenderse_si_conserva_las_reglas_del_legacy(monkeypatch):
    """Invariante: mientras el manual compacto pierda reglas etiquetadas del legacy, su default tiene que ser False."""
    import agent
    from constants import coach_country_context, culinary_knowledge_base_for_country
    from prompts import chat_agent as ca
    from prompts.coach_compact import build_coach_static_manual
    _limpiar(monkeypatch)
    legacy = (ca.CHAT_STREAM_INLINE_PROMPT + coach_country_context("DO") + "\n"
              + culinary_knowledge_base_for_country("DO") + ca.build_tools_instructions_stream("U"))
    compacto = build_coach_static_manual(country="DO", stream=True)
    tag = re.compile(r"\b(?:P\d-[A-Z0-9-]+|BB-\d+)")
    perdidas = set(tag.findall(legacy)) - set(tag.findall(compacto))
    if perdidas:
        assert agent._chat_prompt_compact_enabled() is False, (
            f"El manual compacto pierde {len(perdidas)} reglas etiquetadas y está encendido: {sorted(perdidas)[:8]}")


def test_el_handoff_del_router_siembra_el_alcance_del_turno():
    """Antes `_tc_consume_handoff` se llamaba y su resultado se tiraba: el perfil se releía en cada turno."""
    src = io.open(_BACKEND / "agent.py", encoding="utf-8").read()
    assert src.count("_tc_scope(initial=_tc_consume_handoff(user_id, session_id))") == 2   # stream y no-stream
    from turn_cache import consume_handoff, get_cached, stash_handoff, turn_cache_scope, set_cached
    with turn_cache_scope():
        set_cached("profile", "u-h", {"health_profile": {"country": "DO"}})
        stash_handoff("u-h", "s-h")
    snap = consume_handoff("u-h", "s-h")
    assert snap and ("profile", "u-h") in snap
    stash_handoff("u-h", "s-h")   # sin alcance activo no guarda nada
    assert consume_handoff("u-h", "s-h") is None
    with turn_cache_scope():
        set_cached("profile", "u-h", {"health_profile": {"country": "DO"}})
        stash_handoff("u-h", "s-h")
    with turn_cache_scope(initial=consume_handoff("u-h", "s-h")):
        assert get_cached("profile", "u-h")[0] is True
    assert consume_handoff("u-h", "s-h") is None   # one-shot


def test_marker_del_998_y_el_996_acepta_series_posteriores():
    src = io.open(_BACKEND / "app.py", encoding="utf-8").read()
    assert re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-99[89] · 2026-11-13"', src)


def test_update_form_field_nunca_se_recorta_por_el_gating():
    """El gating por palabras dejaba al coach SIN `update_form_field` en «soy alérgico al maní», «soy vegano»,
    «tengo diabetes» o «vivo en México», que el manual exige guardar de inmediato."""
    import agent
    from langchain_core.messages import HumanMessage
    for frase in ("soy alérgico al maní", "a partir de hoy soy vegano", "tengo diabetes", "ahora vivo en México",
                  "me desayuné 2 huevos"):
        elegidas = {t.name for t in agent._select_tools_for_turn(
            agent.agent_tools, {"messages": [HumanMessage(content=frase)]})}
        assert "update_form_field" in elegidas, frase
        assert len(elegidas) < len(agent.agent_tools), "el recorte sigue ahorrando esquemas"
