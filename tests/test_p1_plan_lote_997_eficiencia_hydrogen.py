"""[P1-PLAN-LOTE-997 · 2026-11-13] Eficiencia de Gemini 4 Hydrogen (Etapas 1, 2 y 3).

Cubre:
  Etapa 1 — Implicit Prompt Caching & Ancla Cross-User + Telemetría:
    1. Prefijo estático global del system prompt idéntico entre usuarios distintos
       (`build_tools_instructions*_estaticas` sin user_id, contexto volátil movido
       al último HumanMessage vía `_VOLATILE_SYS_SPLIT_MARKER` sin romper multimodal).
    2. Ancla cross-user (`gemini_hydrogen.anclar_manual_estatico`) + cableado en
       `llm_provider.ChatOpenAI._get_request_payload` solo para modelos Gemini.
    3. Catálogo fijo de herramientas en Gemini (`_tool_gating_enabled_for_model`).
    4. Captura y propagación de `cached_tokens` y `reasoning_tokens` en
       `llm_telemetry` y `vision_agent`.

  Etapa 2 — Eliminación de waterfalls pre-TTFT:
    5. Caché TTL en RAM + singleflight para query embeddings en `embeddings_provider`.
    6. Deduplicación por turno vía `turn_cache` en módulos DB/dominio.
    7. Fast-path determinista `_rag_es_presente_sin_rag` y prefetcher paralelo.

  Etapa 3 — Compresión SSOT del Coach y Schemas de Herramientas:
    8. Manual estático compacto (`prompts/coach_compact.py`) con paridad de reglas
       clínicas/conductuales, fallback limpio ante monkeypatch y modo voz sin bloat visual.
    9. Docstrings `@tool` compactos en `tools.py` (< 22k chars wire JSON) conservando contratos.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


class _PromptCaptured(RuntimeError):
    pass


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — Implicit Prompt Caching & Telemetría
# ═════════════════════════════════════════════════════════════════════════════

def test_tools_instructions_static_prefix_identical_without_user_id():
    from prompts.chat_agent import (
        build_tools_instructions,
        build_tools_instructions_estaticas,
        build_tools_instructions_stream,
        build_tools_instructions_stream_estaticas,
    )

    assert build_tools_instructions_estaticas() != ""
    assert "El user_id" not in build_tools_instructions_estaticas()
    assert "El user_id" not in build_tools_instructions_stream_estaticas()
    assert build_tools_instructions_estaticas(plan_en_pausa=True) == build_tools_instructions_estaticas(
        plan_en_pausa=True
    )
    assert "user-alpha" in build_tools_instructions("user-alpha")
    assert "user-alpha" in build_tools_instructions_stream("user-alpha")


def test_llm_messages_from_state_splits_volatile_marker_and_preserves_multimodal(monkeypatch):
    import agent

    monkeypatch.setenv("MEALFIT_CHAT_SPLIT_VOLATILE_SYS_PROMPT", "1")   # [998] apagado por defecto
    static_prefix = "INSTRUCCION_ESTATICA_GLOBAL_V1"
    sys_a = f"{static_prefix}{agent._VOLATILE_SYS_SPLIT_MARKER}El user_id actual es: u-A\nDIARIO: 200 kcal"
    sys_b = f"{static_prefix}{agent._VOLATILE_SYS_SPLIT_MARKER}El user_id actual es: u-B\nDIARIO: 950 kcal"

    out_a = agent._llm_messages_from_state([HumanMessage(content="desayuné pan")], sys_a)
    out_b = agent._llm_messages_from_state([HumanMessage(content="desayuné pan")], sys_b)

    assert sum(isinstance(m, SystemMessage) for m in out_a) == 1
    assert isinstance(out_a[0], SystemMessage) and isinstance(out_b[0], SystemMessage)
    # Clave del implicit caching global: SystemMessage[0] es 100% byte-idéntico
    assert out_a[0].content == static_prefix
    assert out_a[0].content == out_b[0].content

    # Sin duplicar mensajes user consecutivos; contexto volátil inyectado en el último HumanMessage
    assert len(out_a) == 2 and isinstance(out_a[1], HumanMessage)
    assert "El user_id actual es: u-A" in out_a[1].content
    assert "El user_id actual es: u-B" in out_b[1].content
    assert out_a[1].content.endswith("desayuné pan")

    # Multimodal content (list) conserva imagen + inyecta texto volátil al principio
    mm_msg = HumanMessage(
        content=[
            {"type": "text", "text": "Qué ves aquí?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
        ]
    )
    out_mm = agent._llm_messages_from_state([mm_msg], sys_a)
    assert isinstance(out_mm[1].content, list)
    assert out_mm[1].content[0]["type"] == "text"
    assert "El user_id actual es: u-A" in out_mm[1].content[0]["text"]
    assert out_mm[1].content[-1]["image_url"]["url"].endswith("AAA")

    # Con nudge mid-turn (SystemMessage posterior al último HumanMessage): mantiene el nudge al final como user
    nudge = SystemMessage(content="Llama a log_consumed_meal ahora")
    out_nudge = agent._llm_messages_from_state([HumanMessage(content="cené pollo"), nudge], sys_a)
    assert out_nudge[0].content == static_prefix
    assert "[Contexto operativo del turno]" in out_nudge[1].content
    assert "[Instrucción del sistema] Llama a log_consumed_meal ahora" == out_nudge[2].content

    # Flag apagado ⇒ fallback clásico sin dividir
    monkeypatch.setenv("MEALFIT_CHAT_SPLIT_VOLATILE_SYS_PROMPT", "0")
    out_legacy = agent._llm_messages_from_state([HumanMessage(content="hola")], sys_a)
    assert out_legacy[0].content == sys_a
    assert out_legacy[1].content == "hola"


def test_tool_gating_sigue_activo_en_gemini_y_el_catalogo_fijo_es_opt_in(monkeypatch):
    """[P1-PLAN-LOTE-998] El default vuelve a ser el probado: el gating recorta el catálogo también en Gemini."""
    import agent

    monkeypatch.delenv("MEALFIT_CHAT_TOOL_GATING_ON_GEMINI", raising=False)
    monkeypatch.delenv("MEALFIT_CHAT_TOOL_GATING_ENABLED", raising=False)
    assert agent._tool_gating_enabled_for_model("gemini-4-hydrogen") is True
    assert agent._tool_gating_enabled_for_model("gemini-3.8-flash") is True
    assert agent._tool_gating_enabled_for_model("gpt-6-luna") is True

    monkeypatch.setenv("MEALFIT_CHAT_TOOL_GATING_ON_GEMINI", "0")
    assert agent._tool_gating_enabled_for_model("gemini-4-hydrogen") is False
    assert agent._tool_gating_enabled_for_model("gpt-6-luna") is True
    monkeypatch.delenv("MEALFIT_CHAT_TOOL_GATING_ON_GEMINI", raising=False)

    bound_batches = []

    class _FakeLLM:
        def __init__(self, model="gemini-4-hydrogen", **_kw):
            self.model = model

        def bind_tools(self, tools):
            bound_batches.append([getattr(t, "name", str(t)) for t in tools])
            return self

        def invoke(self, _msgs):
            return AIMessage(content="ok")

    class _FakeCB:
        def can_proceed(self):
            return True

        def record_success(self):
            pass

        def record_failure(self):
            pass

    import llm_telemetry as lt

    monkeypatch.setattr(agent, "ChatGLM", _FakeLLM)
    monkeypatch.setattr(agent, "_get_circuit_breaker", lambda _m: _FakeCB())
    monkeypatch.setattr(lt, "_emit_llm_usage_event_best_effort", lambda **_k: None)
    monkeypatch.setattr(agent, "_chat_agent_fallback_model", lambda _m: None)

    state = {
        "messages": [HumanMessage(content="me desayuné dos huevos")],
        "sys_prompt": "SYS",
        "user_id": "u-gem",
        "session_id": "s-gem",
        "turn_plate_photos": [],
        "turn_photo_to_log": None,
    }
    monkeypatch.setattr(agent, "_chat_agent_model_name", lambda _u=None: "gemini-4-hydrogen")
    agent.call_model(state)
    assert len(bound_batches[-1]) < len(agent.agent_tools)          # default: gating activo en Gemini
    monkeypatch.setenv("MEALFIT_CHAT_TOOL_GATING_ON_GEMINI", "0")   # opt-in: catálogo fijo
    agent.call_model(state)
    assert len(bound_batches[-1]) == len(agent.agent_tools)
    monkeypatch.delenv("MEALFIT_CHAT_TOOL_GATING_ON_GEMINI", raising=False)

    # En modelos no-Gemini, _select_tools_for_turn sí recorta el catálogo
    monkeypatch.setattr(agent, "_chat_agent_model_name", lambda _u=None: "gpt-6-luna")
    agent.call_model(state)
    assert len(bound_batches[-1]) < len(agent.agent_tools)


def test_llm_telemetry_extracts_cached_and_reasoning_tokens(monkeypatch):
    import db_profiles
    import llm_telemetry as lt

    recorded = []
    monkeypatch.setattr(db_profiles, "log_llm_usage_event", lambda **kw: recorded.append(kw))

    llm = SimpleNamespace(model="gemini-4-hydrogen")
    res_lc = SimpleNamespace(
        usage_metadata={
            "input_tokens": 35000,
            "output_tokens": 150,
            "input_token_details": {"cache_read": 33000},
            "output_token_details": {"reasoning": 45},
        }
    )
    lt._emit_llm_usage_event_best_effort(llm=llm, result=res_lc, duration_s=1.23, node="chat_call_model")
    assert len(recorded) == 1
    assert recorded[0]["input_tokens"] == 35000
    assert recorded[0]["cached_tokens"] == 33000
    assert recorded[0]["metadata"]["reasoning_tokens"] == 45

    res_oai = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "token_usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 60,
                "prompt_tokens_details": {"cached_tokens": 900},
                "completion_tokens_details": {"reasoning_tokens": 12},
            }
        },
    )
    lt._emit_llm_usage_event_best_effort(llm=llm, result=res_oai, duration_s=0.5, node="chat_call_model")
    assert len(recorded) == 2
    assert recorded[1]["input_tokens"] == 1200
    assert recorded[1]["output_tokens"] == 60
    assert recorded[1]["cached_tokens"] == 900
    assert recorded[1]["metadata"]["reasoning_tokens"] == 12


def test_vision_usage_capture_cached_reasoning_and_detail_knob(monkeypatch):
    import vision_agent as va

    cb_plain = va._VisionUsageCapture()
    msg_plain = SimpleNamespace(usage_metadata={"input_tokens": 900, "output_tokens": 40})
    cb_plain.on_chat_model_end(SimpleNamespace(generations=[[SimpleNamespace(message=msg_plain)]], llm_output=None))
    # Sin cache ni reasoning se conserva el dict exacto de P2-VISION-USAGE-TOKENS
    assert cb_plain.usage == {"input_tokens": 900, "output_tokens": 40}

    cb_cached = va._VisionUsageCapture()
    msg_cached = SimpleNamespace(
        usage_metadata={
            "input_tokens": 2300,
            "output_tokens": 180,
            "input_token_details": {"cache_read": 1500},
            "output_token_details": {"reasoning": 32},
        }
    )
    cb_cached.on_chat_model_end(SimpleNamespace(generations=[[SimpleNamespace(message=msg_cached)]], llm_output=None))
    assert cb_cached.usage == {
        "input_tokens": 2300,
        "output_tokens": 180,
        "cached_tokens": 1500,
        "reasoning_tokens": 32,
    }

    # Knob MEALFIT_VISION_IMAGE_DETAIL opt-in
    monkeypatch.delenv("MEALFIT_VISION_IMAGE_DETAIL", raising=False)
    assert va._vision_image_detail() == ""
    monkeypatch.setenv("MEALFIT_VISION_IMAGE_DETAIL", "ultra")
    assert va._vision_image_detail() == ""
    monkeypatch.setenv("MEALFIT_VISION_IMAGE_DETAIL", "medium")
    assert va._vision_image_detail() == "medium"

    sent_payloads = []

    class _FakeVision:
        async def ainvoke(self, messages):
            sent_payloads.append(messages[0].content)
            return {"ok": True}

    monkeypatch.setattr(va, "_resolve_vision_client", lambda _s: _FakeVision())
    asyncio.run(va._invoke_structured_vision(b"\xff\xd8\xff", "describe", dict))
    assert sent_payloads[-1][1]["image_url"]["detail"] == "medium"

    monkeypatch.delenv("MEALFIT_VISION_IMAGE_DETAIL", raising=False)
    asyncio.run(va._invoke_structured_vision(b"\xff\xd8\xff", "describe", dict))
    assert "detail" not in sent_payloads[-1][1]["image_url"]


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — Waterfalls pre-TTFT (Embeddings, turn_cache, Fast-path, Prefetcher)
# ═════════════════════════════════════════════════════════════════════════════

def test_query_embed_ttl_cache_singleflight_and_document_asymmetry(monkeypatch):
    import embeddings_provider as ep
    from turn_cache import turn_cache_scope

    ep.clear_query_embed_cache()
    monkeypatch.setenv("MEALFIT_EMBEDDINGS_PROVIDER", "cohere")
    monkeypatch.setenv("COHERE_API_KEY", "fake-cohere")
    monkeypatch.setenv("MEALFIT_QUERY_EMBED_CACHE_TTL_S", "120")

    calls = {"query": 0, "doc": 0}
    gate = threading.Event()

    class _FakeClient:
        def embed_query(self, text):
            calls["query"] += 1
            gate.wait(timeout=1.0)
            return [0.1, 0.2, 0.3]

        def embed_documents(self, texts):
            calls["doc"] += 1
            return [[0.9, 0.8, 0.7] for _ in texts]

    fake = _FakeClient()
    monkeypatch.setattr(ep, "_build_client", lambda: fake)

    # 1. Singleflight concurrente: 4 hilos paralelos → 1 llamada HTTP real
    def _worker():
        return ep.get_text_embedding("arroz con habichuelas", purpose="query")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_worker) for _ in range(4)]
        time.sleep(0.05)
        gate.set()
        results = [f.result(timeout=3.0) for f in futs]

    assert calls["query"] == 1
    assert all(r == [0.1, 0.2, 0.3] for r in results)

    # 2. Mutar el retorno no corrompe la caché + hit en RAM
    results[0][0] = 999.0
    assert ep.get_text_embedding("arroz con habichuelas", purpose="query") == [0.1, 0.2, 0.3]
    assert calls["query"] == 1

    # 3. purpose="document" NUNCA reusa ni ensucia la caché de query
    doc_vec = ep.get_text_embedding("arroz con habichuelas", purpose="document")
    assert doc_vec == [0.9, 0.8, 0.7] and calls["doc"] == 1
    assert ep.get_text_embedding("arroz con habichuelas", purpose="query") == [0.1, 0.2, 0.3]
    assert calls["query"] == 1

    # 4. clear_query_embed_cache invalida la caché + turn_cache comparte vector en turno
    ep.clear_query_embed_cache()
    with turn_cache_scope():
        v1 = ep.get_text_embedding("sancocho", purpose="query")
        ep.clear_query_embed_cache()
        v2 = ep.get_text_embedding("sancocho", purpose="query")
        assert v1 == v2 == [0.1, 0.2, 0.3]
        assert calls["query"] == 2  # v2 salió del turn_cache
    ep.clear_query_embed_cache()


def test_turn_cache_dedup_profile_inventory_readers_and_handoff(monkeypatch):
    import db
    import db_core
    import db_inventory
    import db_profiles
    import memoria_largo_plazo as mlp
    import nevera_del_coach as ndc
    import nevera_opcional as no
    import plan_mode
    import suplementos
    from turn_cache import clear_reads, consume_handoff, get_cached, set_cached, stash_handoff, turn_cache_scope

    sql_calls = []

    def _stub_sql(query, params=None, fetch_one=False, fetch_all=False):
        sql_calls.append(query)
        if "FROM user_profiles" in query:
            return {
                "id": "u-tc",
                "full_name": "Carlos",
                "plan_tier": "plus",
                "plan_mode": "plan",
                "plan_mode_changed_at": None,
                "nevera_enabled": True,
                "long_term_memory_enabled": True,
                "health_profile": {"allergies": ["Maní"], "country": "DO"},
            }
        if "FROM user_inventory" in query:
            return [
                {
                    "id": "inv1",
                    "ingredient_name": "Huevos",
                    "quantity": 6.0,
                    "unit": "unidad",
                    "source": "manual",
                    "created_at": None,
                    "updated_at": None,
                    "master_ingredient_id": "m1",
                }
            ]
        return [] if fetch_all else {}

    monkeypatch.setattr(db_core, "execute_sql_query", _stub_sql)
    monkeypatch.setattr(db, "execute_sql_query", _stub_sql)
    monkeypatch.setattr(db_profiles, "execute_sql_query", _stub_sql)
    monkeypatch.setattr(db_inventory, "execute_sql_query", _stub_sql)
    monkeypatch.setattr(no, "execute_sql_query", _stub_sql)
    monkeypatch.setattr(plan_mode, "execute_sql_query", _stub_sql)
    monkeypatch.setattr(db_profiles, "connection_pool", object())
    monkeypatch.setattr(db_core, "connection_pool", object())
    monkeypatch.setattr(no, "interruptor_disponible", lambda: True)

    with turn_cache_scope():
        p1 = db_profiles.get_user_profile("u-tc")
        p1["health_profile"]["allergies"].append("MUTATED")
        p2 = db_profiles.get_user_profile("u-tc")
        assert p2["health_profile"]["allergies"] == ["Maní"], "deep-copy evita mutación cruzada"
        assert no.nevera_activa("u-tc") is True
        assert mlp.leer_memoria("u-tc") is True
        assert plan_mode.get_plan_mode("u-tc")["plan_mode"] == "plan"

        inv1 = db_inventory.get_raw_user_inventory("u-tc")
        inv2 = ndc._filas("u-tc")
        assert len(inv1) == 1 and len(inv2) == 1

        profile_queries = [q for q in sql_calls if "FROM user_profiles" in q]
        inv_queries = [q for q in sql_calls if "FROM user_inventory" in q]
        assert len(profile_queries) == 1
        assert len(inv_queries) == 1

        # Prefetch de suplementos y handoff
        set_cached("prefetch_suppl", "u-tc", "SUPPL_PREFETCHED")
        assert suplementos.bloque_para_chat("u-tc") == "SUPPL_PREFETCHED"

        stash_handoff("u-tc", "s-tc")
        clear_reads()
        assert get_cached("profile", "u-tc") == (False, None)

    snap = consume_handoff("u-tc", "s-tc")
    assert isinstance(snap, dict) and ("profile", "u-tc") in snap
    assert consume_handoff("u-tc", "s-tc") is None  # one-shot


def test_rag_es_presente_sin_rag_conservative_boundaries():
    import agent

    # Presentes auto-contenidos / dudas de saldo hoy → True
    for p in (
        "me desayuné 2 huevos revueltos con aguacate",
        "almorcé pollo guisado con arroz",
        "me tomé 2 vasos de agua",
        "anota un café con leche",
        "¿cuántas calorías me quedan hoy?",
        "¿cómo voy hoy con mi proteína?",
        "¿qué me falta hoy por comer?",
    ):
        assert agent._rag_es_presente_sin_rag(p) is True, p

    # Pasado, preferencias, alergias, planes, suplementos, referencias o largo → False
    for p in (
        "hola",
        "ayer desayuné mangú",
        "soy alérgico al maní",
        "no me gusta el hígado",
        "me tomé 5g de creatina",
        "cámbiame el desayuno de hoy",
        "dos paquetes de las de ayer",
        "¿puedo comer pan con mi dieta?",
        "me comí un huevo " + ("extra " * 45),
    ):
        assert agent._rag_es_presente_sin_rag(p) is False, p


def test_chat_paths_fastpath_prefetcher_and_byte_identical_prefix_e2e(monkeypatch):
    import agent
    import db_core
    import db_inventory
    import db_plans
    import nevera_opcional as no
    import shopping_calculator
    from turn_cache import current_turn_cache

    router_calls = []
    embed_calls = []

    class _Graph:
        def get_state(self, _config):
            return SimpleNamespace(values={})

        def invoke(self, inputs, **_k):
            # Cuando compila e invoca el grafo, la caché temporal de pre-turno ya se cerró
            assert current_turn_cache() is None
            captured.update(inputs)
            raise _PromptCaptured

        stream = invoke

    class _Builder:
        def compile(self, **_k):
            return _Graph()

    captured = {}

    monkeypatch.setattr(no, "nevera_activa", lambda _uid: False)
    monkeypatch.setattr(agent, "_memoria_activa_para_chat", lambda _uid: True)
    monkeypatch.setattr(db_core, "connection_pool", None)
    monkeypatch.setattr(db_inventory, "get_user_inventory", lambda _uid: [])
    monkeypatch.setattr(db_plans, "get_latest_usable_meal_plan_with_id", lambda _uid: None)
    monkeypatch.setattr(shopping_calculator, "aggregate_shopping_list", lambda items, **k: list(items))
    monkeypatch.setattr(agent, "build_memory_context", lambda *_a: {"recent_messages": [], "summary_context": ""})
    monkeypatch.setattr(agent, "classify_sentiment", lambda _p: {"instruction": "", "sentiment": "neutral"})

    def _spy_router(p):
        router_calls.append(p)
        return {"skip": False, "query": p}

    def _spy_embed(q):
        embed_calls.append(q)
        return None

    monkeypatch.setattr(agent, "rag_query_router", _spy_router)
    monkeypatch.setattr(agent, "get_embedding", _spy_embed)
    monkeypatch.setattr(agent, "get_multimodal_embedding", lambda _q: None)
    monkeypatch.setattr(agent, "_emit_chat_stream_total_duration_best_effort", lambda *_a: None)
    monkeypatch.setattr(agent, "chat_builder", _Builder())
    monkeypatch.setattr(agent, "chat_checkpoint_pool", None)
    monkeypatch.setattr(agent, "connection_pool", None)
    monkeypatch.delenv("MEALFIT_CHAT_FASTPATH_PRESENT_LOGS", raising=False)
    monkeypatch.setenv("MEALFIT_CHAT_SPLIT_VOLATILE_SYS_PROMPT", "1")   # [998] apagado por defecto
    monkeypatch.delenv("MEALFIT_CHAT_PREFETCH_PARALLEL", raising=False)

    # 1. Stream E2E con 2 usuarios distintos + log presente rápido (skip router & embed)
    prompts_by_user = {}
    for uid, d in (("user-alpha-1", "2026-11-10"), ("user-beta-2", "2026-11-13")):
        captured.clear()
        with pytest.raises(_PromptCaptured):
            list(
                agent.chat_with_agent_stream(
                    session_id=f"s-{uid}",
                    prompt="me desayuné 2 huevos revueltos",
                    user_id=uid,
                    form_data={"country": "DO"},
                    local_date=d,
                    tz_offset=240,
                )
            )
        prompts_by_user[uid] = captured["sys_prompt"]

    assert router_calls == [] and embed_calls == []
    assert agent._VOLATILE_SYS_SPLIT_MARKER in prompts_by_user["user-alpha-1"]
    msgs_a = agent._llm_messages_from_state([HumanMessage(content="me desayuné 2 huevos revueltos")], prompts_by_user["user-alpha-1"])
    msgs_b = agent._llm_messages_from_state([HumanMessage(content="me desayuné 2 huevos revueltos")], prompts_by_user["user-beta-2"])
    # El SystemMessage[0] enviado al LLM es 100% byte-idéntico entre user-alpha-1 y user-beta-2
    assert msgs_a[0].content == msgs_b[0].content
    assert "user-alpha-1" not in msgs_a[0].content
    assert "user-alpha-1" in msgs_a[-1].content and "user-beta-2" in msgs_b[-1].content

    # 2. Non-stream E2E verifica mismo comportamiento + consulta histórica sí dispara router
    captured.clear()
    with pytest.raises(_PromptCaptured):
        agent.chat_with_agent(
            session_id="s-nonstream",
            prompt="¿puedo comer pan con mi alergia?",
            user_id="user-gamma-3",
            form_data={"country": "DO"},
            local_date="2026-11-13",
            tz_offset=240,
        )
    assert router_calls == ["¿puedo comer pan con mi alergia?"]
    assert agent._VOLATILE_SYS_SPLIT_MARKER in captured["sys_prompt"]
    msgs_g = agent._llm_messages_from_state([HumanMessage(content="¿puedo comer pan con mi alergia?")], captured["sys_prompt"])
    assert msgs_g[0].content != "" and "user-gamma-3" not in msgs_g[0].content
    assert "user-gamma-3" in msgs_g[-1].content


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 1B & ETAPA 3 — Cross-User Anchor, Coach Compact & Slim Tool Schemas
# ═════════════════════════════════════════════════════════════════════════════

def test_anclar_manual_estatico_cross_user_idempotent_and_safe(monkeypatch):
    import gemini_hydrogen as gh

    monkeypatch.setenv("MEALFIT_GEMINI_CROSS_USER_CACHE_ANCHOR", "1")   # [998] apagado por defecto
    monkeypatch.delenv("MEALFIT_GEMINI_CACHE_ANCHOR_MIN_CHARS", raising=False)

    manual = "MANUAL_ESTATICO_BIOBOROS " * 400  # ~10k chars (> 7000 default)
    mm_user = {"role": "user", "content": [{"type": "text", "text": "Desayuno"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,X"}}]}

    payload_a = {"messages": [{"role": "system", "content": manual}, {"role": "user", "content": "Usuario A: comí pan"}]}
    payload_b = {"messages": [{"role": "system", "content": manual}, mm_user]}

    assert gh.anclar_manual_estatico(payload_a) is True
    assert gh.anclar_manual_estatico(payload_b) is True
    # Prefijo anclado messages[0..2] idéntico entre usuarios
    assert len(payload_a["messages"]) == 4
    assert payload_a["messages"][:3] == payload_b["messages"][:3]
    assert payload_a["messages"][0]["role"] == "system" and len(payload_a["messages"][0]["content"]) < 250
    assert payload_a["messages"][1]["role"] == "user" and "<manual_operativo_bioboros>" in payload_a["messages"][1]["content"]
    assert payload_a["messages"][2]["role"] == "assistant"
    assert payload_a["messages"][3]["content"] == "Usuario A: comí pan"
    assert isinstance(payload_b["messages"][3]["content"], list)

    # Idempotente si se vuelve a llamar sobre un payload ya anclado
    assert gh.anclar_manual_estatico(payload_a) is False
    assert len(payload_a["messages"]) == 4

    # Prompts cortos (ej. visión / título) NO se reescriben
    short_payload = {"messages": [{"role": "system", "content": "Prompt corto visión"}, {"role": "user", "content": "foto"}]}
    assert gh.anclar_manual_estatico(short_payload) is False
    assert len(short_payload["messages"]) == 2

    # Apagable por knob
    monkeypatch.setenv("MEALFIT_GEMINI_CROSS_USER_CACHE_ANCHOR", "false")
    payload_off = {"messages": [{"role": "system", "content": manual}, {"role": "user", "content": "hola"}]}
    assert gh.anclar_manual_estatico(payload_off) is False


def test_coach_compact_retains_rules_reduces_size_and_cleans_voice_mode():
    from prompts.chat_agent import CHAT_VOICE_MODE_PROMPT, _CHAT_CALL_MODE_RULES
    from prompts.coach_compact import build_coach_static_manual

    text_do = build_coach_static_manual(country="DO", is_call_mode=False, stream=True)
    text_do_2 = build_coach_static_manual(country="DO", is_call_mode=False, stream=True)
    voice_do = build_coach_static_manual(country="DO", is_call_mode=True, stream=True)

    assert text_do == text_do_2
    assert len(text_do) < 16000, f"Manual compacto demasiado grande: {len(text_do)}"
    assert voice_do.startswith(CHAT_VOICE_MODE_PROMPT[:200])
    assert _CHAT_CALL_MODE_RULES.strip() in voice_do
    assert "FORMATO VISUAL:" not in voice_do

    # Retención de reglas y fuentes clave
    for token in (
        "https://pubmed.ncbi.nlm.nih.gov/2305711/",
        "https://fdc.nal.usda.gov/",
        "log_consumed_meal",
        "correct_consumed_meal",
        "proponer_comida",
        "guardar_suplemento",
        "buscar_etiqueta_en_internet",
        "[UI_ACTION: REFRESH_INVENTORY]",
        "[UI_ACTION: REFRESH_HYDRATION]",
        "BB-034",
        "BB-058",
        "BB-059",
        "BB-067",
        "P1-DIARY-DELTA-CORRECTION",
        "P1-PROPORTIONAL-COOKING-FAT-SCALE",
    ):
        assert token in text_do, f"Falta {token!r} en manual compacto"


def test_agent_tools_wire_json_compactado_conservando_las_reglas_de_comportamiento():
    """[P1-PLAN-LOTE-998] Las informativas quedan compactas; las 11 que llevan reglas (registro, corrección, Nevera,
    suplementos, ajustes, regenerar...) conservan su texto original: era el que nació de errores reales."""
    import json
    from langchain_core.utils.function_calling import convert_to_openai_tool
    import tools

    wire_json = json.dumps([convert_to_openai_tool(t) for t in tools.agent_tools], ensure_ascii=False)
    assert len(tools.agent_tools) == 19
    assert len(wire_json) < 27000, f"Wire JSON tools muy pesado: {len(wire_json)}"   # legacy: 29.9 k; compacto total: 16.3 k
    docs = {t.name: (t.description or "") for t in tools.agent_tools}
    for nombre, frase in (
        ("correct_consumed_meal", "NUNCA de qué estabas preguntando"),
        ("correct_consumed_meal", "jamás borres por nombres raros o erratas"),
        ("log_consumed_meal", "Un reintento de la misma escritura se omite"),
        ("log_consumed_meal", "Un alimento distinto se añade sin pedir permiso"),
        ("modify_pantry_inventory", "Ninguno de los dos cuenta como comida consumida"),
        ("consultar_dia_del_plan", "NO prueba de que el usuario se"),
        ("abrir_pantalla_de_la_app", "borrar la cuenta"),
        ("mark_shopping_list_purchased", "Es idempotente dentro del ciclo de compra"),
    ):
        assert frase in " ".join(docs[nombre].split()), f"{nombre}: perdió «{frase}»"

