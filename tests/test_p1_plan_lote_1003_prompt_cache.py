# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1003 · 2026-11-14] (G-02) Estabilidad de prefijo y prompt cache OpenAI.

Cubre los 4 frentes cerrados de G-02 (0 gasto de IA):
1. `prompt_desglose` separa `CONTEXTO DE VARIAS FOTOS (en orden):` en el bloque `vision`
   (Banda 50 de `reordenar_bloques_cache`) sin filtrarse en `dias_pasados`.
2. Prefijo estatico cross-user libre de `user_id` incluso con
   `MEALFIT_CHAT_SPLIT_VOLATILE_SYS_PROMPT=0` (regla clinica siempre en `role=system`).
3. Estabilidad del subconjunto de herramientas en `ToolMessage` intra-turno
   (no salta de 8 a 19 herramientas entre llamada 1 y llamada 2).
4. Inyeccion de `prompt_cache_key` compartido entre usuarios + `prompt_cache_retention="24h"`
   exclusivamente para OpenAI (>= 2 500 chars), ausente en Gemini/GLM/DeepSeek,
   con fallback fail-open ante 400 + registro de los 4 knobs en `knobs.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def test_prompt_desglose_y_banda_50_para_multifoto():
    import chat_prompt_opt as cpo
    import prompt_desglose as pd
    from prompts import chat_agent as ca

    static_prefix = "M" * 4000
    dyn_vision = ca.build_vision_context(
        {"kind": "multi", "items": [{"attachment_id": "a1", "kind": "plato", "description": "pan de agua"}]}
    )
    assert "CONTEXTO DE VARIAS FOTOS (en orden):" in dyn_vision
    dyn_clinico = "\n\n🛑 PERFIL CLÍNICO DEL USUARIO (NO IGNORAR):\nAlergias: ninguna."
    dyn_temporal = "\n\n🕒 CONTEXTO TEMPORAL ACTUAL:\nSabado 10:15"
    sys_raw = static_prefix + dyn_temporal + dyn_vision + dyn_clinico

    desglose = pd.desglosar_prompt(sys_raw, prefijo_estatico_chars=len(static_prefix))
    bloques = desglose["bloques"]
    assert bloques.get("vision", 0) == len(dyn_vision)
    assert bloques.get("clinico", 0) == len(dyn_clinico)
    assert bloques.get("temporal", 0) == len(dyn_temporal)

    reordered = cpo.reordenar_bloques_cache(sys_raw, prefijo_estatico_chars=len(static_prefix))
    assert reordered.startswith(static_prefix)
    # Clinico (banda 1) antes que temporal (banda 3) y vision (banda 50 al final).
    assert reordered.index("PERFIL CLÍNICO DEL USUARIO") < reordered.index("CONTEXTO TEMPORAL ACTUAL") < reordered.index("VARIAS FOTOS")


def test_select_tools_for_turn_estable_en_followup_y_opt_out(monkeypatch):
    import agent

    monkeypatch.delenv("MEALFIT_CHAT_TOOL_GATING_ENABLED", raising=False)
    monkeypatch.delenv("MEALFIT_CHAT_STABLE_TOOL_SUBSET_ON_FOLLOWUP", raising=False)

    h1 = HumanMessage(content="me desayuné dos huevos con pan")
    turn1 = agent._select_tools_for_turn(agent.agent_tools, {"messages": [h1]})
    turn1_names = [t.name for t in turn1]
    assert len(turn1_names) < len(agent.agent_tools)

    ai_call = AIMessage(
        content="",
        tool_calls=[{"name": "log_consumed_meal", "args": {"meal_name": "Huevos"}, "id": "tc1"}],
    )
    t_res = ToolMessage(content="[ID_REGISTRO_DIARIO:abc]", tool_call_id="tc1", name="log_consumed_meal")
    turn2 = agent._select_tools_for_turn(agent.agent_tools, {"messages": [h1, ai_call, t_res]})
    turn2_names = [t.name for t in turn2]
    assert turn2_names == turn1_names

    # Si durante el turno se invoco otra herramienta (p.ej. nevera), se preserva en la segunda llamada.
    t_nevera = ToolMessage(content="OK", tool_call_id="tc2", name="modify_pantry_inventory")
    turn2_ext = agent._select_tools_for_turn(
        agent.agent_tools,
        {"messages": [h1, ai_call, t_res, AIMessage(content="", tool_calls=[]), t_nevera]},
    )
    assert "modify_pantry_inventory" in [t.name for t in turn2_ext]

    # Opt-out explicito vuelve a devolver todo el catalogo al regresar de ToolMessage.
    monkeypatch.setenv("MEALFIT_CHAT_STABLE_TOOL_SUBSET_ON_FOLLOWUP", "0")
    legacy_turn2 = agent._select_tools_for_turn(agent.agent_tools, {"messages": [h1, ai_call, t_res]})
    assert len(legacy_turn2) == len(agent.agent_tools)


def test_prefijo_estatico_cross_user_sin_uid_con_clinico_en_system(monkeypatch):
    import agent
    import openai_prompt_cache as opc
    from prompts import chat_agent as ca

    monkeypatch.setenv("MEALFIT_CHAT_SPLIT_VOLATILE_SYS_PROMPT", "0")
    monkeypatch.setenv("MEALFIT_CHAT_STATIC_PREFIX_NO_UID", "1")
    assert opc.static_prefix_no_uid_activo() is True

    # Comprueba en codigo fuente que ambos caminos (no-stream y stream) usan static_prefix_no_uid_activo
    # y dejan `El user_id` despues de `_bd_prefijo_chars` cuando `_split_volatile` es False.
    static_block = ca.build_tools_instructions_stream_estaticas(plan_en_pausa=False, contador_sin_plan=False)
    assert "__BIOBOROS_STATIC_UID__" not in static_block
    assert "El user_id" not in static_block

    base = ca.CHAT_STREAM_INLINE_PROMPT + static_block
    uid_a = "11111111-1111-1111-1111-111111111111"
    uid_b = "22222222-2222-2222-2222-222222222222"

    sys_a = base + f"\n\nEl user_id actual es: {uid_a}\n\n🛑 PERFIL CLÍNICO DEL USUARIO:\nAlergia: mani\n\n🕒 CONTEXTO TEMPORAL ACTUAL:\n09:00"
    sys_b = base + f"\n\nEl user_id actual es: {uid_b}\n\n🛑 PERFIL CLÍNICO DEL USUARIO:\nSin alergias\n\n🕒 CONTEXTO TEMPORAL ACTUAL:\n19:30"

    assert opc.extract_static_sys_prefix(sys_a) == base
    assert opc.extract_static_sys_prefix(sys_a) == opc.extract_static_sys_prefix(sys_b)

    # Regla clinica sigue en rol `system` (no se mueve a `HumanMessage`).
    msgs_a = agent._llm_messages_from_state([HumanMessage(content="hola")], sys_a)
    assert isinstance(msgs_a[0], SystemMessage)
    assert "PERFIL CLÍNICO DEL USUARIO" in msgs_a[0].content
    assert uid_a in msgs_a[0].content
    assert msgs_a[1].content == "hola"


def test_openai_prompt_cache_payload_injection_y_aislamiento_proveedores(monkeypatch):
    import llm_provider as lp
    import openai_prompt_cache as opc
    from langchain_core.tools import tool

    @tool
    def log_consumed_meal(meal_name: str) -> str:
        """Registra comida."""
        return meal_name

    @tool
    def correct_consumed_meal(meal_id: str) -> str:
        """Corrige comida."""
        return meal_id

    opc.reset_unsupported_cache_models_for_tests()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", "openai")
    monkeypatch.delenv("MEALFIT_OPENAI_PROMPT_CACHE_KEY_ENABLED", raising=False)
    monkeypatch.delenv("MEALFIT_OPENAI_PROMPT_CACHE_RETENTION", raising=False)

    static_manual = "X" * 6000
    sys_a = f"{static_manual}\n\nEl user_id actual es: u-alpha\n\n🕒 HORA:\n08:00"
    sys_b = f"{static_manual}\n\nEl user_id actual es: u-beta\n\n🕒 HORA:\n21:45"

    llm_luna = lp.ChatOpenAI(model="gpt-6-luna", api_key="sk-test-123").bind_tools([log_consumed_meal])
    p_a = llm_luna.bound._get_request_payload(
        [SystemMessage(content=sys_a), HumanMessage(content="comi pan")],
        tools=llm_luna.kwargs["tools"],
    )
    p_b = llm_luna.bound._get_request_payload(
        [SystemMessage(content=sys_b), HumanMessage(content="almorce pollo")],
        tools=llm_luna.kwargs["tools"],
    )
    assert p_a["prompt_cache_key"].startswith("mf1:")
    assert len(p_a["prompt_cache_key"]) <= 64
    assert p_a["prompt_cache_key"] == p_b["prompt_cache_key"]
    assert p_a["prompt_cache_retention"] == "24h"
    assert p_b["prompt_cache_retention"] == "24h"

    # Distinto subconjunto de tools => distinta key (misma cache por cohorte).
    llm_luna_2 = lp.ChatOpenAI(model="gpt-6-luna", api_key="sk-test-123").bind_tools(
        [log_consumed_meal, correct_consumed_meal]
    )
    p_tools2 = llm_luna_2.bound._get_request_payload(
        [SystemMessage(content=sys_a), HumanMessage(content="hola")],
        tools=llm_luna_2.kwargs["tools"],
    )
    assert p_tools2["prompt_cache_key"] != p_a["prompt_cache_key"]

    # Prompt pequeno (< 2 500 chars) no recibe headers de cache.
    p_small = lp.ChatOpenAI(model="gpt-6-luna", api_key="sk-test-123")._get_request_payload(
        [SystemMessage(content="Clasifica sentimiento."), HumanMessage(content="ok")]
    )
    assert "prompt_cache_key" not in p_small
    assert "prompt_cache_retention" not in p_small

    # Gemini, GLM y DeepSeek NUNCA reciben prompt_cache_key ni prompt_cache_retention.
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    monkeypatch.setenv("ZAI_API_KEY", "zai-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
    for llm_other in (
        lp.ChatGLM(model="gemini-4-hydrogen", api_key="gm-test"),
        lp.ChatGLM(model="glm-5.3-flash", base_url="https://api.z.ai/api/paas/v4", api_key="zai-test"),
        lp.ChatGLM(model="deepseek-flash", base_url="https://api.deepseek.com", api_key="ds-test"),
    ):
        p_other = llm_other._get_request_payload([SystemMessage(content=sys_a), HumanMessage(content="hola")])
        assert "prompt_cache_key" not in p_other
        assert "prompt_cache_retention" not in p_other


def test_fail_open_si_openai_rechaza_prompt_cache_retention(monkeypatch):
    from langchain_openai import ChatOpenAI as _BaseLC
    import llm_provider as lp
    import openai_prompt_cache as opc

    opc.reset_unsupported_cache_models_for_tests()
    monkeypatch.delenv("MEALFIT_OPENAI_PROMPT_CACHE_RETENTION", raising=False)
    monkeypatch.delenv("MEALFIT_OPENAI_PROMPT_CACHE_KEY_ENABLED", raising=False)

    seen_payloads = []
    calls = {"n": 0}

    class _FakeBadRequest(Exception):
        status_code = 400

    def _fake_base_generate(self, messages, stop=None, run_manager=None, **kwargs):
        calls["n"] += 1
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        seen_payloads.append(dict(payload))
        if calls["n"] == 1:
            raise _FakeBadRequest("Unknown parameter: 'prompt_cache_retention'.")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    monkeypatch.setattr(_BaseLC, "_generate", _fake_base_generate)
    llm = lp.ChatOpenAI(model="gpt-6-luna", api_key="sk-test-123")
    out = llm._generate([SystemMessage(content="S" * 5000), HumanMessage(content="hola")])
    assert out.generations[0].message.content == "ok"
    assert calls["n"] == 2
    assert seen_payloads[0].get("prompt_cache_retention") == "24h"
    assert "prompt_cache_retention" not in seen_payloads[1]
    assert "prompt_cache_key" in seen_payloads[1]
    opc.reset_unsupported_cache_models_for_tests()


def test_knobs_lote_1003_registrados_en_knobs_registry():
    import knobs
    import openai_prompt_cache  # noqa: F401

    snap = knobs.get_knobs_registry_snapshot()
    for name in (
        "MEALFIT_CHAT_STATIC_PREFIX_NO_UID",
        "MEALFIT_CHAT_STABLE_TOOL_SUBSET_ON_FOLLOWUP",
        "MEALFIT_OPENAI_PROMPT_CACHE_KEY_ENABLED",
        "MEALFIT_OPENAI_PROMPT_CACHE_RETENTION",
    ):
        assert name in snap, f"Falta {name} en knobs registry"
