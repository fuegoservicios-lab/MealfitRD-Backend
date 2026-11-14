"""[P1-PLAN-LOTE-1009 · 2026-11-14] Regression tests for G-13 (Responses API), G-05 (Vision usage_metadata capture + image prep speed), G-20 (Photo/Web source-intent guard), and G-07 (Deterministic single-call fast tool confirmation)."""
from __future__ import annotations

import asyncio
import io
import os
import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _fake_jpeg(w: int = 1600, h: int = 1200) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (180, 120, 60)).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_g13_responses_api_for_gpt61_terra(monkeypatch):
    """`gpt-6.1-terra` usa `use_responses_api=True` y conserva razonamiento 'low' incluso con tools."""
    import llm_provider
    import chat_hard_turn as cht
    import agent

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

    llm_terra = llm_provider.build_chat_llm(
        "gpt-6.1-terra", reasoning_effort="low", max_tokens=1000
    )
    assert getattr(llm_terra, "use_responses_api", False) is True
    assert getattr(llm_terra, "reasoning_effort", None) == "low"

    # `_get_request_payload` con `use_responses_api=True` NO pisa reasoning_effort="none".
    payload_holder: dict = {}

    def _stub_super_payload(self, input_, *, stop=None, **kwargs):
        return {"model": "gpt-6.1-terra", "tools": [{"type": "function"}], "reasoning_effort": "low"}

    monkeypatch.setattr(
        llm_provider.ChatOpenAI.__mro__[1], "_get_request_payload", _stub_super_payload, raising=False
    )
    # Llamada directa sobre ChatOpenAI (clase subclaseada en llm_provider)
    built_p = llm_terra._get_request_payload([HumanMessage(content="hola")])
    assert built_p["reasoning_effort"] == "low"

    # `gpt-6-luna` con Chat Completions sigue poniendo reasoning_effort="none" cuando lleva tools
    llm_luna = llm_provider.build_chat_llm("gpt-6-luna")
    assert getattr(llm_luna, "use_responses_api", False) in (False, None)
    built_l = llm_luna._get_request_payload([HumanMessage(content="hola")])
    assert built_l["reasoning_effort"] == "none"

    # `construir_llm_turno_dificil` y `_build_chat_fallback_llm` habilitan Responses API en `gpt-6.1-terra`
    captured = {}

    def _spy_build(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return SimpleNamespace(model=model, **kwargs)

    monkeypatch.setattr(llm_provider, "build_chat_llm", _spy_build)
    monkeypatch.setattr(agent, "build_chat_llm", _spy_build)

    cht.construir_llm_turno_dificil(
        "gpt-6.1-terra",
        {"reasoning_effort": "low", "max_tokens": 900},
        timeout_s=30.0,
    )
    assert captured["kwargs"].get("use_responses_api") is True

    agent._build_chat_fallback_llm("gpt-6.1-terra")
    assert captured["kwargs"].get("use_responses_api") is True


def test_g05_vision_include_raw_captures_usage_even_when_callback_misses(monkeypatch):
    """`_invoke_structured_vision` captura tokens desde `AIMessage.usage_metadata` cuando callback no reporta usage."""
    import vision_agent as va

    async def _run():
        class _RawFakeClient:
            async def ainvoke(self, msgs):
                raw_ai = AIMessage(
                    content="",
                    usage_metadata={
                        "input_tokens": 1320,
                        "output_tokens": 95,
                        "total_tokens": 1415,
                        "input_token_details": {"cache_read": 400},
                    },
                )
                parsed_obj = va._MealVisionResult(
                    description="Pollo con arroz blanco y habichuelas",
                    is_food=True,
                    photo_kind="plato",
                    meal_name="Pollo guisado con arroz y habichuelas",
                    calories=620,
                    protein=42,
                    carbs=65,
                    healthy_fats=16,
                )
                return {"raw": raw_ai, "parsed": parsed_obj, "parsing_error": None}

        monkeypatch.setattr(va, "_resolve_vision_client", lambda schema: _RawFakeClient())
        monkeypatch.setattr(va, "is_vision_enabled", lambda: True)

        with va.vision_usage_scope():
            out = await va.process_image_with_vision(_fake_jpeg())
            assert out["is_food"] is True
            usage = va.get_last_vision_usage()
            assert usage == {
                "input_tokens": 1320,
                "output_tokens": 95,
                "cached_tokens": 400,
            }

    asyncio.run(_run())


def test_g05_image_prep_fast_resize_quality_and_contract():
    """`prepare_image_for_vision` comprime más ligero (quality=80) conservando max_side=1024."""
    import image_prep

    assert image_prep._JPEG_QUALITY == 80
    out, info = image_prep.prepare_image_for_vision(_fake_jpeg(2000, 1500))
    assert info["resized"] is True
    assert info["sent_wh"] == (1024, 768)
    assert len(out) > 0


def test_g20_source_intent_guard_blocks_passive_profile_and_app_mutations():
    """Fotos y resultados web no mutan perfil ni ajustes de app sin intención explícita del usuario."""
    import agent

    # 1. Foto de etiqueta nutricional sin petición de perfil -> BLOQUEADO
    msgs_photo = [HumanMessage(content="📷")]
    state_photo = {"turn_plate_photos": [{"attachment_id": "att-1"}], "turn_photo_to_log": False}
    block1 = agent._guard_source_intent_mutation(
        "update_form_field", {"field": "allergies", "new_value": "mani"}, state_photo, msgs_photo
    )
    assert block1 is not None and "BLOQUEADO" in block1

    # 2. Resultado de buscar_etiqueta_en_internet sin petición de ajuste -> BLOQUEADO
    msgs_web = [
        HumanMessage(content="Qué tiene el yogurt greco Oikos?"),
        AIMessage(content="", tool_calls=[{"id": "tc1", "name": "buscar_etiqueta_en_internet", "args": {"query": "oikos"}}]),
        ToolMessage(content="Fuente web: etiqueta nutricional 15g proteína, puede contener trazas de nueces.", tool_call_id="tc1"),
    ]
    block2 = agent._guard_source_intent_mutation(
        "cambiar_ajuste_de_la_app", {"ajuste": "tema", "valor": "light"}, {}, msgs_web
    )
    assert block2 is not None and "BLOQUEADO" in block2

    # 3. Con foto pero petición EXPLÍCITA del usuario ("guarda en mi perfil soy alérgico al maní") -> PERMITIDO
    msgs_explicit = [HumanMessage(content="Guarda en mi perfil que soy alérgico al maní")]
    allow1 = agent._guard_source_intent_mutation(
        "update_form_field", {"field": "allergies", "new_value": "mani"}, state_photo, msgs_explicit
    )
    assert allow1 is None


def test_g07_fast_tool_confirmation_skips_second_llm_call(monkeypatch):
    """Una llamada limpia a `log_water_glass` emite confirmación determinista y enruta directo a `END`."""
    import agent

    class _FakeWaterTool:
        name = "log_water_glass"

        def invoke(self, args):
            return "Listo: se sumaron 1 vaso(s). El usuario ahora lleva 5 de 8 vasos hoy."

    monkeypatch.setattr(agent, "agent_tools", [_FakeWaterTool()])
    monkeypatch.delenv("MEALFIT_CHAT_FAST_TOOL_CONFIRM_ENABLED", raising=False)

    ai_tc = AIMessage(
        content="",
        tool_calls=[{"id": "wc-1", "name": "log_water_glass", "args": {"user_id": "u1", "count_delta": 1}}],
    )
    state = {
        "user_id": "u1",
        "session_id": "s1",
        "sys_prompt": "Eres un coach.",
        "modo_voz": False,
        "messages": [HumanMessage(content="Me tomé un vaso de agua"), ai_tc],
    }
    res = agent.execute_tools(state)
    msgs = res["messages"]
    assert len(msgs) == 2
    assert isinstance(msgs[0], ToolMessage)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].additional_kwargs.get("fast_tool_confirm") is True
    assert "5 de 8 vasos" in msgs[1].content

    assert agent.route_after_tools({"messages": state["messages"] + msgs}) == agent.END

    # Con pregunta compuesta ("y qué como ahora?") NO salta call_model
    state_q = dict(state)
    state_q["messages"] = [HumanMessage(content="Me tomé un vaso de agua y qué ceno?"), ai_tc]
    res_q = agent.execute_tools(state_q)
    assert len(res_q["messages"]) == 1
    assert agent.route_after_tools({"messages": state_q["messages"] + res_q["messages"]}) == "call_model"

    # En modo voz NO salta call_model
    state_v = dict(state)
    state_v["modo_voz"] = True
    res_v = agent.execute_tools(state_v)
    assert len(res_v["messages"]) == 1
    assert agent.route_after_tools({"messages": state_v["messages"] + res_v["messages"]}) == "call_model"
