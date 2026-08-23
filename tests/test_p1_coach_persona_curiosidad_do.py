"""[P1-COACH-PERSONA-CURIOSIDAD-DO · 2026-08-23]

La personalidad de curiosidad debe conservarse byte-idéntica para DO y quedar
neutra para países beta en las dos ramas reales del prompt STREAM.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


_BACKEND = Path(__file__).resolve().parents[1]
_PERSONALITIES = ("guilt", "motivation", "curiosity", "frustration", "sadness", "neutral")


class _PromptCaptured(RuntimeError):
    pass


def _render_real_stream_prompt(monkeypatch, country: str, personality: str, static_prefix: bool) -> str:
    import agent
    import prompts.plan_generator as plan_generator
    from prompts.sentiment import PERSONALITY_PROFILES

    profile = {**PERSONALITY_PROFILES[personality], "sentiment": personality}
    captured = {}

    class _Graph:
        def get_state(self, _config):
            return SimpleNamespace(values={})

        def stream(self, inputs, **_kwargs):
            captured["prompt"] = inputs["sys_prompt"]
            raise _PromptCaptured

    class _Builder:
        def compile(self, **_kwargs):
            return _Graph()

    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(agent, "_plan_vigente_para_prompt", lambda *_args: None)
    monkeypatch.setattr(
        agent,
        "build_memory_context",
        lambda *_args: {"recent_messages": [], "summary_context": ""},
    )
    monkeypatch.setattr(agent, "classify_sentiment", lambda _prompt: dict(profile))
    monkeypatch.setattr(agent, "_chat_prompt_static_prefix", lambda: static_prefix)
    monkeypatch.setattr(agent, "CHAT_STREAM_INLINE_PROMPT", "PROMPT_BASE_SIN_PAIS")
    monkeypatch.setattr(agent, "culinary_knowledge_base_for_country", lambda _country: "")
    for name in (
        "build_tools_instructions_stream",
        "build_temporal_context",
        "build_vision_context",
        "build_circadian_context",
        "build_temporal_proactive_context",
        "build_inventory_context",
        "build_user_identity_context",
        "build_clinical_guard_context",
        "build_language_directive",
    ):
        monkeypatch.setattr(agent, name, lambda *_args, **_kwargs: "")
    monkeypatch.setattr(plan_generator, "build_super_personalization_context", lambda _form: "")
    monkeypatch.setattr(agent, "chat_builder", _Builder())
    monkeypatch.setattr(agent, "chat_checkpoint_pool", None)
    monkeypatch.setattr(agent, "connection_pool", None)

    stream = agent.chat_with_agent_stream(
        "session-test",
        "¿Por qué importa la proteína?",
        user_id="guest",
        form_data={"country": country},
    )
    with pytest.raises(_PromptCaptured):
        list(stream)
    return captured["prompt"]


@pytest.mark.parametrize("static_prefix", (True, False), ids=("static", "legacy"))
def test_stream_beta_renderiza_las_seis_personalidades_sin_gentilicio(monkeypatch, static_prefix):
    for personality in _PERSONALITIES:
        prompt = _render_real_stream_prompt(monkeypatch, "ES", personality, static_prefix)
        assert not re.search(r"dominican\w*", prompt, flags=re.IGNORECASE), personality
    curiosity = _render_real_stream_prompt(monkeypatch, "ES", "curiosity", static_prefix)
    assert "alimentos cotidianos que el usuario reconozca" in curiosity


@pytest.mark.parametrize("static_prefix", (True, False), ids=("static", "legacy"))
def test_stream_do_conserva_las_seis_instrucciones_byte_identicas(monkeypatch, static_prefix):
    from prompts.sentiment import PERSONALITY_PROFILES

    for personality in _PERSONALITIES:
        instruction = PERSONALITY_PROFILES[personality]["instruction"]
        prompt = _render_real_stream_prompt(monkeypatch, "DO", personality, static_prefix)
        if instruction:
            assert instruction in prompt, personality
        else:
            assert prompt == "PROMPT_BASE_SIN_PAIS\n"


def test_wiring_resuelve_pais_antes_de_cosechar_sentimiento_y_normaliza_una_vez():
    source = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "chat_with_agent_stream"
    )
    country_assignments = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_coach_country" for target in node.targets)
    ]
    result_calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "result"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_f_sent"
    ]
    normalizations = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "sentiment_result"
            for target in node.targets
        )
    ]
    assert len(country_assignments) == 2  # success + fail-safe del único try
    assert len(result_calls) == 1
    assert max(node.lineno for node in country_assignments) < result_calls[0].lineno
    assert len(normalizations) == 1
    assert result_calls[0].lineno < normalizations[0].lineno


def test_marker_movil_del_gap():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P1-COACH-PERSONA-CURIOSIDAD-DO" in app
