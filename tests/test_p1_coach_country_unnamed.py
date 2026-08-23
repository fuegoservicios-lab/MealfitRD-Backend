"""[P1-COACH-COUNTRY-UNNAMED · 2026-08-23]

El país debe estar nombrado en las cuatro ramas reales del coach; DO y el
rollback del knob conservan el prompt histórico byte a byte.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


_BACKEND = Path(__file__).resolve().parents[1]


class _PromptCaptured(RuntimeError):
    pass


def _render_coach_prompt(monkeypatch, path: str, country: str, static_prefix: bool, enabled: bool = True) -> str:
    import agent
    import prompts.plan_generator as plan_generator
    from prompts.sentiment import PERSONALITY_PROFILES

    captured = {}

    class _Graph:
        def get_state(self, _config):
            return SimpleNamespace(values={})

        def invoke(self, inputs, **_kwargs):
            captured["prompt"] = inputs["sys_prompt"]
            raise _PromptCaptured

        def stream(self, inputs, **_kwargs):
            captured["prompt"] = inputs["sys_prompt"]
            raise _PromptCaptured

    class _Builder:
        def compile(self, **_kwargs):
            return _Graph()

    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true" if enabled else "false")
    monkeypatch.setattr(agent, "_plan_vigente_para_prompt", lambda *_args: None)
    monkeypatch.setattr(
        agent,
        "build_memory_context",
        lambda *_args: {"recent_messages": [], "summary_context": ""},
    )
    monkeypatch.setattr(
        agent,
        "classify_sentiment",
        lambda _prompt: {**PERSONALITY_PROFILES["neutral"], "sentiment": "neutral"},
    )
    monkeypatch.setattr(agent, "_chat_prompt_static_prefix", lambda: static_prefix)
    monkeypatch.setattr(agent, "CHAT_AGENT_INLINE_PROMPT", "PROMPT_COACH_BASE")
    monkeypatch.setattr(agent, "CHAT_STREAM_INLINE_PROMPT", "PROMPT_COACH_BASE")
    for name in (
        "build_tools_instructions",
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
    monkeypatch.setattr(agent, "_emit_chat_stream_total_duration_best_effort", lambda *_args: None)
    monkeypatch.setattr(agent, "chat_builder", _Builder())
    monkeypatch.setattr(agent, "chat_checkpoint_pool", None)
    monkeypatch.setattr(agent, "connection_pool", None)

    kwargs = dict(
        session_id="session-country",
        prompt="¿Qué puedo cenar?",
        user_id=None,
        form_data={"country": country},
    )
    with pytest.raises(_PromptCaptured):
        if path == "stream":
            list(agent.chat_with_agent_stream(**kwargs))
        else:
            agent.chat_with_agent(**kwargs)
    return captured["prompt"]


@pytest.mark.parametrize("path", ("nonstream", "stream"))
@pytest.mark.parametrize("static_prefix", (True, False), ids=("static", "legacy"))
def test_las_cuatro_ramas_nombradas_distinguen_es_de_mx(monkeypatch, path, static_prefix):
    es = _render_coach_prompt(monkeypatch, path, "ES", static_prefix)
    mx = _render_coach_prompt(monkeypatch, path, "MX", static_prefix)
    assert "España" in es
    assert "México" in mx
    assert es != mx
    forbidden = ("ejemplos de este prompt", "su gente reconozca por su nombre")
    assert not any(text in es.lower() or text in mx.lower() for text in forbidden)


@pytest.mark.parametrize("path", ("nonstream", "stream"))
@pytest.mark.parametrize("static_prefix", (True, False), ids=("static", "legacy"))
def test_do_es_byte_identico_al_rollback_del_knob(monkeypatch, path, static_prefix):
    do = _render_coach_prompt(monkeypatch, path, "DO", static_prefix, enabled=True)
    rollback = _render_coach_prompt(monkeypatch, path, "ES", static_prefix, enabled=False)
    assert do == rollback


def test_wiring_cubre_exactamente_los_cuatro_espejos_del_coach():
    source = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert code.count("coach_country_context(_coach_country)") == 4


def test_marker_movil_del_gap():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P1-COACH-COUNTRY-UNNAMED" in app
