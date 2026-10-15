"""[P1-CHAT-PROVIDER-FALLBACK · 2026-10-14] Tests para el respaldo del coach
cuando el proveedor principal falla o su circuit breaker está abierto.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from agent import _chat_agent_fallback_model, _build_chat_fallback_llm
from llm_provider import GLM_FLASH, GPT6_LUNA


def test_fallback_model_selection():
    """Con GLM/DeepSeek como principal y OPENAI_API_KEY presente, el respaldo es GPT-6 Luna."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "MEALFIT_CHAT_AGENT_FALLBACK_MODEL": GPT6_LUNA}):
        fb = _chat_agent_fallback_model(GLM_FLASH)
        assert fb == GPT6_LUNA


def test_fallback_disabled_via_knob():
    """Knob 'off' o 'none' apaga el respaldo."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "MEALFIT_CHAT_AGENT_FALLBACK_MODEL": "off"}):
        assert _chat_agent_fallback_model(GLM_FLASH) is None


def test_fallback_none_if_missing_key():
    """Sin OPENAI_API_KEY, no se ofrece respaldo OpenAI."""
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        assert _chat_agent_fallback_model(GLM_FLASH) is None


def test_fallback_none_if_same_as_primary():
    """Si el respaldo coincide con el principal, no hace bucle contra sí mismo."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "MEALFIT_CHAT_AGENT_FALLBACK_MODEL": GPT6_LUNA}):
        assert _chat_agent_fallback_model(GPT6_LUNA) is None
