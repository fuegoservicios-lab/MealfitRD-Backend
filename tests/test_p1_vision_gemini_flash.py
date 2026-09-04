"""[P1-VISION-GEMINI-FLASH · 2026-09-04] Gemini 3.8 Flash como provider del escáner.

Roboflow Vision Evals (52 modelos, 2026-09): Gemini 3.8 Flash 85,1 % (#3), gpt-5.6-luna
72,7 % (#19), GLM-5V-Turbo 65,3 % (#36); identificación 97,9 % vs 84,4 %, conteo 78,8 % vs
66,2 %. Va por la capa compatible con OpenAI de Google con `ChatOpenAI` PLANO — cero
dependencia nueva, SOLO visión (el pipeline de planes no conoce el prefijo).

Invariantes: `VISION_API_KEY` > `GEMINI_API_KEY`/`GOOGLE_API_KEY`, fail-loud sin key; base
URL por defecto; la rama Luna (OpenAI) y la rama GLM quedan intactas; precio EXACTO por
modelo en el libro de costo (un gemini desconocido sigue costando None).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def test_is_google_model_only_for_gemini_ids_and_never_in_plan_router():
    from llm_provider import is_google_model, is_openai_model, GEMINI_OPENAI_BASE_URL
    assert is_google_model("gemini-3.8-flash") and is_google_model(" GEMINI-3.8-PRO ")
    assert not is_google_model("gpt-5.6-luna") and not is_google_model("glm-5.3-flash") and not is_google_model("")
    assert not is_openai_model("gemini-3.8-flash")
    assert GEMINI_OPENAI_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/openai/"
    # SOLO visión: la fábrica del pipeline no rutea gemini a ningún cliente
    src = (_BACKEND / "llm_provider.py").read_text(encoding="utf-8")
    body = src[src.index("def build_chat_llm("):]
    body = body[:body.find("\ndef ", 10)]
    assert "is_google_model" not in body and "gemini" not in body.lower()


def test_gemini_branch_builds_plain_chatopenai_with_default_base_and_key_precedence(monkeypatch):
    import vision_agent as va
    built = {}

    class _FakeOpenAI:
        def __init__(self, **kw):
            built.update(kw)

        def with_structured_output(self, schema):
            return ("structured", schema)

    class _NeverGLM:
        def __init__(self, **kw):
            raise AssertionError("un modelo gemini no debe construir ChatGLM")

    monkeypatch.setattr(va, "ChatOpenAI", _FakeOpenAI)
    monkeypatch.setattr(va, "ChatGLM", _NeverGLM)
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gemini-3.8-flash")
    monkeypatch.delenv("MEALFIT_VISION_BASE_URL", raising=False)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-not-real")

    out = va._resolve_vision_client(dict)
    assert out == ("structured", dict)
    assert built["model"] == "gemini-3.8-flash"
    assert built["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert built["api_key"] == "gemini-key-not-real"
    assert built["temperature"] == 0.1
    assert "extra_body" not in built  # sin `thinking` ni defaults de Z.ai

    # VISION_API_KEY gana; la base URL del knob gana sobre el default
    monkeypatch.setenv("VISION_API_KEY", "vision-key")
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", "https://proxy.example/v1/")
    va._resolve_vision_client(dict)
    assert built["api_key"] == "vision-key" and built["base_url"] == "https://proxy.example/v1/"

    # GOOGLE_API_KEY como alias; sin ninguna → fail-loud (nunca llamar al proveedor equivocado)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-alias")
    va._resolve_vision_client(dict)
    assert built["api_key"] == "google-alias"
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="key de Gemini"):
        va._resolve_vision_client(dict)


def test_luna_and_glm_branches_untouched(monkeypatch):
    import vision_agent as va
    built = {}

    class _Fake:
        def __init__(self, **kw):
            built.clear(); built.update(kw); built["cls"] = type(self).__name__

        def with_structured_output(self, schema):
            return self

    class ChatOpenAIF(_Fake):
        pass

    class ChatGLMF(_Fake):
        pass

    monkeypatch.setattr(va, "ChatOpenAI", ChatOpenAIF)
    monkeypatch.setattr(va, "ChatGLM", ChatGLMF)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", "https://api.openai.com/v1")
    va._resolve_vision_client(dict)
    assert built["cls"] == "ChatOpenAIF" and built["api_key"] == "openai-key"
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "glm-4.6v")
    va._resolve_vision_client(dict)
    assert built["cls"] == "ChatGLMF"


def test_vision_enabled_without_base_url_only_for_gemini(monkeypatch):
    import vision_agent as va
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "openai_compatible")
    monkeypatch.delenv("MEALFIT_VISION_BASE_URL", raising=False)
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gemini-3.8-flash")
    assert va.is_vision_enabled() is True
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gpt-5.6-luna")
    assert va.is_vision_enabled() is False  # Luna sigue exigiendo la base URL


def test_cost_book_has_exact_gemini_row_and_no_prefix_guessing():
    from db_profiles import compute_llm_cost_micros, _DEFAULT_LLM_PRICING_MICROS_PER_M
    assert _DEFAULT_LLM_PRICING_MICROS_PER_M["gemini-3.8-flash"] == {"input": 750_000, "output": 3_750_000, "cached": 75_000}
    assert compute_llm_cost_micros("gemini-3.8-flash", 1_000_000, 0) == 750_000
    assert compute_llm_cost_micros("gemini-3.5-flash", 1000, 500) is None
    assert "gemini" not in {k for k in _DEFAULT_LLM_PRICING_MICROS_PER_M if k == "gemini"}


def test_doc_script_and_marker():
    assert "P1-VISION-GEMINI-FLASH" in (_BACKEND / "docs" / "vision_luna.md").read_text(encoding="utf-8")
    assert (_BACKEND / "scripts" / "check_vision_model.py").exists()
    assert "P1-VISION-GEMINI-FLASH · 2026-09-04" in (_BACKEND / "app.py").read_text(encoding="utf-8")
