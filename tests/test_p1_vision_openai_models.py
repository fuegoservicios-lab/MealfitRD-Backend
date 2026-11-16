"""[P1-VISION-GPT5-MINI · 2026-11-16] Despacho y costo de modelos OpenAI modernos en vision_agent.

Invariantes:
  1. `gpt-5-mini` y `gpt-5.4-mini` tienen filas exactas en `_DEFAULT_LLM_PRICING_MICROS_PER_M`.
  2. `gpt-5-mini` reenvía `MEALFIT_VISION_REASONING_EFFORT` (`minimal|low|medium|high`) al cliente OpenAI.
  3. Familia `gpt-6` (`gpt-6.1-terra`) omite `temperature` (HTTP 400 si != 1) y activa `use_responses_api=True`.
  4. Si `MEALFIT_VISION_BASE_URL` apunta a `generativelanguage.googleapis.com`, un modelo OpenAI la ignora.
"""
from __future__ import annotations


def test_pricing_book_includes_gpt5_mini_and_gpt54_mini():
    from db_profiles import compute_llm_cost_micros, _DEFAULT_LLM_PRICING_MICROS_PER_M

    assert _DEFAULT_LLM_PRICING_MICROS_PER_M["gpt-5-mini"] == {
        "input": 250_000,
        "output": 2_000_000,
        "cached": 25_000,
    }
    assert _DEFAULT_LLM_PRICING_MICROS_PER_M["gpt-5.4-mini"] == {
        "input": 750_000,
        "output": 4_500_000,
        "cached": 75_000,
    }
    assert compute_llm_cost_micros("gpt-5-mini", 1_000_000, 1_000_000, cached_tokens=500_000) == (
        125_000 + 12_500 + 2_000_000
    )


def test_vision_openai_forwards_effort_and_ignores_google_base_url(monkeypatch):
    import vision_agent as va

    built = {}

    class _FakeOpenAI:
        def __init__(self, **kw):
            built.clear()
            built.update(kw)

        def with_structured_output(self, schema, **kw):
            return ("structured", schema)

    monkeypatch.setattr(va, "ChatOpenAI", _FakeOpenAI)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gpt-5-mini")
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("MEALFIT_VISION_REASONING_EFFORT", "low")

    va._resolve_vision_client(dict)
    assert built["model"] == "gpt-5-mini"
    assert built["base_url"] is None
    assert built["api_key"] == "sk-test"
    assert built["reasoning_effort"] == "low"
    assert built["temperature"] == 0.1


def test_vision_gpt6_terra_omits_temperature_and_uses_responses_api(monkeypatch):
    import vision_agent as va

    built = {}

    class _FakeOpenAI:
        def __init__(self, **kw):
            built.clear()
            built.update(kw)

        def with_structured_output(self, schema, **kw):
            return ("structured", schema)

    monkeypatch.setattr(va, "ChatOpenAI", _FakeOpenAI)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gpt-6.1-terra")
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("MEALFIT_VISION_REASONING_EFFORT", "low")

    va._resolve_vision_client(dict)
    assert built["model"] == "gpt-6.1-terra"
    assert built["base_url"] == "https://api.openai.com/v1"
    assert "temperature" not in built
    assert built["use_responses_api"] is True
    assert built["reasoning_effort"] == "low"
