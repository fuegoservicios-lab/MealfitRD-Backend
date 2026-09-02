"""[P0-GLM-MIGRATION · 2026-09-02] Ancla del proveedor Z.ai GLM-5.3 (cross-link del marker
`_LAST_KNOWN_PFIX`, test_p2_hist_audit_14_marker_test_link).

Contratos (medidos EN VIVO 2026-09-02 con clave de pago, ver docs/llm_tier_routing.md):
  A. `ChatGLM` apunta a Z.ai, razona SIEMPRE (`thinking.enabled`) y fija `reasoning_effort`
     con default `low` (knob `MEALFIT_GLM_REASONING_EFFORT`); traduce el vocabulario heredado.
  B. `with_structured_output` fuerza `function_calling` y reencamina `json_mode` (Z.ai ignora
     el esquema en json_mode/json_schema).
  C. Precio de LISTA en la tabla de costos (no la promo), longest-prefix flash > glm-5.3.
  D. El detector de proveedor mira el HOST, no el nombre del modelo.
  E. Cero menciones del proveedor anterior (blanket completo en
     test_p0_llm_provider_migration.py::test_h_previous_provider_name_absent_from_repo).
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import llm_provider as lp

_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("MEALFIT_GLM_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("MEALFIT_ZAI_BASE_URL", raising=False)


def test_a_wrapper_points_to_zai_and_reasons_low_by_default():
    llm = lp.ChatGLM(model=lp.GLM_FLASH)
    assert lp.GLM_FLASH == "glm-5.3-flash" and lp.GLM_PRO == "glm-5.3"
    assert "api.z.ai/api/paas/v4" in (llm.openai_api_base or "")
    assert (llm.extra_body or {}).get("thinking") == {"type": "enabled"}
    assert llm.reasoning_effort == "low"


@pytest.mark.parametrize("legacy,expected", [
    ({"type": "disabled"}, "low"),
    ({"type": "enabled", "effort": "medium"}, "high"),
    ({"type": "enabled", "effort": "xhigh"}, "max"),
    ({"type": "enabled", "effort": "max"}, "max"),
])
def test_a_legacy_thinking_vocabulary_is_translated(legacy, expected):
    llm = lp.ChatGLM(model=lp.GLM_PRO, extra_body={"thinking": legacy})
    assert llm.reasoning_effort == expected
    assert llm.extra_body["thinking"] == {"type": "enabled"}, "GLM no puede apagar el razonamiento"


def test_a_explicit_reasoning_effort_wins_and_is_normalized():
    assert lp.ChatGLM(model=lp.GLM_FLASH, reasoning_effort="medium").reasoning_effort == "high"
    assert lp.ChatGLM(model=lp.GLM_FLASH, reasoning_effort="max").reasoning_effort == "max"


def test_a_openai_pointed_instance_gets_no_thinking():
    llm = lp.ChatGLM(model="gpt-5.6-luna", api_key="sk-fake", base_url="https://api.openai.com/v1")
    assert not (llm.extra_body or {}).get("thinking")


def test_b_structured_output_forces_function_calling_and_reroutes_json_mode():
    from pydantic import BaseModel

    class S(BaseModel):
        x: int

    r_default = lp.ChatGLM(model=lp.GLM_FLASH).with_structured_output(S)
    r_json = lp.ChatGLM(model=lp.GLM_FLASH).with_structured_output(S, method="json_mode")
    for r in (r_default, r_json):
        kw = getattr(r.first, "kwargs", {}) or {}
        assert "tools" in kw, "function_calling = tools bound; json_mode ignora el esquema en Z.ai"


def test_c_list_prices_and_prefix_precedence():
    from db_profiles import _DEFAULT_LLM_PRICING_MICROS_PER_M as T, compute_llm_cost_micros
    assert T["glm-5.3-flash"] == {"input": 150_000, "output": 500_000, "cached": 30_000}
    assert T["glm-5.3"] == {"input": 1_400_000, "output": 4_400_000, "cached": 260_000}
    assert compute_llm_cost_micros("glm-5.3-flash", 1_000_000, 1_000_000) == 650_000
    assert compute_llm_cost_micros("glm-5.3", 1_000_000, 1_000_000) == 5_800_000


def test_d_provider_detector_is_host_based():
    assert lp._is_glm_provider("https://api.z.ai/api/paas/v4") is True
    assert lp._is_glm_provider("https://open.bigmodel.cn/api/paas/v4") is True
    assert lp._is_glm_provider("https://api.openai.com/v1") is False
    assert lp._is_glm_provider() is True  # knob default


def test_e_marker_bumped_and_no_old_name_in_provider_module():
    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert '_LAST_KNOWN_PFIX = "P0-GLM-MIGRATION · 2026-09-02"' in app_src
    lp_src = (_BACKEND / "llm_provider.py").read_text(encoding="utf-8").lower()
    assert ("deep" + "seek") not in lp_src
