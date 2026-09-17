# -*- coding: utf-8 -*-
"""[P0-DEEPSEEK-FLASH · P1-PLAN-LOTE-74 · 2026-09-16] DeepSeek (V4.1 Flash / V4 Pro) como proveedor LLM alterno
por knob.

Z.ai se quedó sin saldo (429/1113) y cayó toda la IA; el dueño pidió probar DeepSeek Flash. Desde
`P1-SINGLE-PROVIDER-RESTORE` un proveedor alterno debe nacer con knob + test ancla propios: este es el ancla.
Contrato: con el knob en su default (`zai`) NADA cambia; con `deepseek`, el wrapper apunta a DeepSeek con su key,
traduce los IDs GLM de los defaults por feature, mantiene el razonamiento con el mismo vocabulario y respeta
`thinking.type=disabled` (que DeepSeek sí soporta). Una instancia apuntada a OpenAI no se toca.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


@pytest.fixture
def lp(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-zai-key-not-real")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key-not-real")
    monkeypatch.delenv("MEALFIT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MEALFIT_DEEPSEEK_BASE_URL", raising=False)
    import llm_provider as _lp
    return _lp


@pytest.fixture
def deepseek(lp, monkeypatch):
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", "deepseek")
    return lp


# ── Con el knob por defecto, nada cambia ──────────────────────────────────────────────────────────

def test_default_provider_is_zai_and_untouched(lp):
    assert lp.llm_provider_name() == "zai"
    llm = lp.ChatGLM(model=lp.GLM_FLASH)
    assert "api.z.ai/api/paas/v4" in (llm.openai_api_base or "")
    assert llm.model_name == lp.GLM_FLASH
    assert lp._is_glm_provider() is True and lp._is_deepseek_provider() is False


def test_unknown_provider_value_falls_back_to_zai(lp, monkeypatch):
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", "ollama")
    assert lp.llm_provider_name() == "zai"


# ── Con el knob en deepseek ───────────────────────────────────────────────────────────────────────

def test_deepseek_points_to_its_api_and_translates_glm_ids(deepseek):
    lp = deepseek
    assert lp._is_deepseek_provider() is True and lp._is_glm_provider() is False
    flash = lp.ChatGLM(model=lp.GLM_FLASH)
    assert "api.deepseek.com" in (flash.openai_api_base or "")
    assert flash.model_name == "deepseek-flash"
    assert lp.ChatGLM(model=lp.GLM_PRO).model_name == "deepseek-v4-pro"
    assert lp.ChatGLM(model="deepseek-flash").model_name == "deepseek-flash"   # un ID propio pasa tal cual


def test_deepseek_reasons_with_the_same_vocabulary(deepseek):
    lp = deepseek
    llm = lp.ChatGLM(model=lp.GLM_FLASH)
    assert (llm.extra_body or {}).get("thinking") == {"type": "enabled"}
    assert llm.reasoning_effort == "low"                                          # default del knob
    assert lp.ChatGLM(model=lp.GLM_FLASH, reasoning_effort="medium").reasoning_effort == "high"
    assert lp.ChatGLM(model=lp.GLM_FLASH, extra_body={"thinking": {"effort": "max"}}).reasoning_effort == "max"


def test_deepseek_honours_thinking_disabled(deepseek):
    """GLM no puede apagar el razonamiento (se traduce a `low`); DeepSeek sí, y se respeta."""
    lp = deepseek
    llm = lp.ChatGLM(model=lp.GLM_FLASH, extra_body={"thinking": {"type": "disabled"}}, reasoning_effort="high")
    assert llm.extra_body["thinking"] == {"type": "disabled"}
    assert not getattr(llm, "reasoning_effort", None)


def test_openai_pointed_instance_is_left_alone_under_deepseek(deepseek):
    lp = deepseek
    llm = lp.ChatGLM(model="gpt-5.6-luna", api_key="sk-fake", base_url="https://api.openai.com/v1")
    assert llm.model_name == "gpt-5.6-luna"
    assert not (llm.extra_body or {}).get("thinking")
    assert lp.is_openai_model("deepseek-flash") is False


def test_deepseek_key_is_env_only_with_placeholder(deepseek, monkeypatch):
    lp = deepseek
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    lp._warned_missing_deepseek_key = False
    assert lp._deepseek_api_key() == "MISSING_DEEPSEEK_API_KEY"
    import re
    src = _src("llm_provider.py")
    assert 'os.environ.get("DEEPSEEK_API_KEY")' in src
    assert not re.search(r"sk-[0-9a-f]{24,}", src), "una clave real jamás en el módulo"


def test_structured_output_keeps_function_calling_and_honours_json_mode_on_deepseek(deepseek, monkeypatch):
    lp = deepseek
    seen = {}

    def fake(self, schema=None, **kwargs):
        seen.update(kwargs)
        return "ok"

    instancias = []

    def fake2(self, schema=None, **kwargs):
        instancias.append(self)
        return fake(self, schema, **kwargs)

    monkeypatch.setattr(lp.ChatOpenAI, "with_structured_output", fake2)
    original = lp.ChatGLM(model=lp.GLM_FLASH)
    original.with_structured_output(dict)
    assert seen["method"] == "function_calling"
    # medido en vivo: thinking rechaza tool_choice forzado → la copia va sin razonar; la original no cambia
    assert instancias[-1].extra_body["thinking"] == {"type": "disabled"} and not instancias[-1].reasoning_effort
    assert original.extra_body["thinking"] == {"type": "enabled"} and original.reasoning_effort == "low"
    original.with_structured_output(dict, method="json_mode")
    assert seen["method"] == "json_mode" and instancias[-1] is original


# ── Precio, marcador, doc, ejemplo de env ─────────────────────────────────────────────────────────

def test_prices_registered_off_peak():
    from db_profiles import _DEFAULT_LLM_PRICING_MICROS_PER_M as T, compute_llm_cost_micros
    assert T["deepseek-flash"] == {"input": 150_000, "output": 600_000, "cached": 3_000}
    assert T["deepseek-v4-pro"] == {"input": 660_000, "output": 1_980_000, "cached": 22_000}
    assert compute_llm_cost_micros("deepseek-flash", 1_000_000, 1_000_000) == 750_000


def test_marker_doc_and_env_example():
    app = _src("app.py")
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 74 and m.group(2) >= "2026-09-16"   # la serie sigue; el marker nunca baja
    assert "[P1-PLAN-LOTE-74 · 2026-09-16]" in app and "P0-DEEPSEEK-FLASH · 2026-09-16" in app
    doc = _src("docs/llm_tier_routing.md")
    assert "P0-DEEPSEEK-FLASH" in doc and "MEALFIT_LLM_PROVIDER" in doc and "deepseek-v4-pro" in doc
    env = _src(".env.example")
    assert "DEEPSEEK_API_KEY" in env and "MEALFIT_LLM_PROVIDER" in env


# ── El proveedor vuelve solo por knob y solo en sus superficies ──────────────────────────────────

SUPERFICIES = {"llm_provider.py", "db_profiles.py", "app.py", ".env.example", "docs/llm_tier_routing.md",
               "docs/coach_bateria_2026_09_15.md"}   # [P1-PLAN-LOTE-77] la batería del coach medida con el proveedor alterno


def _menciones(token: str) -> set:
    skip = {"venv", "venv-test", ".venv", "test_venv", ".git", "migrations", "__pycache__", ".pytest_cache",
            "node_modules", "tests"}
    exts = {".py", ".md", ".yml", ".yaml", ".txt", ".example", ".toml", ".ini", ".cfg", ".json"}
    out = set()
    for p in _BACKEND.rglob("*"):
        if any(part in skip for part in p.parts) or not p.is_file():
            continue
        if p.suffix not in exts and p.name != ".env.example":
            continue
        if token in p.read_text(encoding="utf-8", errors="ignore").lower():
            out.add(p.relative_to(_BACKEND).as_posix())
    return out


def test_the_provider_lives_only_in_its_surfaces_and_never_by_its_june_names():
    """La decisión del 02-sep («cero menciones») la revirtió el dueño el 16-sep, pero acotada: DeepSeek existe
    solo tras `MEALFIT_LLM_PROVIDER` y solo en estas superficies; un fichero nuevo que lo mencione se añade
    aquí a propósito. Los nombres de la migración de junio no vuelven."""
    token = "deep" + "seek"
    assert _menciones(token) == SUPERFICIES, _menciones(token) ^ SUPERFICIES
    for legado in (token + "-chat", token + "-reasoner", "chat" + token, "langchain_" + token):
        assert not _menciones(legado), legado
    assert '_env_str("MEALFIT_LLM_PROVIDER", "zai"' in _src("llm_provider.py")
