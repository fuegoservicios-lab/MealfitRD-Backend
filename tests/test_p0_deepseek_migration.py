"""[P0-DEEPSEEK-MIGRATION · 2026-06-12] Test ancla de la migración
Gemini → DeepSeek + router de modelos por tier de suscripción.

Contratos que ancla:
  A. Blanket: cero construcciones Gemini en código productivo (imports,
     constructores, GEMINI_API_KEY, model IDs `gemini-*` en string literals).
  B. Router por tier: gratis/guest/desconocido → deepseek-v4-flash;
     basic/plus/ultra → deepseek-v4-pro. Fail-cheap en errores de lookup.
  C. Wrapper ChatDeepSeek: drop-in del constructor legacy (swallow de
     google_api_key/safety_settings/thinking_budget, max_output_tokens →
     max_tokens, stream_usage habilitado, base_url DeepSeek, boot sin key
     no explota).
  D. Seguridad: la API key JAMÁS hardcodeada en el código (solo env).
  E. Knobs del provider registrados en _KNOBS_REGISTRY.
  F. Consistencia CB en agent.py: el gate del circuit breaker resuelve el
     modelo con el MISMO user_id que el constructor (tier-routing).
  G. Pricing de telemetría en DeepSeek (compute_llm_cost_micros).
  H. Degradación: embeddings/vision providers `disabled` responden con el
     contrato soft-fail (None / analysis_failed) sin tocar red.

Si quieres revertir o cambiar el mapping tier→modelo, los knobs
`MEALFIT_MODEL_FREE_TIER` / `MEALFIT_MODEL_PAID_TIER` lo permiten sin
redeploy — actualiza este test SOLO si la decisión de producto cambia.
"""
import io
import os
import re
import sys
import tokenize
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def _strip_comments(src: str) -> str:
    """Blanquea los tokens COMMENT preservando posiciones (línea/columna).
    Las menciones narrativas a Gemini en comentarios son historia legítima
    del repo; lo que se banea es CÓDIGO vivo (imports, constructores,
    literals de model IDs). Los strings/docstrings SÍ se escanean."""
    lines = src.splitlines(keepends=True)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                (srow, scol), (erow, ecol) = tok.start, tok.end
                line = lines[srow - 1]
                lines[srow - 1] = line[:scol] + " " * (ecol - scol) + line[ecol:]
    except Exception:
        # Tokenize falló (no debería en archivos que parsean) — escanear crudo.
        return src
    return "".join(lines)


def _prod_py_files():
    """Archivos productivos del backend: top-level + routers/ + prompts/.
    Excluye tests/, scripts/ (CLI one-shot, whitelisted), conftest y
    migrations."""
    files = []
    for pattern in ("*.py", "routers/*.py", "prompts/*.py"):
        for p in BACKEND.glob(pattern):
            if p.name.startswith("test_") or p.name == "conftest.py":
                continue
            files.append(p)
    assert len(files) > 20, "glob de archivos prod sospechosamente corto"
    return files


# Señales DURAS de uso del API Gemini en código (no matchean menciones
# narrativas en comentarios/docstrings — esas son historia legítima del repo).
_BANNED_CODE_PATTERNS = (
    "from langchain_google_genai",
    "import langchain_google_genai",
    "from google.genai import",
    "from google.genai.types import",
    "import google.generativeai",
    "ChatGoogleGenerativeAI(",
    "GoogleGenerativeAIEmbeddings(",
    "GEMINI_API_KEY",
    "google_api_key=",
    '"gemini-',
    "'gemini-",
)


def test_a_blanket_no_gemini_in_prod_code():
    """A. Ninguna construcción Gemini viva en código productivo."""
    violations = []
    for p in _prod_py_files():
        src = _strip_comments(p.read_text(encoding="utf-8"))
        for pat in _BANNED_CODE_PATTERNS:
            for m in re.finditer(re.escape(pat), src):
                line_no = src.count("\n", 0, m.start()) + 1
                line = src.splitlines()[line_no - 1].strip()
                violations.append(f"{p.name}:{line_no}: [{pat}] {line[:120]}")
    assert not violations, (
        "P0-DEEPSEEK-MIGRATION violado — referencias Gemini vivas en código "
        "prod (el provider es DeepSeek via llm_provider.py):\n"
        + "\n".join(violations)
    )


def test_a2_requirements_swapped():
    req = (BACKEND / "requirements.txt").read_text(encoding="utf-8")
    assert "langchain-google-genai" not in req, (
        "langchain-google-genai debe estar fuera de requirements.txt"
    )
    assert "langchain-openai" in req, (
        "langchain-openai (cliente del provider DeepSeek) debe estar pineado"
    )


# ------------------------------------------------------------------
# B. Router por tier
# ------------------------------------------------------------------

def test_b_resolve_model_for_tier_matrix():
    from llm_provider import (
        DEEPSEEK_FLASH,
        DEEPSEEK_PRO,
        resolve_model_for_tier,
    )

    # Decisión de producto 2026-06-12: free → flash, pagado → pro.
    assert resolve_model_for_tier("gratis") == DEEPSEEK_FLASH
    assert resolve_model_for_tier("basic") == DEEPSEEK_PRO
    assert resolve_model_for_tier("plus") == DEEPSEEK_PRO
    assert resolve_model_for_tier("ultra") == DEEPSEEK_PRO
    # Fail-cheap: None / vacío / desconocido / casing raro.
    assert resolve_model_for_tier(None) == DEEPSEEK_FLASH
    assert resolve_model_for_tier("") == DEEPSEEK_FLASH
    assert resolve_model_for_tier("enterprise") == DEEPSEEK_FLASH
    assert resolve_model_for_tier("  BASIC  ") == DEEPSEEK_PRO  # normaliza


def test_b2_resolve_model_for_user_paths(monkeypatch):
    import llm_provider

    llm_provider.invalidate_tier_cache()

    # Paid user → PRO.
    monkeypatch.setattr(
        "db.get_user_plan_tier", lambda uid: "plus", raising=False
    )
    assert (
        llm_provider.resolve_model_for_user("11111111-1111-1111-1111-111111111111")
        == llm_provider.DEEPSEEK_PRO
    )

    # Free user → FLASH.
    llm_provider.invalidate_tier_cache()
    monkeypatch.setattr(
        "db.get_user_plan_tier", lambda uid: "gratis", raising=False
    )
    assert (
        llm_provider.resolve_model_for_user("22222222-2222-2222-2222-222222222222")
        == llm_provider.DEEPSEEK_FLASH
    )

    # Lookup explota → fail-cheap a FLASH (jamás PRO por error).
    llm_provider.invalidate_tier_cache()

    def _boom(uid):
        raise RuntimeError("db caída")

    monkeypatch.setattr("db.get_user_plan_tier", _boom, raising=False)
    assert (
        llm_provider.resolve_model_for_user("33333333-3333-3333-3333-333333333333")
        == llm_provider.DEEPSEEK_FLASH
    )

    # Guests / None → FLASH sin tocar DB.
    assert llm_provider.resolve_model_for_user(None) == llm_provider.DEEPSEEK_FLASH
    assert llm_provider.resolve_model_for_user("guest") == llm_provider.DEEPSEEK_FLASH
    llm_provider.invalidate_tier_cache()


def test_b3_tier_cache_hit_avoids_second_lookup(monkeypatch):
    import llm_provider

    llm_provider.invalidate_tier_cache()
    calls = {"n": 0}

    def _counted(uid):
        calls["n"] += 1
        return "basic"

    monkeypatch.setattr("db.get_user_plan_tier", _counted, raising=False)
    uid = "44444444-4444-4444-4444-444444444444"
    assert llm_provider.get_user_tier(uid) == "basic"
    assert llm_provider.get_user_tier(uid) == "basic"
    assert calls["n"] == 1, "el 2do lookup debe servirse del cache TTL"
    llm_provider.invalidate_tier_cache(uid)
    assert llm_provider.get_user_tier(uid) == "basic"
    assert calls["n"] == 2, "invalidate_tier_cache(user) debe forzar re-lookup"
    llm_provider.invalidate_tier_cache()


def test_b4_knob_override_wins_over_tier(monkeypatch):
    """El override por knob (P3-PREVIEW-MODEL-KNOB) SIEMPRE gana al tier."""
    monkeypatch.setenv("MEALFIT_MODEL_PAID_TIER", "deepseek-x-custom")
    from llm_provider import resolve_model_for_tier

    assert resolve_model_for_tier("ultra") == "deepseek-x-custom"


def test_b5_route_model_is_tier_based_parser():
    """Parser-based: `_route_model` del orquestador rutea por tier (no por
    complejidad clínica) y preserva el early-return de force_fast."""
    src = (BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    m = re.search(
        r"def _route_model\(form_data.*?\n(?=\n*(?:def |class |# \[P2-PROD))",
        src,
        re.DOTALL,
    )
    assert m, "_route_model no encontrado en graph_orchestrator.py"
    body = m.group(0)
    assert "get_user_tier" in body, "_route_model debe resolver tier del user"
    assert "PAID_TIERS" in body, "_route_model debe rutear por PAID_TIERS"
    assert "user_id_var.get()" in body, (
        "_route_model lee la identidad del ContextVar user_id_var"
    )
    assert "if force_fast:" in body and "_FLASH_MODEL_NAME" in body


# ------------------------------------------------------------------
# C. Wrapper ChatDeepSeek
# ------------------------------------------------------------------

def test_c_wrapper_swallows_legacy_kwargs_and_maps_output_cap():
    from llm_provider import ChatDeepSeek, DEEPSEEK_FLASH

    llm = ChatDeepSeek(
        model=DEEPSEEK_FLASH,
        temperature=0.2,
        timeout=15,
        max_retries=1,
        max_output_tokens=64,
        google_api_key="legacy-debe-ignorarse",
        safety_settings={"legacy": True},
        thinking_budget=2048,
    )
    assert llm.model_name == DEEPSEEK_FLASH
    assert llm.max_tokens == 64, "max_output_tokens debe mapear a max_tokens"
    assert llm.stream_usage is True, (
        "stream_usage=True es requerido para que astream alimente "
        "llm_usage_events (P1-COST-INSTRUMENTATION-FIX)"
    )
    base = str(getattr(llm, "openai_api_base", "") or "")
    assert "api.deepseek.com" in base, f"base_url inesperada: {base!r}"


def test_c2_boot_without_api_key_does_not_raise(monkeypatch):
    """Construcción a module-import (agent.py::llm) no puede tirar el boot
    si falta la key — la invocación fallará con 401 explícito (paridad con
    el comportamiento histórico de google_api_key=None)."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from llm_provider import ChatDeepSeek, DEEPSEEK_FLASH

    llm = ChatDeepSeek(model=DEEPSEEK_FLASH)
    assert llm is not None


# ------------------------------------------------------------------
# D. La API key jamás hardcodeada
# ------------------------------------------------------------------

def test_d_no_hardcoded_api_keys_in_backend_source():
    pattern = re.compile(r"sk-[a-zA-Z0-9]{20,}")
    offenders = []
    for p in _prod_py_files():
        if pattern.search(p.read_text(encoding="utf-8")):
            offenders.append(p.name)
    # tests/ también: una key real en un test se filtra al repo igual.
    for p in (BACKEND / "tests").glob("*.py"):
        if p.name == Path(__file__).name:
            continue
        if pattern.search(p.read_text(encoding="utf-8")):
            offenders.append(f"tests/{p.name}")
    assert not offenders, (
        f"API key con formato `sk-...` hardcodeada en: {offenders}. "
        "La key vive SOLO en env (DEEPSEEK_API_KEY)."
    )


# ------------------------------------------------------------------
# E. Knobs registrados
# ------------------------------------------------------------------

def test_e_provider_knobs_registered():
    from knobs import get_knobs_registry_snapshot
    from llm_provider import (
        _deepseek_base_url,
        _tier_cache_ttl_s,
        model_free_tier,
        model_paid_tier,
    )

    # Forzar registro (los knobs se registran al leerse).
    _deepseek_base_url()
    model_free_tier()
    model_paid_tier()
    _tier_cache_ttl_s()
    snapshot = get_knobs_registry_snapshot()
    for knob in (
        "MEALFIT_DEEPSEEK_BASE_URL",
        "MEALFIT_MODEL_FREE_TIER",
        "MEALFIT_MODEL_PAID_TIER",
        "MEALFIT_TIER_CACHE_TTL_S",
    ):
        assert knob in snapshot, f"knob {knob} no registrado en _KNOBS_REGISTRY"


# ------------------------------------------------------------------
# F. Consistencia CB ↔ constructor en agent.py (tier-routing)
# ------------------------------------------------------------------

def test_f_agent_cb_gate_uses_same_uid_as_constructor():
    src = (BACKEND / "agent.py").read_text(encoding="utf-8")
    # call_model: constructor y gate con el mismo _model_uid.
    assert "model=_chat_agent_model_name(_model_uid)" in src, (
        "chat_llm debe construirse con _chat_agent_model_name(_model_uid)"
    )
    assert "_cb_model = _chat_agent_model_name(_model_uid)" in src, (
        "el gate CB de call_model debe resolver con el MISMO _model_uid"
    )
    # swap_meal: ídem con _swap_uid.
    assert "model=_chat_agent_swap_model_name(_swap_uid)" in src
    assert "_swap_cb_model = _chat_agent_swap_model_name(_swap_uid)" in src


def test_f2_chat_helpers_tier_routed_with_override(monkeypatch):
    """Los helpers de modelo del chat usan tier-routing y respetan override."""
    src = (BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "def _chat_agent_model_name(user_id: Optional[str] = None)" in src
    assert "def _chat_agent_swap_model_name(user_id: Optional[str] = None)" in src
    assert src.count("resolve_model_for_user(user_id)") >= 2


# ------------------------------------------------------------------
# G. Pricing de telemetría en DeepSeek
# ------------------------------------------------------------------

def test_g_pricing_table_deepseek():
    from db_profiles import compute_llm_cost_micros

    # flash: 1M in (sin cache) + 1M out = $0.14 + $0.28 = $0.42 = 420_000 micros
    assert compute_llm_cost_micros("deepseek-v4-flash", 1_000_000, 1_000_000) == 420_000
    # pro: 1M in + 1M out = $0.435 + $0.87 = $1.305 = 1_305_000 micros
    assert compute_llm_cost_micros("deepseek-v4-pro", 1_000_000, 1_000_000) == 1_305_000
    # cache hit descuenta input: flash 1M in TODO cacheado + 0 out
    #   = 1M × $0.0028/M = 2_800 micros
    assert compute_llm_cost_micros("deepseek-v4-flash", 1_000_000, 0, 1_000_000) == 2_800
    # Modelos desconocidos (incl. los gemini retirados) → None (fila sin costo).
    assert compute_llm_cost_micros("gemini-3.5-flash", 1000, 500) is None
    assert compute_llm_cost_micros("modelo-fantasma", 1000, 500) is None


def test_g2_pricing_override_knob_renamed(monkeypatch):
    """El knob de override es MEALFIT_LLM_PRICING_JSON (no el legacy GEMINI)."""
    monkeypatch.setenv(
        "MEALFIT_LLM_PRICING_JSON",
        '{"deepseek-v4-flash": {"input": 1000, "output": 2000, "cached": 100}}',
    )
    from db_profiles import compute_llm_cost_micros

    assert compute_llm_cost_micros("deepseek-v4-flash", 1_000_000, 1_000_000) == 3_000


# ------------------------------------------------------------------
# H. Providers pendientes degradan soft-fail
# ------------------------------------------------------------------

def test_h_embeddings_disabled_returns_none(monkeypatch):
    """[P1-COHERE-EMBED-V4] El default ahora es provider=cohere con GATING
    por presencia de key: sin COHERE_API_KEY el comportamiento observable es
    idéntico a disabled (degradación limpia, sin error-spam)."""
    monkeypatch.delenv("MEALFIT_EMBEDDINGS_PROVIDER", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDINGS_API_KEY", raising=False)
    import embeddings_provider

    assert embeddings_provider.is_embeddings_enabled() is False
    assert embeddings_provider.get_text_embedding("habichuelas") is None
    assert embeddings_provider.get_embeddings_model_id() == "disabled"
    assert embeddings_provider.get_embeddings_client() is None


def test_h2_vision_disabled_soft_fail(monkeypatch):
    monkeypatch.delenv("MEALFIT_VISION_PROVIDER", raising=False)
    import asyncio

    import vision_agent

    assert vision_agent.is_vision_enabled() is False
    result = asyncio.run(vision_agent.process_image_with_vision(b"\xff\xd8fake"))
    assert result["analysis_failed"] is True
    assert result["is_food"] is False
    # Shape completa que el frontend espera (P2-DIARY-SCAN-MACROS).
    for key in ("description", "meal_name", "calories", "protein", "carbs", "healthy_fats"):
        assert key in result
