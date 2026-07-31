"""[P1-NET-LUNA · 2026-07-31] La red post-fallo del pipeline es CROSS-PROVIDER:
`gpt-5.6-luna` (OpenAI) en vez de `deepseek-v4-pro`.

Razón (decisión owner): flash y pro son el MISMO proveedor — el incidente que
motivó la red actual (gym baseline: circuit breaker abierto 172×) fue DeepSeek
rate-limiteando bajo carga, y en ese modo de fallo pro cae JUNTO con flash: la
"red" no atrapaba nada y se caía al plan matemático. Luna es OpenAI (infra,
key y límites propios) = diversidad REAL. Simetría cross-provider: el pipeline
(DeepSeek) cae a OpenAI; el reviewer clínico (OpenAI, P1-REVIEWER-TIER-MODELS)
cae a DeepSeek.

Contratos que ancla:
  A. `_plan_pro_model_name()` default = `GPT56_LUNA` (constante SSOT de
     llm_provider); fail-safe sin OPENAI_API_KEY → `DEEPSEEK_PRO` (nunca sin
     red); rollback `MEALFIT_PRO_MODEL=deepseek-v4-pro`.
  B. Los 8 consumidores de modelo variable construyen con dispatch por
     proveedor — un modelo OpenAI construido con ChatDeepSeek iría al base_url
     de DeepSeek con la key equivocada (lección P1-DAYGEN-LUNA-CANARY, que
     cerró el day-gen y dejó vivas estas superficies hermanas).
  C. El thinking del corrector quirúrgico (extra_body DeepSeek-only) se salta
     cuando la red es OpenAI.
  D. Marker bumpeado.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_APP_SRC = (_BACKEND / "app.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------
# A. Default + fail-safe + rollback
# ------------------------------------------------------------------

def test_a_net_default_is_luna_with_key(monkeypatch):
    import graph_orchestrator as go

    monkeypatch.delenv("MEALFIT_PRO_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    assert go._plan_pro_model_name() == "gpt-5.6-luna"


def test_a2_failsafe_to_pro_without_key(monkeypatch):
    import graph_orchestrator as go

    monkeypatch.delenv("MEALFIT_PRO_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert go._plan_pro_model_name() == "deepseek-v4-pro", (
        "sin OPENAI_API_KEY la red debe volver a pro — nunca quedarse sin red"
    )


def test_a3_rollback_knob_wins(monkeypatch):
    import graph_orchestrator as go

    monkeypatch.setenv("MEALFIT_PRO_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    assert go._plan_pro_model_name() == "deepseek-v4-pro"


def test_a4_deepseek_knob_needs_no_openai_key(monkeypatch):
    import graph_orchestrator as go

    monkeypatch.setenv("MEALFIT_PRO_MODEL", "deepseek-v4-pro")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert go._plan_pro_model_name() == "deepseek-v4-pro"


# ------------------------------------------------------------------
# B. Dispatch por proveedor en TODOS los consumidores (parser)
# ------------------------------------------------------------------

def test_b_no_hardcoded_chatdeepseek_with_variable_model():
    """Ningún ChatDeepSeek( con model= variable que pueda ser OpenAI.
    Las construcciones legítimas restantes de ChatDeepSeek con modelo variable
    deben estar en ramas ya gateadas a DeepSeek (thinking) — se enumeran."""
    hardcoded = re.findall(
        r"ChatDeepSeek\(\s*\n?\s*model=(_PRO_MODEL_NAME|_evaluator_model|_corrector_model|planner_model)\b",
        _GO_SRC,
    )
    # Únicas permitidas: las ramas thinking (DeepSeek-only por gate explícito).
    # El corrector quirúrgico thinking usa _PRO_MODEL_NAME pero está gateado por
    # `and not _net_is_openai`.
    assert hardcoded.count("_PRO_MODEL_NAME") <= 1, (
        f"construcciones ChatDeepSeek hardcodeadas con modelo variable: {hardcoded} — "
        "deben usar el dispatch (ChatOpenAIInstrumented if is_openai_model(...) else ChatDeepSeek)"
    )
    assert "_evaluator_model" not in hardcoded
    assert "_corrector_model" not in hardcoded
    assert "planner_model" not in hardcoded


def test_b2_dispatch_sites_present():
    # Los consumidores de la red y de knobs de modelo despachan por proveedor.
    assert _GO_SRC.count("ChatOpenAIInstrumented if is_openai_model(") >= 5, (
        "esperaba dispatch por proveedor en planner primario, planner fallback, "
        "evaluator y correctores"
    )
    assert _GO_SRC.count("(ChatOpenAIInstrumented if _net_is_openai else ChatDeepSeek)(") >= 2, (
        "esperaba dispatch en el corrector quirúrgico (estándar + diagnóstico raw)"
    )


def test_c_surgical_thinking_gated_to_deepseek():
    m = re.search(
        r"if SURGICAL_PRO_THINKING_ENABLED and not _net_is_openai:",
        _GO_SRC,
    )
    assert m, (
        "el thinking del corrector quirúrgico (extra_body DeepSeek-only) debe "
        "saltarse cuando la red es OpenAI"
    )


# ------------------------------------------------------------------
# D. Marker
# ------------------------------------------------------------------

def test_d_marker_and_anchor():
    assert "P1-NET-LUNA" in _GO_SRC
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _APP_SRC)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-NET-LUNA" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-31"
