"""[P1-COST-INSTRUMENTATION-FIX · 2026-05-15] Regression guard: la captura
de `usage_metadata` vive en los overrides de `ChatGoogleGenerativeAI`
(subclase wrapper), NO solo en `_safe_ainvoke`.

Bug observado en test E2E 2026-05-15:
  - Pipeline real ejecutó ~12 LLM calls (plan_skeleton + 3 day_gen + 3
    self_critique + 1 fact_checker + 1 judge + 1 reviewer + 1 plan_title).
  - `llm_usage_events` registró solo 2 calls (~$0.0006 vs realidad ~$0.20).

Root cause: `_safe_ainvoke` extrae `usage_metadata` del `result`, pero:
  - Cuando el LLM está wrappeado en `.with_structured_output(Model)` (planner,
    judge, reviewer), `result` es un Pydantic model — no tiene `usage_metadata`.
  - Cuando se usa `.astream(messages)` directamente (day_generator), el flow
    no pasa por `_safe_ainvoke` para nada.

Fix: añadir captura DENTRO de los overrides `ChatGoogleGenerativeAI.ainvoke`,
`.astream`, `.agenerate` — antes de que cualquier wrapper post-procese el
AIMessage. Esto cubre TODOS los paths.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_PATH = _BACKEND_ROOT / "graph_orchestrator.py"
_ENV_PATH = _BACKEND_ROOT / ".env"


def _read_graph() -> str:
    return _GRAPH_PATH.read_text(encoding="utf-8")


def _strip_comments(text: str) -> str:
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )


def _extract_method_body(class_body: str, method_name: str) -> str:
    """Extrae el cuerpo de un método (async def o def) dentro de una clase."""
    pattern = re.compile(
        rf"^    (async\s+)?def\s+{re.escape(method_name)}\b.*?(?=^    (async\s+)?def\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(class_body)
    return m.group(0) if m else ""


def test_ainvoke_override_captures_usage():
    """`ChatGoogleGenerativeAI.ainvoke` debe llamar
    `_emit_llm_usage_event_best_effort` ANTES del `return result`."""
    text = _read_graph()
    class_match = re.search(
        r"class ChatGoogleGenerativeAI\(_ChatGoogleGenerativeAI\):.*?(?=^class |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert class_match, "Subclase `ChatGoogleGenerativeAI` no encontrada."
    class_body = class_match.group(0)

    ainvoke_body = _extract_method_body(class_body, "ainvoke")
    assert ainvoke_body, "Override de `ainvoke` no encontrado."
    assert "_emit_llm_usage_event_best_effort" in ainvoke_body, (
        "P1-COST-INSTRUMENTATION-FIX: `ainvoke` override debe invocar "
        "`_emit_llm_usage_event_best_effort` para capturar usage_metadata del "
        "AIMessage ANTES de que `.with_structured_output(...)` lo wrappee."
    )
    # El emit debe estar ANTES del `return result`.
    emit_idx = ainvoke_body.find("_emit_llm_usage_event_best_effort")
    return_idx = ainvoke_body.find("return result")
    assert emit_idx > 0 and return_idx > emit_idx, (
        "El emit debe estar ANTES del `return result` para no perder casos."
    )


def test_astream_override_accumulates_for_usage():
    """`ChatGoogleGenerativeAI.astream` debe acumular chunks y emitir
    usage_metadata al final del stream. Sin esto, day_generator (único callsite
    de `.astream(...)` directo) no se contabiliza."""
    text = _read_graph()
    class_match = re.search(
        r"class ChatGoogleGenerativeAI\(_ChatGoogleGenerativeAI\):.*?(?=^class |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    class_body = class_match.group(0)
    astream_body = _extract_method_body(class_body, "astream")
    assert astream_body
    assert "_emit_llm_usage_event_best_effort" in astream_body, (
        "P1-COST-INSTRUMENTATION-FIX: `astream` override debe acumular chunks "
        "y emitir usage_metadata al final. Sin esto el day_generator no se "
        "captura en `llm_usage_events`."
    )
    # Debe acumular chunks (no solo yield).
    assert "_accumulated" in astream_body or "accumulated" in astream_body, (
        "El override debe ACUMULAR chunks (e.g. via `_accumulated = chunk` + "
        "`_accumulated = _accumulated + chunk`) para tener el message final "
        "con usage_metadata."
    )


def test_agenerate_override_extracts_from_llmresult():
    """`agenerate` retorna LLMResult; debe extraer el AIMessage de
    `result.generations[0][0].message` para emitir usage_metadata."""
    text = _read_graph()
    class_match = re.search(
        r"class ChatGoogleGenerativeAI\(_ChatGoogleGenerativeAI\):.*?(?=^class |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    class_body = class_match.group(0)
    agen_body = _extract_method_body(class_body, "agenerate")
    assert agen_body
    assert "_emit_llm_usage_event_best_effort" in agen_body, (
        "`agenerate` override debe invocar el helper."
    )
    assert "generations" in agen_body, (
        "Debe inspeccionar `result.generations` para encontrar el AIMessage."
    )


# ----- .env knobs aplicados -------------------------------------------------

def test_env_hedge_threshold_raised():
    """`MEALFIT_HEDGE_AFTER_BASE_S=60` en .env reduce hedging spam.
    Pre-fix default era 45s, días tardan 55-120s → hedge casi siempre dispara."""
    text = _ENV_PATH.read_text(encoding="utf-8")
    m = re.search(r"^MEALFIT_HEDGE_AFTER_BASE_S\s*=\s*(\d+)", text, re.MULTILINE)
    assert m, "Falta `MEALFIT_HEDGE_AFTER_BASE_S` en .env."
    val = int(m.group(1))
    assert val >= 55, (
        f"P1-HEDGE-THRESHOLD: hedge debe ser ≥55s (default código 45s). "
        f"Encontré {val}. Bajar dispara hedge en >70% de los días."
    )


def test_env_coherence_guard_warn_mode():
    """`MEALFIT_SHOPPING_COHERENCE_GUARD=warn` evita el banner UI falso
    cuando los caps recortan magnitudes intencionalmente."""
    text = _ENV_PATH.read_text(encoding="utf-8")
    m = re.search(
        r"^MEALFIT_SHOPPING_COHERENCE_GUARD\s*=\s*(\w+)",
        text,
        re.MULTILINE,
    )
    assert m, "Falta `MEALFIT_SHOPPING_COHERENCE_GUARD` en .env."
    val = m.group(1).strip().lower()
    assert val == "warn", (
        f"P1-COHERENCE-GUARD-WARN-MODE: knob debe ser 'warn' (no 'block') "
        f"hasta que el guard se haga cap-aware. Encontré: {val!r}."
    )
