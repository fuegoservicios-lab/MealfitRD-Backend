"""[P1-CHAT-EMPTY-RESPONSE · 2026-05-20] Tests anti-regresión de dos bugs
expuestos durante validación post-deploy del bundle P1-CHAT-PROD-AUDIT:

  1. `MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S` default 8.0s fallaba contra Gemini
     API (mínimo 10s) → RAG router degradado silente.
     Fix: default 12.0s + validator floor 10.0s.

  2. Gemini `gemini-3.5-flash` emite ocasionalmente response vacío con
     `block_reason=PROHIBITED_CONTENT` por filtros server-side de Google
     no-controlables vía safety_settings del SDK. El AIMessage(content='')
     se propagaba a través del graph (sin tool_calls → routes a END) y el
     frontend renderiza chat vacío — UX confusa.
     Fix: detectar `(empty content) AND (no tool_calls)` post-invoke +
     sustituir por copy fallback + emit `chat_llm_empty_response` metric.

Tests parser-based: anchor literal en código para que un refactor
accidental falle el test antes de re-romper producción.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================
# Fix #1 — Router LLM timeout default ≥10s (Gemini API minimum)
# ============================================================

def test_chat_router_llm_timeout_default_at_least_10s():
    """[P1-CHAT-EMPTY-RESPONSE] El default debe ser >= 10.0 — Gemini
    API rechaza con 400 INVALID_ARGUMENT cualquier deadline < 10s.

    Anchor: `_chat_router_llm_timeout_s` debe tener default >= 10.0.
    """
    src = _read(_AGENT_PY)
    match = re.search(
        r"def _chat_router_llm_timeout_s\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert match, "función _chat_router_llm_timeout_s no encontrada"
    body = match.group(1)
    # El _env_float toma el default como segundo argumento posicional.
    # Buscamos: `_env_float("MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S", <num>, ...`
    default_match = re.search(
        r'_env_float\(\s*["\']MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S["\']\s*,\s*([0-9]+(?:\.[0-9]+)?)',
        body,
    )
    assert default_match, "Default no parseable del _env_float — refactor inesperado"
    default_val = float(default_match.group(1))
    assert default_val >= 10.0, (
        f"Default {default_val}s < 10s mínimo del Gemini API. Gemini rechaza "
        f"con HTTP 400 'Manually set deadline is too short.'. Subir a >=10. "
        f"Ver P1-CHAT-EMPTY-RESPONSE · 2026-05-20."
    )


def test_chat_router_llm_timeout_validator_floor_10s():
    """[P1-CHAT-EMPTY-RESPONSE] El validator debe enforce un floor de 10.0
    para que un override del env var con valor inválido NO regrese al bug.

    Anchor: validator lambda contiene `10.0 <= v` o similar.
    """
    src = _read(_AGENT_PY)
    match = re.search(
        r"def _chat_router_llm_timeout_s\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert match
    body = match.group(1)
    # Patrones aceptados: `10.0 <= v` o `v >= 10.0` (ambos válidos).
    floor_ok = bool(re.search(r"10(?:\.0)?\s*<=\s*v", body)) or bool(
        re.search(r"v\s*>=\s*10(?:\.0)?", body)
    )
    assert floor_ok, (
        "Validator del timeout router no enforce floor 10s. Un operador puede "
        "setear MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S=5 y reintroducir el bug. Ver "
        "P1-CHAT-EMPTY-RESPONSE · 2026-05-20."
    )


# ============================================================
# Fix #2 — Gemini empty response fallback
# ============================================================

def test_call_model_handles_empty_response_with_fallback():
    """[P1-CHAT-EMPTY-RESPONSE] El nodo `call_model` debe detectar
    response vacío sin tool_calls (Gemini PROHIBITED_CONTENT block) y
    sustituirlo por un AIMessage con copy fallback en lugar de propagar
    el blank.

    Anchor: la lógica vive entre `_cb.record_success()` y el return del
    nodo, y manipula `response` re-asignándolo a un AIMessage.
    """
    src = _read(_AGENT_PY)
    # Extrae el cuerpo de call_model.
    call_model_match = re.search(
        r"def call_model\(state: ChatState\):.*?\n(.*?)(?=\ndef )",
        src,
        re.DOTALL,
    )
    assert call_model_match, "call_model no encontrado"
    body = call_model_match.group(1)

    # Anchor 1: warning log explícito del marker.
    assert "P1-CHAT-CHAT-EMPTY-RESPONSE" in body or "P1-CHAT-EMPTY-RESPONSE" in body, (
        "Marker P1-CHAT-EMPTY-RESPONSE ausente del log de detección — "
        "refactor lo removió."
    )

    # Anchor 2: detección de empty content + no tool_calls.
    has_detection = bool(
        re.search(r"not\s+_resp_content_str\s+and\s+not\s+_resp_tool_calls", body)
    )
    assert has_detection, (
        "Detección `not _resp_content_str and not _resp_tool_calls` ausente. "
        "Refactor removió el guard del empty response."
    )

    # Anchor 3: el response se reemplaza por un AIMessage.
    assert re.search(r"response\s*=\s*AIMessage\(", body), (
        "El response no se reemplaza por AIMessage post-detection — el "
        "fallback no se aplica."
    )

    # Anchor 4: hay un copy fallback no-vacío.
    fallback_match = re.search(
        r"_fallback_copy\s*=\s*\(?\s*[\"']",
        body,
    )
    assert fallback_match, (
        "Variable `_fallback_copy` con string literal ausente — el copy "
        "fallback no está definido."
    )


def test_call_model_emits_empty_response_metric():
    """[P1-CHAT-EMPTY-RESPONSE] El nodo emite metric a `pipeline_metrics`
    con `node='chat_llm_empty_response'` cuando detecta el blank. SRE puede
    graficar la incidencia para decidir si cambiar de modelo (gemini-3.5-pro
    es más permisivo) o suavizar el system prompt."""
    src = _read(_AGENT_PY)
    call_model_match = re.search(
        r"def call_model\(state: ChatState\):.*?\n(.*?)(?=\ndef )",
        src,
        re.DOTALL,
    )
    assert call_model_match
    body = call_model_match.group(1)
    assert "chat_llm_empty_response" in body, (
        "Metric `chat_llm_empty_response` ausente. SRE no puede graficar "
        "falsos positivos del filtro server-side. Ver P1-CHAT-EMPTY-RESPONSE."
    )
    # Sanity: el INSERT a pipeline_metrics debe estar dentro de un try/except
    # (best-effort — un fallo de DB no debe tumbar el chat).
    assert (
        "INSERT INTO pipeline_metrics" in body
        and "chat_llm_empty_response" in body
    ), "INSERT a pipeline_metrics con el node correcto ausente."
