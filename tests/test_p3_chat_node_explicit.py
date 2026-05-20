"""[P3-CHAT-NODE-EXPLICIT · 2026-05-20] Tests anti-regresión del `node`
explícito en la telemetría LLM del chat-flow.

Bug observado:
    `_emit_llm_usage_event_best_effort` (graph_orchestrator.py:2400) solo
    resolvía el campo `node` desde el ContextVar `_current_node_var`. El
    chat-flow (agent.py:call_model) NO setea ese var → todas las filas
    persistidas en `llm_usage_events` desde el chat quedaban con `node=NULL`.
    SRE no podía filtrar costos chat vs plan-gen, ni graficar P99 latency
    por subsistema.

Fix:
    - Helper acepta `node` como kwarg opcional; tiene prioridad sobre el
      ContextVar.
    - Callsite del chat (`agent.py:call_model`) pasa `node='chat_call_model'`
      explícito.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_PY = _BACKEND_ROOT / "graph_orchestrator.py"
_AGENT_PY = _BACKEND_ROOT / "agent.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_emit_helper_accepts_explicit_node_kwarg():
    """[P3-CHAT-NODE-EXPLICIT] `_emit_llm_usage_event_best_effort` debe
    aceptar `node: str = None` como kwarg. Sin esto, los callers no pueden
    overridear el ContextVar."""
    src = _read(_GRAPH_PY)
    match = re.search(
        r"def _emit_llm_usage_event_best_effort\(([^)]+)\)",
        src,
    )
    assert match, "_emit_llm_usage_event_best_effort no encontrada"
    signature = match.group(1)
    assert "node" in signature, (
        "Signature NO acepta `node` como kwarg. Sin esto, callers no pueden "
        "override del ContextVar. Ver P3-CHAT-NODE-EXPLICIT · 2026-05-20."
    )
    # Sanity: debe ser kwarg-only opcional con default None.
    assert re.search(r"node\s*:\s*str\s*=\s*None", signature), (
        "Signature debe ser `node: str = None` (kwarg opcional, default None) "
        "para preservar backward-compat con callers existentes."
    )


def test_emit_helper_prioritizes_explicit_node():
    """[P3-CHAT-NODE-EXPLICIT] Si el kwarg `node` se pasa, debe tener
    prioridad sobre el ContextVar `_current_node_var`. Sin esto, el chat
    seguiría con `node=NULL` aunque pase el kwarg."""
    src = _read(_GRAPH_PY)
    # Extraer body del helper.
    match = re.search(
        r"def _emit_llm_usage_event_best_effort\(.*?\):(.+?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert match
    body = match.group(1)
    # Anchor: `if node:` short-circuit antes del ContextVar lookup.
    assert re.search(r"if\s+node\s*:", body), (
        "Lógica `if node:` ausente — el kwarg no tiene prioridad sobre "
        "el ContextVar. Ver P3-CHAT-NODE-EXPLICIT · 2026-05-20."
    )
    # Sanity: el `current_node = node` se asigna explícitamente.
    assert re.search(r"current_node\s*=\s*node", body), (
        "Asignación `current_node = node` ausente — el helper no usa el "
        "kwarg para el INSERT."
    )


def _extract_balanced_call(src: str, fn_name: str) -> str:
    """Extrae el call `fn_name(...)` con parens balanceados (cubre el caso
    de args con parens anidados como `duration_s=_time.time() - x`)."""
    idx = src.find(f"{fn_name}(")
    if idx < 0:
        return ""
    start = src.find("(", idx)
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[idx:j + 1]
    return ""


def test_chat_callsite_passes_node_explicit():
    """[P3-CHAT-NODE-EXPLICIT] `agent.py:call_model` debe pasar `node='chat_call_model'`
    al helper. Sin esto, las filas del chat siguen con `node=NULL`."""
    src = _read(_AGENT_PY)
    call_str = _extract_balanced_call(src, "_emit_llm_usage_event_best_effort")
    assert call_str, "_emit_llm_usage_event_best_effort callsite no encontrado en agent.py"
    assert re.search(r"node\s*=\s*['\"]chat_call_model['\"]", call_str), (
        f"Callsite no pasa `node='chat_call_model'`. Sin esto, telemetría "
        f"queda con node=NULL. Ver P3-CHAT-NODE-EXPLICIT · 2026-05-20.\n"
        f"Callsite actual:\n{call_str}"
    )
