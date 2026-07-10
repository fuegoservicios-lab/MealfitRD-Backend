"""[P3-CONTEXT-COMPRESSION-SKIP · 2026-07-10] `context_compression_node` gatea la llamada LLM
(~6s medidos en corr=d57ffe04, 8301→1136 chars) con un umbral HARDCODED (`len(history_context) <
2000`). Sin knob, tunear ese umbral (ej. subirlo tras medir que compresiones de 2-4k chars no valen
la pena vs el ahorro de latencia) requiere redeploy — viola la convención del repo ("cambios de
comportamiento que pueden necesitar revertirse sin redeploy van como knob"). Fix: knob
`MEALFIT_CONTEXT_COMPRESSION_MIN_CHARS` (default 2000 — CERO cambio de comportamiento hoy), clamp
razonable. Las 2 capas de cache (in-process + KV persistente) ya existían y quedan intactas.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO_SRC = f.read()


def test_marker_present():
    assert "P3-CONTEXT-COMPRESSION-SKIP" in _GO_SRC


def test_knob_exists_with_default_2000_preserving_behavior():
    assert 'CONTEXT_COMPRESSION_MIN_CHARS = _env_int("MEALFIT_CONTEXT_COMPRESSION_MIN_CHARS", 2000' in _GO_SRC


def test_node_uses_the_knob_not_hardcoded_literal():
    i = _GO_SRC.index("async def context_compression_node")
    j = _GO_SRC.index("\n@_node_label", i + 1) if "\n@_node_label" in _GO_SRC[i + 1:] else i + 3000
    window = _GO_SRC[i:min(j, i + 3000)]
    assert "len(history_context) < 2000" not in window, \
        "el nodo ya no debe tener el umbral hardcodeado — debe leer el knob"
    assert "len(history_context) < CONTEXT_COMPRESSION_MIN_CHARS" in window


def test_knob_import(monkeypatch):
    import graph_orchestrator as go
    assert go.CONTEXT_COMPRESSION_MIN_CHARS == 2000


def test_knob_respects_env_override(monkeypatch):
    monkeypatch.setenv("MEALFIT_CONTEXT_COMPRESSION_MIN_CHARS", "4000")
    import importlib
    import graph_orchestrator as go
    importlib.reload(go)
    try:
        assert go.CONTEXT_COMPRESSION_MIN_CHARS == 4000
    finally:
        monkeypatch.delenv("MEALFIT_CONTEXT_COMPRESSION_MIN_CHARS", raising=False)
        importlib.reload(go)  # restaurar default para no filtrar estado a otros tests
