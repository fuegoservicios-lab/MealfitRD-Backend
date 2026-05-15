"""[P1-LLM-TIMEOUT-METRICS · 2026-05-15] Anchor parser-based: el wrapper
central `_safe_ainvoke` debe emitir un tick estructurado a `pipeline_metrics`
en cada `asyncio.TimeoutError` para que post-hoc se pueda graficar timeouts
por modelo / hora.

Contexto del bug original:
    Los timeouts del LLM (~30%+ de planes fallidos en incidentes de cuota
    Gemini) solo quedaban en logs. Los logs rotan; sin métrica persistente,
    SRE no puede graficar "cuántos planes timeoutearon hoy por modelo X".

Fix:
    - Nuevo helper `_emit_llm_timeout_metric(node, timeout_threshold_s,
      actual_wait_s, llm, extra_metadata)` (SSOT, best-effort try/except
      silencioso).
    - `_safe_ainvoke` mide `time.time()` antes del wait_for; en
      `TimeoutError` branch invoca el helper con `node='_safe_ainvoke_timeout'`
      ANTES de re-raise.

Tooltip-anchor: P1-LLM-TIMEOUT-METRICS-START
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH_ORCH = _BACKEND / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def orch_src() -> str:
    return _GRAPH_ORCH.read_text(encoding="utf-8")


def test_helper_defined(orch_src: str):
    assert re.search(
        r"def _emit_llm_timeout_metric\(",
        orch_src,
    ), (
        "P1-LLM-TIMEOUT-METRICS: helper `_emit_llm_timeout_metric` no definido."
    )


def test_helper_inserts_pipeline_metric(orch_src: str):
    """El helper debe ejecutar INSERT a pipeline_metrics."""
    anchor = re.search(r"def _emit_llm_timeout_metric\b[\s\S]+?(?=\ndef |\nclass )", orch_src)
    assert anchor is not None, "helper body not found"
    body = anchor.group(0)
    assert "INSERT INTO pipeline_metrics" in body, (
        "P1-LLM-TIMEOUT-METRICS: el helper no inserta a pipeline_metrics."
    )
    assert "timeout_threshold_s" in body and "actual_wait_s" in body, (
        "P1-LLM-TIMEOUT-METRICS: el helper no propaga los campos canónicos."
    )


def test_safe_ainvoke_invokes_helper(orch_src: str):
    """`_safe_ainvoke` debe llamar al helper en su TimeoutError branch."""
    safe_match = re.search(
        r"async def _safe_ainvoke\([\s\S]+?(?=\n(?:def |async def |class ))",
        orch_src,
    )
    assert safe_match is not None, "_safe_ainvoke body not found"
    body = safe_match.group(0)
    assert "asyncio.TimeoutError" in body, (
        "_safe_ainvoke ya no atrapa asyncio.TimeoutError — refactor breaking?"
    )
    assert "_emit_llm_timeout_metric" in body, (
        "P1-LLM-TIMEOUT-METRICS: _safe_ainvoke no llama al helper en el catch — "
        "los timeouts del LLM siguen sin tick estructurado a pipeline_metrics."
    )


def test_safe_ainvoke_measures_actual_wait(orch_src: str):
    """`_safe_ainvoke` debe medir `time.time()` para reportar actual_wait_s."""
    safe_match = re.search(
        r"async def _safe_ainvoke\([\s\S]+?(?=\n(?:def |async def |class ))",
        orch_src,
    )
    assert safe_match is not None
    body = safe_match.group(0)
    assert "_safe_ainvoke_started_at" in body, (
        "P1-LLM-TIMEOUT-METRICS: variable `_safe_ainvoke_started_at` ausente — "
        "no se puede calcular actual_wait_s sin marker de inicio."
    )


def test_marker_tooltip_present(orch_src: str):
    assert "P1-LLM-TIMEOUT-METRICS" in orch_src, (
        "Marker tooltip ausente — si renombras el bloque, actualiza este test."
    )
