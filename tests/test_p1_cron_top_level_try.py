"""[P1-CRON-TOP-LEVEL-TRY · 2026-05-15] Anchor parser-based: el cron
`_aggregate_coherence_block_history_metrics` (P3-B, horario) debe estar
envuelto en try/except + finally tick observable.

Contexto del bug original:
    El cron tenía múltiples `return` tempranos (db_core import fail,
    supabase=None, fetch falla) que salían silenciosamente. El watchdog
    `_alert_pipeline_metrics_silence` observa el heartbeat
    `_chunk_heartbeat_baseline` (que sigue OK), pero NO observa este cron
    específico — un fallo de 3h consecutivas era invisible para SRE.

Fix:
    Top-level try / except / finally siguiendo el patrón P2-LIVE-9 ya
    aplicado a `_shopping_coherence_alert_job`:
    - `try:` wrapping body
    - `except Exception` catches non-internal failures + sets skip_reason
    - `finally:` siempre emite tick a pipeline_metrics con `skip_reason`,
      counts agregados, surface_breakdown.

Tooltip-anchor: P1-CRON-TOP-LEVEL-TRY-START
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _extract_aggregate_fn_body(src: str) -> str:
    anchor = re.search(
        r"def _aggregate_coherence_block_history_metrics\b[^\n]*\n",
        src,
    )
    assert anchor is not None, (
        "P1-CRON-TOP-LEVEL-TRY: `_aggregate_coherence_block_history_metrics` "
        "ya no existe — ¿renombrado? actualizar test."
    )
    start = anchor.end()
    rest = src[start:]
    # Buscar siguiente top-level def/class.
    next_decl = re.search(r"\n(?:def |class |# ---)", rest)
    end = start + (next_decl.start() if next_decl else len(rest))
    return src[start:end]


def test_top_level_try_present(cron_src: str):
    body = _extract_aggregate_fn_body(cron_src)
    # Buscar `    try:` al inicio del body (4 espacios = top-level dentro de def).
    assert re.search(r"\n    try:\s*\n", body), (
        "P1-CRON-TOP-LEVEL-TRY regresión: `_aggregate_coherence_block_history_metrics` "
        "ya no tiene try/except/finally top-level. Una excepción no atrapada "
        "saldría silente del scheduler queue."
    )


def test_finally_emits_tick(cron_src: str):
    body = _extract_aggregate_fn_body(cron_src)
    assert re.search(r"\n    finally:\s*\n", body), (
        "P1-CRON-TOP-LEVEL-TRY: el `finally:` top-level falta — sin él, "
        "early returns no emiten tick observable."
    )
    # El finally debe contener un INSERT a pipeline_metrics con node
    # _aggregate_coherence_block_history_metrics + skip_reason.
    finally_match = re.search(r"\n    finally:\s*\n([\s\S]+)$", body)
    assert finally_match is not None
    finally_body = finally_match.group(1)
    assert "INSERT INTO pipeline_metrics" in finally_body, (
        "P1-CRON-TOP-LEVEL-TRY: el finally no inserta a pipeline_metrics — "
        "el tick observable queda perdido."
    )
    assert "skip_reason" in finally_body, (
        "P1-CRON-TOP-LEVEL-TRY: el tick no incluye `skip_reason` — sin él, "
        "los 5+ paths de salida no se distinguen en dashboards."
    )


def test_skip_reasons_canonical_set(cron_src: str):
    """Los códigos de skip_reason del finally deben cubrir todos los paths
    de early-return identificados."""
    body = _extract_aggregate_fn_body(cron_src)
    canonical = {
        "db_core_import_failed",
        "supabase_not_initialized",
        "fetch_plans_failed",
        "unhandled_exception",
    }
    missing = [r for r in canonical if r not in body]
    assert not missing, (
        f"P1-CRON-TOP-LEVEL-TRY: skip_reasons canónicos faltantes: {missing}. "
        f"Cada early-return debe setear su skip_reason correspondiente."
    )


def test_marker_tooltip_present(cron_src: str):
    assert "P1-CRON-TOP-LEVEL-TRY" in cron_src, (
        "Marker tooltip ausente — si renombras el bloque, actualiza este test."
    )
