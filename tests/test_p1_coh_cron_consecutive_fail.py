"""[P1-COH-CRON-CONSECUTIVE-FAIL · 2026-05-15] Anchor parser-based: el cron
`_shopping_coherence_alert_job` (diario, 04:00 UTC) debe rastrear fallos
consecutivos en `app_kv_store` y emitir alert cuando ≥ knob threshold.

Contexto:
    P2-LIVE-9 (2026-05-11) introdujo el tick observable per-run (skip_reason,
    eval_errors, persist_errors). Pero NO había alerta automática cuando
    múltiples runs consecutivos fallaban — SRE solo lo notaría revisando
    dashboards manualmente.

Fix:
    - Counter `coh_alert_job_failures_count` en app_kv_store con payload
      `{count, last_failure_at, last_skip_reason, ...}`.
    - INCR cuando skip_reason!=None OR eval_errors>0 OR persist_errors>0.
    - RESET a 0 en run exitoso (skip_reason=None, eval_errors=0, persist_errors=0).
    - Knob `MEALFIT_COH_ALERT_CONSECUTIVE_FAIL_THRESHOLD` (default 2).
    - Alert `shopping_coherence_alert_job_failures_burst` con auto-resolve
      cuando counter vuelve a 0.

Tooltip-anchor: P1-COH-CRON-CONSECUTIVE-FAIL-START
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


def _extract_coh_alert_body(src: str) -> str:
    anchor = re.search(
        r"def _shopping_coherence_alert_job\b[^\n]*\n",
        src,
    )
    assert anchor is not None, (
        "P1-COH-CRON-CONSECUTIVE-FAIL: `_shopping_coherence_alert_job` no "
        "encontrada — ¿renombrada? actualizar test."
    )
    start = anchor.end()
    rest = src[start:]
    next_decl = re.search(r"\n(?:def |class |# ---)", rest)
    end = start + (next_decl.start() if next_decl else len(rest))
    return src[start:end]


def test_counter_kv_key_present(cron_src: str):
    body = _extract_coh_alert_body(cron_src)
    assert "coh_alert_job_failures_count" in body, (
        "P1-COH-CRON-CONSECUTIVE-FAIL: el KV key `coh_alert_job_failures_count` "
        "no aparece — sin él el contador de fallos consecutivos no persiste."
    )


def test_knob_registered(cron_src: str):
    body = _extract_coh_alert_body(cron_src)
    assert "MEALFIT_COH_ALERT_CONSECUTIVE_FAIL_THRESHOLD" in body, (
        "P1-COH-CRON-CONSECUTIVE-FAIL: el knob no se lee — sin él el umbral "
        "queda hardcoded y SRE no puede ajustarlo sin redeploy."
    )


def test_alert_key_canonical(cron_src: str):
    body = _extract_coh_alert_body(cron_src)
    assert "shopping_coherence_alert_job_failures_burst" in body, (
        "P1-COH-CRON-CONSECUTIVE-FAIL: alert_key canónica ausente — la alerta "
        "automática no se emite cuando el threshold se sobrepasa."
    )


def test_auto_resolve_on_success(cron_src: str):
    """Run exitoso debe resolver la alert (UPDATE system_alerts SET resolved_at = NOW())."""
    body = _extract_coh_alert_body(cron_src)
    assert re.search(
        r"UPDATE system_alerts SET resolved_at = NOW\(\)[\s\S]{0,200}shopping_coherence_alert_job_failures_burst",
        body,
    ), (
        "P1-COH-CRON-CONSECUTIVE-FAIL: el auto-resolve de la alert tras run "
        "exitoso falta — la alerta quedaría pegada hasta intervención manual."
    )


def test_marker_tooltip_present(cron_src: str):
    assert "P1-COH-CRON-CONSECUTIVE-FAIL" in cron_src, (
        "Marker tooltip ausente — si renombras el bloque, actualiza este test."
    )
