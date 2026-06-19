"""[P2 audit fresco · 2026-06-18] Batch 2: validación-CI + observabilidad de flota + UX honesta.

Cubre:
  - P2-VALIDATION-INTEGRITY-PURE (#8): helpers puros `_macro_in_band` / `_integrity_in_band` testeables en CI.
  - P2-RESOLUTION-COVERAGE-DRIFT (#13): métrica `resolution_coverage` emitida + cron de drift + registro.
  - P2-REVIEW-FAILED-RATE (#14): cron de tasa de entregas no-fallback que fallan review + registro.
  - P2-CRITICAL-REJECTION-CODE (#20): rechazo crítico emite code distinto (`critical_restriction`), no 503/"IA saturada".

Determinista (funcional para los helpers puros + parser-anchors para crons/wiring que requieren Neon/SSE).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_BE = Path(__file__).resolve().parent.parent
_CRON = (_BE / "cron_tasks.py").read_text(encoding="utf-8")
_GO = (_BE / "graph_orchestrator.py").read_text(encoding="utf-8")
_PLANS = (_BE / "routers" / "plans.py").read_text(encoding="utf-8")
_PLAN_JSX = (_BE.parent / "frontend" / "src" / "pages" / "Plan.jsx").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #8 validation helpers (funcional)
def test_validation_macro_in_band():
    sys.path.insert(0, str(_BE / "scripts"))
    import clinical_validation_export as cve
    assert cve._macro_in_band(100, 100) is True
    assert cve._macro_in_band(112, 100) is True       # techo banda
    assert cve._macro_in_band(90, 100) is True        # piso banda
    assert cve._macro_in_band(113, 100) is False
    assert cve._macro_in_band(89, 100) is False
    assert cve._macro_in_band(50, 0) is False         # target<=0 → no aplica


def test_validation_integrity_in_band():
    sys.path.insert(0, str(_BE / "scripts"))
    import clinical_validation_export as cve
    assert cve._integrity_in_band(100, 100) is True
    assert cve._integrity_in_band(115, 100) is True   # 15% exacto
    assert cve._integrity_in_band(116, 100) is False  # >15%
    assert cve._integrity_in_band(50, 0) is False      # recomputado<=0 → no aplica
    assert cve._INTEGRITY_TOL == 0.15
    assert cve._RES_PCT_FLOOR == 60


# --------------------------------------------------------------------------- #13 resolution_coverage drift
def test_resolution_coverage_metric_emitted():
    assert "P2-RESOLUTION-COVERAGE-METRIC" in _GO
    assert '"node": "resolution_coverage"' in _GO


def test_resolution_coverage_drift_cron_and_registration():
    assert "def _resolution_coverage_drift_alert_job():" in _CRON
    assert "resolution_coverage_drift" in _CRON              # alert_key
    assert "node = 'resolution_coverage'" in _CRON           # lee la métrica correcta
    assert 'get_job("resolution_coverage_drift_alert")' in _CRON  # registrado en el scheduler
    assert "_resolution_coverage_drift_alert_job," in _CRON


# --------------------------------------------------------------------------- #14 review-failed rate
def test_review_failed_rate_cron_and_registration():
    assert "def _review_failed_delivered_rate_alert_job():" in _CRON
    assert "review_failed_delivered_rate_high" in _CRON      # alert_key
    assert "metadata->>'review_passed' = 'false'" in _CRON   # filtro correcto
    assert "node = 'clinical_band'" in _CRON                 # reusa la métrica existente
    assert 'get_job("review_failed_delivered_rate_alert")' in _CRON
    assert "_review_failed_delivered_rate_alert_job," in _CRON


# --------------------------------------------------------------------------- #20 critical-rejection code
def test_backend_emits_critical_restriction_code():
    assert "P2-CRITICAL-REJECTION-CODE" in _PLANS
    # SSE: code distinto en vez de llm_unavailable
    assert "'code': 'critical_restriction'" in _PLANS
    # sync: 422 (no 503) cuando _critical_rejection
    assert "status_code=422" in _PLANS
    # ramifica sobre _critical_rejection ANTES del genérico _is_fallback
    assert 'result.get("_critical_rejection")' in _PLANS


def test_frontend_handles_critical_restriction_code():
    assert "P2-CRITICAL-REJECTION-CODE" in _PLAN_JSX
    assert "critical_restriction" in _PLAN_JSX
    assert "Revisa tus restricciones" in _PLAN_JSX
    # mapea el 422 del fallback síncrono al code
    assert "response2.status === 422" in _PLAN_JSX
