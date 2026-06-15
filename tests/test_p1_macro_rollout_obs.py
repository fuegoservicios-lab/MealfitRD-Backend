"""[P1-MACRO-ROLLOUT-OBS · 2026-06-14] Cron de tasa de FALLBACK de generación.

Prerequisito de observabilidad para activar el motor de macros (MEALFIT_MACRO_SOLVER_ENABLED) con
canary: cuenta la fracción de planes entregados como fallback matemático (métrica `clinical_band`,
`delivered_was_fallback='true'`) y emite el alert `plan_fallback_rate_high` si supera el umbral.
La auditoría midió 45% de fallback sin explicar → esta alerta lo vuelve vigilable durante el rollout.

Cubre:
  A. fallback alto → INSERT alert `plan_fallback_rate_high`.
  B. fallback bajo → UPDATE resolved_at (auto-resolve).
  C. muestra insuficiente → solo tick observable (no toca system_alerts).
  D. parser-based anchor: marker + cron registrado.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cron_tasks

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PATH = _BACKEND_ROOT / "cron_tasks.py"


def _run_with(total, fallback):
    q = MagicMock(return_value=[{"total": total, "fallback": fallback}])
    w = MagicMock()
    with patch.object(cron_tasks, "execute_sql_query", q), \
         patch.object(cron_tasks, "execute_sql_write", w):
        cron_tasks._plan_fallback_rate_alert_job()
    return w


def _writes_to_system_alerts(w):
    return [c for c in w.call_args_list if "system_alerts" in c.args[0]]


def test_high_fallback_emits_alert():
    w = _run_with(total=100, fallback=45)  # 45% > umbral 25%
    sa = _writes_to_system_alerts(w)
    assert sa, "no escribió a system_alerts con fallback alto"
    sql, params = sa[0].args[0], sa[0].args[1]
    assert "INSERT INTO system_alerts" in sql
    assert params[0] == "plan_fallback_rate_high"


def test_low_fallback_resolves():
    w = _run_with(total=100, fallback=10)  # 10% <= umbral 25%
    sa = _writes_to_system_alerts(w)
    assert sa, "no resolvió la alerta con fallback bajo"
    sql, params = sa[0].args[0], sa[0].args[1]
    assert "UPDATE system_alerts SET resolved_at" in sql
    assert params[0] == "plan_fallback_rate_high"


def test_insufficient_samples_only_tick():
    w = _run_with(total=5, fallback=3)  # < min_samples (10) → skip
    assert _writes_to_system_alerts(w) == [], "no debe tocar system_alerts con muestra insuficiente"
    # El tick observable a pipeline_metrics SÍ debe emitirse siempre.
    assert any("pipeline_metrics" in c.args[0] for c in w.call_args_list)


def test_marker_and_cron_registered():
    src = _CRON_PATH.read_text(encoding="utf-8")
    assert "P1-MACRO-ROLLOUT-OBS" in src
    assert "def _plan_fallback_rate_alert_job(" in src
    assert 'id="plan_fallback_rate_alert"' in src
