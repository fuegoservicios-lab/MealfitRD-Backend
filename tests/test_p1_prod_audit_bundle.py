"""[P1-PROD-AUDIT-BUNDLE · 2026-05-28] Regression guards para los 4 P1 del
audit prod-readiness 2026-05-28 (0 P0 nuevos).

Parser-based (lee el source como texto, sin importar módulos pesados → corre
con `pytest --noconftest`). Cubre:

  P1-1 ADMIN-TOKEN-CONSTTIME      — `_verify_admin_token` usa hmac.compare_digest.
  P1-2 QUALITY-DEGRADED-ALL-BRANCHES — `_mark_plan_result_quality_degraded` en
                                        las 3 ramas "end" degradadas de should_retry.
  P1-3 ZOMBIE-FAILED-NOTIFY       — `_notify_zombie_plan_generation_failed` en la
                                        rama 0-días de _finalize_zombie_partial_plans.
  P1-4 PIPELINE-METRICS-RETENTION — cron `_purge_old_pipeline_metrics` + knobs +
                                        migración BRIN SSOT dual-dir.
  Marker — `_LAST_KNOWN_PFIX` fecha >= 2026-05-28.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE_ROOT = _BACKEND_ROOT.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_PLANS = _BACKEND_ROOT / "routers" / "plans.py"
_ORQ = _BACKEND_ROOT / "graph_orchestrator.py"
_CRON = _BACKEND_ROOT / "cron_tasks.py"
_MIGRATION_NAME = "p1_pipeline_metrics_retention_brin_idx_2026_05_28.sql"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P1-1: constant-time admin token compare
# ---------------------------------------------------------------------------
def test_p1_1_admin_token_constant_time():
    src = _read(_PLANS)
    assert re.search(r"^\s*import hmac\b", src, re.MULTILINE), (
        "P1-1: `import hmac` ausente en plans.py."
    )
    # El cuerpo de `_verify_admin_token` debe comparar con compare_digest.
    body = src[src.index("def _verify_admin_token("):]
    body = body[: body.index("\n\n\n")] if "\n\n\n" in body else body[:1500]
    assert "hmac.compare_digest(token, cron_secret)" in body, (
        "P1-1: `_verify_admin_token` no usa hmac.compare_digest."
    )


def test_p1_1_no_plain_token_compare_regression():
    body = _read(_PLANS)
    body = body[body.index("def _verify_admin_token("):]
    body = body[:1500]
    assert "token != cron_secret" not in body, (
        "P1-1 regresión: volvió el `!=` plano (side-channel de timing)."
    )


# ---------------------------------------------------------------------------
# P1-2: _quality_degraded en todas las ramas "end" degradadas
# ---------------------------------------------------------------------------
def test_p1_2_helper_defined():
    src = _read(_ORQ)
    assert "def _mark_plan_result_quality_degraded(" in src, (
        "P1-2: helper _mark_plan_result_quality_degraded ausente."
    )
    assert "P1-QUALITY-DEGRADED-ALL-BRANCHES" in src, (
        "P1-2: tooltip-anchor P1-QUALITY-DEGRADED-ALL-BRANCHES ausente."
    )


@pytest.mark.parametrize("reason", ["high_contextual", "invalid_pipeline_start", "budget_exhausted"])
def test_p1_2_helper_called_in_each_degraded_branch(reason):
    src = _read(_ORQ)
    needle = f'_mark_plan_result_quality_degraded(state, reason="{reason}"'
    assert needle in src, (
        f"P1-2: rama '{reason}' de should_retry no llama al helper "
        f"_mark_plan_result_quality_degraded → usuario sin banner."
    )


# ---------------------------------------------------------------------------
# P1-3: notify user en la rama 0-días de _finalize_zombie_partial_plans
# ---------------------------------------------------------------------------
def test_p1_3_notify_helper_defined():
    src = _read(_CRON)
    assert "def _notify_zombie_plan_generation_failed(" in src, (
        "P1-3: helper _notify_zombie_plan_generation_failed ausente."
    )
    assert "P1-ZOMBIE-FAILED-NOTIFY" in src, (
        "P1-3: tooltip-anchor P1-ZOMBIE-FAILED-NOTIFY ausente."
    )
    # Setea el banner _user_action_required con reason_code canónico.
    helper = src[src.index("def _notify_zombie_plan_generation_failed("):]
    helper = helper[: helper.index("\ndef _finalize_zombie_partial_plans(")]
    assert "_user_action_required" in helper, (
        "P1-3: el helper no setea _user_action_required."
    )
    assert "generation_failed" in helper, (
        "P1-3: el helper no usa reason/reason_code 'generation_failed'."
    )
    assert "_dispatch_push_notification(" in helper, (
        "P1-3: el helper no despacha push best-effort."
    )


def test_p1_3_invoked_only_for_failed_branch():
    src = _read(_CRON)
    # La invocación vive bajo el guard `if new_status == "failed" and user_id:`.
    assert 'if new_status == "failed" and user_id:' in src, (
        "P1-3: guard de invocación ausente."
    )
    assert "_notify_zombie_plan_generation_failed(plan_id, user_id)" in src, (
        "P1-3: la rama failed de _finalize_zombie_partial_plans no notifica."
    )


# ---------------------------------------------------------------------------
# P1-4: retención de pipeline_metrics (cron + knobs + migración BRIN SSOT)
# ---------------------------------------------------------------------------
def test_p1_4_retention_cron_defined_and_registered():
    src = _read(_CRON)
    assert "def _purge_old_pipeline_metrics(" in src, (
        "P1-4: cron _purge_old_pipeline_metrics ausente."
    )
    assert "P1-PIPELINE-METRICS-RETENTION" in src, (
        "P1-4: tooltip-anchor P1-PIPELINE-METRICS-RETENTION ausente."
    )
    assert 'id="purge_old_pipeline_metrics"' in src, (
        "P1-4: el cron no está registrado en el scheduler."
    )
    assert "DELETE FROM pipeline_metrics" in src, (
        "P1-4: el cron no hace DELETE de pipeline_metrics."
    )


@pytest.mark.parametrize("knob", [
    "MEALFIT_PIPELINE_METRICS_GC_ENABLED",
    "MEALFIT_PIPELINE_METRICS_RETENTION_DAYS",
    "MEALFIT_PIPELINE_METRICS_GC_MAX_ROWS",
    "MEALFIT_PIPELINE_METRICS_GC_INTERVAL_HOURS",
])
def test_p1_4_knobs_present(knob):
    assert knob in _read(_CRON), f"P1-4: knob {knob} ausente."


def test_p1_4_migration_ssot_dual_dir_identical():
    backend_mig = _BACKEND_ROOT / "migrations" / _MIGRATION_NAME
    root_mig = _WORKSPACE_ROOT / "migrations" / _MIGRATION_NAME
    assert backend_mig.exists(), f"P1-4: falta migración backend {backend_mig}."
    assert root_mig.exists(), f"P1-4: falta migración root {root_mig}."
    assert backend_mig.read_bytes() == root_mig.read_bytes(), (
        "P1-4: migración SSOT drift entre los dos dirs (P3-MIGRATIONS-SSOT)."
    )


def test_p1_4_migration_creates_brin_idempotent():
    mig = (_BACKEND_ROOT / "migrations" / _MIGRATION_NAME).read_text(encoding="utf-8")
    assert "CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_created_brin" in mig, (
        "P1-4: la migración no crea el índice BRIN idempotente."
    )
    assert "USING brin (created_at)" in mig, (
        "P1-4: el índice no es BRIN sobre created_at."
    )
    assert "RAISE EXCEPTION" in mig, "P1-4: falta sanity check post-apply."


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------
def test_last_known_pfix_meets_bundle_floor():
    src = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m is not None, "marker _LAST_KNOWN_PFIX no encontrado."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m is not None, f"marker sin fecha ISO: {marker!r}."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    assert marker_date >= date(2026, 5, 28), (
        f"marker `{marker}` (fecha {marker_date}) por debajo del floor del "
        f"bundle P1-PROD-AUDIT-BUNDLE (2026-05-28)."
    )
