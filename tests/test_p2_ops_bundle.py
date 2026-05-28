"""[P2-OPS-BUNDLE · 2026-05-26] Tests parser-based + funcionales para el
bundle P2 del audit prod-readiness (post-P1-CRON-BUNDLE).

4 sub-P2 cerrados (P2-5 ya cerrado en P1-2 con `MEALFIT_FACTS_DRAIN_USERS_PER_TICK`):

  - **P2-1 (P2-RACE-FIX-ORPHAN)**: CTE atómico con `FOR UPDATE SKIP LOCKED` +
    UPDATE en single statement en `_cleanup_orphan_chunks`. Cierra race
    "entre SELECT y UPDATE alguien INSERT un meal_plan".
  - **P2-2 (P2-BATCH-ATOMICITY)**: revert del counter `recovery_attempts`
    si `_enqueue_plan_chunk` falla en `_recover_failed_chunks_for_long_plans`.
    Cierra pérdida silenciosa (counter bumpeado sin re-encolado real).
  - **P2-6 (P2-NIGHTLY-OBS)**: per-user `_call_with_timeout` + total budget
    + `_track_cron_consecutive_failure` en
    `_nightly_refresh_all_pending_snapshots`. Paridad con P1-4.
  - **P2-3 (P2-UNPUSHED-AGE)** + **P2-4 (P2-CB-FOSSIL)**: scripts CLI en
    `backend/scripts/`. Tests parser-based del filename + docstring anchor.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"
_SCRIPTS = _BACKEND_ROOT / "scripts"

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# Mismo stub de apscheduler que `test_p1_cron_bundle.py`.
try:
    import apscheduler  # noqa: F401
    import apscheduler.triggers.cron  # noqa: F401
except Exception:
    sys.modules.setdefault("apscheduler", MagicMock())
    sys.modules.setdefault("apscheduler.triggers", MagicMock())
    sys.modules.setdefault("apscheduler.triggers.cron", MagicMock())


def _read_cron() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


def _extract_function(src: str, name: str) -> str:
    pattern = re.compile(
        rf"^def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:\n"
        rf"(?:[ \t]+.*\n|[ \t]*\n)+",
        re.MULTILINE,
    )
    m = pattern.search(src)
    assert m, f"No se encontró def `{name}` en cron_tasks.py."
    return m.group(0)


# ===========================================================================
# Cross-link anchor (test_p2_hist_audit_14_marker_test_link)
# ===========================================================================

def test_marker_anchor_present_in_cron_tasks():
    src = _read_cron()
    assert "P2-OPS-BUNDLE" in src or "P2-RACE-FIX" in src, (
        "Falta anchor del bundle P2 en `cron_tasks.py`."
    )


# ===========================================================================
# P2-1 · P2-RACE-FIX-ORPHAN
# ===========================================================================

def test_orphan_cleanup_uses_for_update_skip_locked():
    src = _read_cron()
    body = _extract_function(src, "_cleanup_orphan_chunks")
    assert "FOR UPDATE SKIP LOCKED" in body, (
        "`_cleanup_orphan_chunks` debe usar `FOR UPDATE SKIP LOCKED` para "
        "cerrar el race entre SELECT y UPDATE. Sin el lock, un INSERT de "
        "meal_plan entre los dos statements puede dejar reservas liberadas "
        "de un chunk que ya no es huérfano."
    )


def test_orphan_cleanup_uses_cte_with_returning():
    """El CTE atómico debe incluir RETURNING para que el caller obtenga los
    IDs cancelados (necesario para el paso post-UPDATE de release_reservations)."""
    src = _read_cron()
    body = _extract_function(src, "_cleanup_orphan_chunks")
    # Debe haber un CTE `WITH orphans AS (...)` + UPDATE...FROM orphans + RETURNING.
    assert "WITH orphans AS" in body, "Falta CTE `WITH orphans AS (...)`."
    assert "RETURNING" in body, (
        "El UPDATE debe incluir `RETURNING` para que el caller obtenga los IDs."
    )


def test_orphan_cleanup_releases_reservations_after_update():
    """release_chunk_reservations debe llamarse DESPUÉS del UPDATE atómico
    (no antes). Si crash entre UPDATE y release, las reservas quedan
    huérfanas pero el otro cron de orphan-reservations las recoge."""
    src = _read_cron()
    body = _extract_function(src, "_cleanup_orphan_chunks")
    pos_update = body.find("RETURNING")
    pos_release = body.find("release_chunk_reservations")
    assert pos_update != -1 and pos_release != -1
    assert pos_update < pos_release, (
        "`release_chunk_reservations` debe llamarse DESPUÉS del UPDATE atómico. "
        "Pre-P2-RACE-FIX el orden era invertido y abría el race."
    )


# ===========================================================================
# P2-2 · P2-BATCH-ATOMICITY
# ===========================================================================

def test_recovery_reverts_counter_on_enqueue_failure():
    src = _read_cron()
    body = _extract_function(src, "_recover_failed_chunks_for_long_plans")
    assert "P2-BATCH-ATOMICITY" in body, (
        "Falta anchor `P2-BATCH-ATOMICITY` en `_recover_failed_chunks_for_long_plans`."
    )
    # Debe haber un try/except en torno al `_enqueue_plan_chunk`.
    m = re.search(
        r"try:\s*\n\s*_enqueue_plan_chunk\s*\(",
        body,
    )
    assert m is not None, (
        "El `_enqueue_plan_chunk` debe estar envuelto en `try:` para detectar "
        "fallos y revertir el counter."
    )
    # Debe haber un revert path con `last_recovery_rollback_at`.
    assert "last_recovery_rollback_at" in body, (
        "El revert path debe persistir `last_recovery_rollback_at` (telemetría "
        "para identificar chunks que entraron en este path)."
    )


def test_recovery_continues_loop_after_enqueue_failure():
    """El except del enqueue debe terminar con `continue` para no romper el batch."""
    src = _read_cron()
    body = _extract_function(src, "_recover_failed_chunks_for_long_plans")
    # Buscar `except Exception as enq_err:` y verificar que su block tiene continue.
    m = re.search(
        r"except\s+Exception\s+as\s+enq_err\s*:\s*\n(?:.*\n){1,30}?\s*continue",
        body,
    )
    assert m is not None, (
        "El except del enqueue debe terminar con `continue` para procesar el "
        "siguiente candidate. Sin `continue`, un enqueue fallido aborta el batch."
    )


# ===========================================================================
# P2-6 · P2-NIGHTLY-OBS
# ===========================================================================

def test_nightly_refresh_uses_call_with_timeout():
    src = _read_cron()
    body = _extract_function(src, "_nightly_refresh_all_pending_snapshots")
    assert "_call_with_timeout" in body, (
        "`_nightly_refresh_all_pending_snapshots` debe envolver "
        "`get_user_inventory_net` en `_call_with_timeout` para evitar que un "
        "user lento bloquee el cron entero (paridad con P1-4)."
    )
    assert "MEALFIT_NIGHTLY_REFRESH_PER_USER_TIMEOUT_S" in body, (
        "Knob `MEALFIT_NIGHTLY_REFRESH_PER_USER_TIMEOUT_S` (default 30, "
        "clamp [5, 300]) debe controlar el timeout per-user."
    )
    assert "MEALFIT_NIGHTLY_REFRESH_TOTAL_BUDGET_S" in body, (
        "Knob `MEALFIT_NIGHTLY_REFRESH_TOTAL_BUDGET_S` (default 600, "
        "clamp [60, 1800]) debe controlar el budget total."
    )


def test_nightly_refresh_uses_consecutive_failure_tracker():
    src = _read_cron()
    body = _extract_function(src, "_nightly_refresh_all_pending_snapshots")
    assert "_track_cron_consecutive_failure" in body, (
        "El cron debe invocar `_track_cron_consecutive_failure` con su alert_key."
    )
    assert "nightly_refresh_all_pending_snapshots_failures_burst" in body, (
        "Convención `<cron>_failures_burst` debe respetarse + estar documentada "
        "en `docs/system_alerts_resolution_table.md`."
    )


# ===========================================================================
# P2-3 · P2-UNPUSHED-AGE script
# ===========================================================================

def test_unpushed_age_script_exists():
    p = _SCRIPTS / "check_unpushed_age.py"
    assert p.exists(), (
        f"Falta script {p}. Es el tooling local que detecta commits unpushed "
        f"con age > threshold (root-cause de deploy_lag_drift_vs_expected del "
        f"P0-AUDIT 2026-05-25)."
    )


def test_unpushed_age_script_has_anchor():
    p = _SCRIPTS / "check_unpushed_age.py"
    src = p.read_text(encoding="utf-8")
    assert "P2-UNPUSHED-AGE" in src, "Falta anchor `P2-UNPUSHED-AGE` en el script."


def test_unpushed_age_script_main_executable():
    """El script debe tener un `if __name__ == '__main__':` para CLI usage."""
    p = _SCRIPTS / "check_unpushed_age.py"
    src = p.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in src, (
        "Falta el main guard — script no es invocable como CLI."
    )


def test_unpushed_age_supports_max_age_hours_arg():
    """El script debe aceptar --max-age-hours (default 24)."""
    p = _SCRIPTS / "check_unpushed_age.py"
    src = p.read_text(encoding="utf-8")
    assert "--max-age-hours" in src, "Falta flag `--max-age-hours`."
    assert "default=24" in src, "Default del threshold debe ser 24."


def test_unpushed_age_supports_include_dirty():
    """Bonus: el script debe soportar `--include-dirty` para pre-commit hooks."""
    p = _SCRIPTS / "check_unpushed_age.py"
    src = p.read_text(encoding="utf-8")
    assert "--include-dirty" in src, "Falta flag `--include-dirty`."


def test_unpushed_age_does_not_import_backend_runtime():
    """El script NO debe importar módulos del runtime backend
    (cron_tasks, db_*, graph_orchestrator). Pensado para correr standalone
    en local, no en el contenedor desplegado."""
    p = _SCRIPTS / "check_unpushed_age.py"
    src = p.read_text(encoding="utf-8")
    forbidden = ["import cron_tasks", "from cron_tasks", "import db_", "from db_",
                 "import graph_orchestrator", "from graph_orchestrator"]
    for f in forbidden:
        assert f not in src, (
            f"Script importa runtime backend (`{f}`) — DEBE ser standalone."
        )


# ===========================================================================
# P2-4 · P2-CB-FOSSIL script
# ===========================================================================

def test_cb_fossil_script_exists():
    p = _SCRIPTS / "cleanup_stale_cb_rows.py"
    assert p.exists(), f"Falta script {p}."


def test_cb_fossil_script_has_anchor():
    p = _SCRIPTS / "cleanup_stale_cb_rows.py"
    src = p.read_text(encoding="utf-8")
    assert "P2-CB-FOSSIL" in src


def test_cb_fossil_preview_default():
    """Sin `--apply`, el script SOLO debe hacer preview SELECT, no DELETE."""
    p = _SCRIPTS / "cleanup_stale_cb_rows.py"
    src = p.read_text(encoding="utf-8")
    # Hay un branch `if not args.apply:` que retorna antes del DELETE.
    assert re.search(r"if not args\.apply\s*:", src), (
        "Falta guard `if not args.apply:` que previene DELETE accidental."
    )


def test_cb_fossil_requires_models_flag_for_apply():
    """`--apply` sin `--models` debe abortar (defensa contra DELETE masivo)."""
    p = _SCRIPTS / "cleanup_stale_cb_rows.py"
    src = p.read_text(encoding="utf-8")
    # Hay un branch `if not selected_models:` con `return 2` después de --apply.
    m = re.search(
        r"--apply requiere --models",
        src,
    )
    assert m is not None, (
        "Falta el mensaje de error que exige --models al usar --apply."
    )


def test_cb_fossil_filters_zero_canonical():
    """El DELETE debe filtrar `failures=0 AND last_failure=0 AND is_open=false`
    (zero canonical) — defense contra borrar CBs activos."""
    p = _SCRIPTS / "cleanup_stale_cb_rows.py"
    src = p.read_text(encoding="utf-8")
    assert "(value->>'failures')::int = 0" in src
    assert "(value->>'last_failure')::float = 0" in src
    assert "is_open" in src


# ===========================================================================
# Funcional: argparse del script unpushed_age
# ===========================================================================

def test_unpushed_age_script_invocable():
    """Smoke test: el script puede invocarse con --help sin crashear."""
    import subprocess
    p = _SCRIPTS / "check_unpushed_age.py"
    result = subprocess.run(
        [sys.executable, str(p), "--help"],
        capture_output=True, text=True, timeout=10,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, (
        f"`{p} --help` retornó {result.returncode}. stderr: {result.stderr[:500]}"
    )
    assert "--max-age-hours" in result.stdout
    assert "--include-dirty" in result.stdout


def test_cb_fossil_script_invocable():
    """Smoke test: cleanup_stale_cb_rows --help no crashea."""
    import subprocess
    p = _SCRIPTS / "cleanup_stale_cb_rows.py"
    result = subprocess.run(
        [sys.executable, str(p), "--help"],
        capture_output=True, text=True, timeout=10,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0
    assert "--apply" in result.stdout
    assert "--models" in result.stdout
