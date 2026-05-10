"""[P2-6 · 2026-05-08] Tests del cron `_alert_atomic_pool_fallback`.

Bug original (audit 2026-05-07):
  `update_user_health_profile_atomic` en `backend/db_profiles.py` cae al
  path no-atómico (lost-update silencioso) cuando `connection_pool=None` y
  `MEALFIT_REQUIRE_ATOMIC_POOL≠1` (default). El counter `_POOL_FALLBACK_STATE`
  se incrementaba pero solo era observable vía `/api/system/atomic-pool-health`
  (polling manual). En prod sin alerting, una misconfig de pool podía durar
  días enmascarada — cada fallback emite WARNING pero no llega a system_alerts.

Decisión (no-default-change):
  El plan original sugería cambiar `REQUIRE_ATOMIC_POOL` default a True. NO se
  cambió: rompería dev/tests/scripts locales sin pool, y el usuario prefiere
  knob-based decisions (memoria `feedback_knob_preference`). La prod debe
  setear `MEALFIT_REQUIRE_ATOMIC_POOL=1` en su deploy.

Fix entregado:
  Cron `_alert_atomic_pool_fallback` (cron_tasks.py, registrado cada
  POOL_FALLBACK_ALERT_INTERVAL_MINUTES=30 min default):
    1. Lee snapshot via `db_profiles.get_atomic_pool_fallback_snapshot()`.
    2. Si `fallback_count > 0` Y `last_at` está dentro de
       POOL_FALLBACK_ALERT_WINDOW_MINUTES (60 default), persiste alert
       dedupe ('atomic_pool_fallback_active') con severity=critical.
    3. Cooldown via `triggered_at > NOW() - cooldown_hours` evita spam.
    4. Self-healing sweep: si `last_at` está fuera del window, marca la
       alerta abierta como `resolved_at=NOW()`. Sin cron de cleanup
       separado.

Cobertura:
  - Counter=0 → no alerta (sweep ejecuta UPDATE resolved_at).
  - Counter>0 + last_at reciente → INSERT con severity=critical y metadata.
  - Counter>0 + last_at viejo (>window) → no alerta + sweep resolve.
  - Cooldown activo → segundo tick no re-inserta.
  - last_at no parseable → fail-graceful (no excepción).
  - Snapshot raise → log warning + return sin crash.
  - Constants registradas con defaults razonables.
"""
import importlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_env():
    keys = [
        "MEALFIT_POOL_FALLBACK_ALERT_INTERVAL_MINUTES",
        "MEALFIT_POOL_FALLBACK_ALERT_WINDOW_MINUTES",
        "MEALFIT_POOL_FALLBACK_ALERT_COOLDOWN_HOURS",
    ]
    snap = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in snap.items():
        if v is not None:
            os.environ[k] = v


def _import_cron():
    if "cron_tasks" in sys.modules:
        importlib.reload(sys.modules["cron_tasks"])
    else:
        import cron_tasks  # noqa: F401
    return sys.modules["cron_tasks"]


def _make_snapshot(fallback_count: int, *, last_at_minutes_ago: int = 1,
                   pool_available: bool = False, strict_mode: bool = False,
                   user_id: str = "user-abc") -> dict:
    """Construye un snapshot con `last_at` relativo a NOW."""
    now = datetime.now(timezone.utc)
    last_at = (now - timedelta(minutes=last_at_minutes_ago)).isoformat() if fallback_count else None
    return {
        "fallback_count": fallback_count,
        "first_at": last_at,
        "last_at": last_at,
        "last_user_id": user_id if fallback_count else None,
        "pool_available": pool_available,
        "strict_mode": strict_mode,
    }


# ---------------------------------------------------------------------------
# 1. Constants registradas
# ---------------------------------------------------------------------------
def test_constants_registered_with_safe_defaults():
    """Las 3 constants nuevas deben tener defaults razonables y clamps."""
    import constants
    assert constants.POOL_FALLBACK_ALERT_INTERVAL_MINUTES >= 5
    assert constants.POOL_FALLBACK_ALERT_WINDOW_MINUTES >= 5
    assert constants.POOL_FALLBACK_ALERT_COOLDOWN_HOURS >= 1


# ---------------------------------------------------------------------------
# 2. Counter=0 → no alerta + sweep auto-resolve
# ---------------------------------------------------------------------------
def test_counter_zero_no_insert_but_runs_sweep():
    cron = _import_cron()
    snap = _make_snapshot(fallback_count=0)
    with patch("cron_tasks.get_atomic_pool_fallback_snapshot", return_value=snap, create=True), \
         patch("cron_tasks.execute_sql_write") as mock_write, \
         patch("cron_tasks.execute_sql_query") as mock_query, \
         patch("db_profiles.get_atomic_pool_fallback_snapshot", return_value=snap):
        mock_query.return_value = None
        cron._alert_atomic_pool_fallback()
    # Debe haber UN UPDATE de auto-resolve (sweep) y NINGÚN INSERT.
    sql_calls = [str(c.args[0]) if c.args else "" for c in mock_write.call_args_list]
    inserts = [s for s in sql_calls if "INSERT INTO system_alerts" in s]
    updates_resolve = [s for s in sql_calls if "UPDATE system_alerts" in s and "resolved_at = NOW()" in s]
    assert len(inserts) == 0, "No debe haber INSERT cuando counter=0"
    assert len(updates_resolve) == 1, "Debe haber UPDATE de sweep auto-resolve"


# ---------------------------------------------------------------------------
# 3. Counter>0 + reciente → INSERT critical
# ---------------------------------------------------------------------------
def test_counter_positive_recent_inserts_critical_alert():
    cron = _import_cron()
    snap = _make_snapshot(fallback_count=3, last_at_minutes_ago=5)
    with patch("cron_tasks.execute_sql_write") as mock_write, \
         patch("cron_tasks.execute_sql_query") as mock_query, \
         patch("db_profiles.get_atomic_pool_fallback_snapshot", return_value=snap):
        mock_query.return_value = None  # No cooldown activo
        cron._alert_atomic_pool_fallback()
    sql_calls = [str(c.args[0]) for c in mock_write.call_args_list]
    inserts = [s for s in sql_calls if "INSERT INTO system_alerts" in s]
    assert len(inserts) == 1, f"Esperado 1 INSERT, got {len(inserts)}: {sql_calls!r}"
    # Verificar que el segundo argumento del INSERT contiene severity y metadata.
    insert_args = mock_write.call_args_list[0].args[1]
    # alert_key, severity, title, message, metadata_json, affected_users_json
    assert insert_args[0] == "atomic_pool_fallback_active"
    assert insert_args[1] == "critical"
    metadata = json.loads(insert_args[4])
    assert metadata["fallback_count"] == 3
    assert metadata["pool_available"] is False
    assert "last_at" in metadata


# ---------------------------------------------------------------------------
# 4. Counter>0 + last_at viejo → no alerta + sweep
# ---------------------------------------------------------------------------
def test_counter_positive_old_no_insert_runs_sweep():
    cron = _import_cron()
    # last_at hace 2 horas, fuera del window de 60 min.
    snap = _make_snapshot(fallback_count=2, last_at_minutes_ago=120)
    with patch("cron_tasks.execute_sql_write") as mock_write, \
         patch("cron_tasks.execute_sql_query") as mock_query, \
         patch("db_profiles.get_atomic_pool_fallback_snapshot", return_value=snap):
        mock_query.return_value = None
        cron._alert_atomic_pool_fallback()
    sql_calls = [str(c.args[0]) for c in mock_write.call_args_list]
    inserts = [s for s in sql_calls if "INSERT INTO system_alerts" in s]
    updates_resolve = [s for s in sql_calls if "UPDATE system_alerts" in s and "resolved_at = NOW()" in s]
    assert len(inserts) == 0, "No alertar si last_at fuera del window"
    assert len(updates_resolve) == 1, "Sweep debe correr para auto-resolve"


# ---------------------------------------------------------------------------
# 5. Cooldown activo → no re-INSERT
# ---------------------------------------------------------------------------
def test_cooldown_blocks_repeated_insert():
    cron = _import_cron()
    snap = _make_snapshot(fallback_count=5, last_at_minutes_ago=2)
    with patch("cron_tasks.execute_sql_write") as mock_write, \
         patch("cron_tasks.execute_sql_query") as mock_query, \
         patch("db_profiles.get_atomic_pool_fallback_snapshot", return_value=snap):
        # Simular alert existente dentro de cooldown.
        mock_query.return_value = {"triggered_at": datetime.now(timezone.utc).isoformat()}
        cron._alert_atomic_pool_fallback()
    sql_calls = [str(c.args[0]) for c in mock_write.call_args_list]
    inserts = [s for s in sql_calls if "INSERT INTO system_alerts" in s]
    assert len(inserts) == 0, "Cooldown debe bloquear el INSERT"


# ---------------------------------------------------------------------------
# 6. last_at no parseable → fail-graceful
# ---------------------------------------------------------------------------
def test_unparseable_last_at_does_not_crash():
    cron = _import_cron()
    snap = {
        "fallback_count": 1,
        "first_at": "garbage-not-iso",
        "last_at": "garbage-not-iso",
        "last_user_id": "user-x",
        "pool_available": False,
        "strict_mode": False,
    }
    with patch("cron_tasks.execute_sql_write") as mock_write, \
         patch("cron_tasks.execute_sql_query") as mock_query, \
         patch("db_profiles.get_atomic_pool_fallback_snapshot", return_value=snap):
        mock_query.return_value = None
        # No debe levantar excepción.
        cron._alert_atomic_pool_fallback()
    # Como last_at no parseó, is_recent=False → sweep, no INSERT.
    sql_calls = [str(c.args[0]) for c in mock_write.call_args_list]
    inserts = [s for s in sql_calls if "INSERT INTO system_alerts" in s]
    assert len(inserts) == 0


# ---------------------------------------------------------------------------
# 7. Snapshot raise → fail-graceful
# ---------------------------------------------------------------------------
def test_snapshot_exception_does_not_crash():
    cron = _import_cron()

    def _broken():
        raise RuntimeError("db_profiles broken")

    with patch("db_profiles.get_atomic_pool_fallback_snapshot", side_effect=_broken), \
         patch("cron_tasks.execute_sql_write") as mock_write:
        # No debe levantar.
        cron._alert_atomic_pool_fallback()
    # Sin snapshot no se persiste nada.
    assert mock_write.call_count == 0


# ---------------------------------------------------------------------------
# 8. Smoke estructural: cron registrado en register_chunk_scheduler
# ---------------------------------------------------------------------------
def test_cron_registered_in_scheduler_setup():
    """El bloque de registro `if not scheduler.get_job("alert_atomic_pool_fallback")`
    debe existir en cron_tasks.py para que el cron se active al startup."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "cron_tasks.py"
    content = src.read_text(encoding="utf-8")
    assert 'scheduler.get_job("alert_atomic_pool_fallback")' in content, (
        "Cron no registrado: falta bloque `if not scheduler.get_job(...)` "
        "para `alert_atomic_pool_fallback`. Se registró el cron pero no "
        "se activará al startup."
    )
    assert "_alert_atomic_pool_fallback," in content, (
        "scheduler.add_job debe referenciar la función directamente."
    )


# ---------------------------------------------------------------------------
# 9. Smoke estructural: severity=critical hardcodeado
# ---------------------------------------------------------------------------
def test_alert_severity_is_critical_in_source():
    """Lost-updates silenciosos en health_profile justifican severity=critical
    (no warning). Si futuro refactor lo bajara a warning, este test pita."""
    import pathlib, re
    src = (pathlib.Path(__file__).parent.parent / "cron_tasks.py").read_text(encoding="utf-8")
    fn_start = src.find("def _alert_atomic_pool_fallback(")
    fn_end = src.find("\ndef ", fn_start + 1)
    block = src[fn_start:fn_end]
    assert re.search(r'severity\s*=\s*["\']critical["\']', block), (
        "Severity de la alerta debe ser 'critical'. Lost-updates en "
        "health_profile son silent data corruption — no warning."
    )
