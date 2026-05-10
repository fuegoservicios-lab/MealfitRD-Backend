"""
Tests P0-5: Atomicidad del TZ-drift resync.

Escenario crítico: usuario viaja entre chunks y dos workers del mismo meal_plan
evalúan el temporal gate simultáneamente. Antes:
  - Ambos leían el snapshot stale.
  - Ambos calculaban drift contra el mismo _tz_offset_snapshot viejo.
  - Ambos disparaban push notification (UX duplicada).
  - Ambos hacían UPDATE redundante (idempotente pero generaba bloat).

Ahora con `pg_advisory_xact_lock(hashtextextended(meal_plan_id::text, 0))`:
  - Solo un worker entra al bloque crítico a la vez.
  - El segundo re-lee el snapshot DESPUÉS del lock; si ya está alineado, skip.
  - No-op si pool no disponible (fallback a UPDATE legacy con warning).

Cubre:
  1. Drift real → resync se ejecuta una vez (transacción atómica).
  2. Worker 2 entra al lock con snapshot ya alineado por worker 1 → skip resync.
  3. Pool no disponible → fallback a UPDATE legacy sin crashear.
  4. Sin drift y sin safety triggers → no se entra al bloque resync.
"""
import json
import uuid
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timezone, timedelta

import pytest


# ---------------------------------------------------------------------------
# Helpers para construir un snapshot mínimo
# ---------------------------------------------------------------------------
def _build_snapshot(tz_offset=-180, plan_start_iso=None, totalDays=15, deferrals=0):
    """Snapshot mínimo con los campos que `_check_chunk_learning_ready` usa."""
    if plan_start_iso is None:
        plan_start_iso = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    snap = {
        "totalDays": totalDays,
        "form_data": {
            "tzOffset": tz_offset,
            "tz_offset_minutes": tz_offset,
            "_plan_start_date": plan_start_iso,
        },
    }
    if deferrals:
        snap["_learning_ready_deferrals"] = deferrals
    return snap


def _build_plan_data(days_count=6):
    """plan_data con días previos suficientes para que el gate llegue al bloque TZ."""
    return {
        "days": [
            {"day": d, "meals": [{"name": f"M{d}", "ingredients": ["pollo"]}]}
            for d in range(1, days_count + 1)
        ]
    }


# ---------------------------------------------------------------------------
# Test 1: Drift real → resync atómico se ejecuta (single worker)
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_query")
def test_p0_5_drift_triggers_atomic_resync(mock_query, mock_push):
    """
    Worker único con drift de 120 min: debe entrar al bloque atómico, hacer
    UPDATE de tzOffset y disparar push (primera vez).
    """
    from cron_tasks import _check_chunk_learning_ready

    # tz_offset_snapshot = -180 (Argentina), live = -300 (Nueva York)
    snapshot = _build_snapshot(tz_offset=-180)
    plan_data = _build_plan_data(days_count=6)

    # Mock del SELECT health_profile para retornar tz live
    mock_query.return_value = {"health_profile": {"tz_offset_minutes": -300}}

    # Mock del connection_pool para capturar las queries SQL ejecutadas
    captured_sql = []

    class MockCursor:
        def __init__(self):
            self.last_query = None
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, query, params=None):
            captured_sql.append((query, params))
            self.last_query = query
        def fetchone(self):
            # Re-leer pipeline_snapshot dentro del lock — devolvemos el mismo
            # snapshot stale (escenario donde nadie ha hecho resync aún).
            if self.last_query and "FROM plan_chunk_queue" in self.last_query:
                return {"pipeline_snapshot": snapshot}
            return None

    class MockTransaction:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class MockConn:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def transaction(self): return MockTransaction()
        def cursor(self, **kw): return MockCursor()

    class MockPool:
        def connection(self): return MockConn()

    with patch("db_core.connection_pool", MockPool()):
        if True:
            res = _check_chunk_learning_ready(
                user_id="user-traveler",
                meal_plan_id="plan-tz-1",
                week_number=2,
                days_offset=3,
                plan_data=plan_data,
                snapshot=snapshot,
            )

    # Validaciones:
    # a) Se tomó el advisory lock con hashtextextended del meal_plan_id.
    lock_calls = [c for c in captured_sql if "pg_advisory_xact_lock" in (c[0] or "")]
    assert len(lock_calls) == 1, (
        f"Esperaba 1 llamada a pg_advisory_xact_lock, hubo {len(lock_calls)}"
    )
    assert "hashtextextended" in lock_calls[0][0]
    # [P1-5] La key del lock ahora es namespaced: `meal_plan:tz_resync:<id>` en
    # vez del UUID crudo. Esto evita colisiones accidentales con otros locks por
    # meal_plan (e.g., catchup) y unifica el espacio de keys vía
    # `acquire_meal_plan_advisory_lock(purpose='tz_resync')`.
    assert lock_calls[0][1] == ("meal_plan:tz_resync:plan-tz-1",)

    # b) Se ejecutó el SELECT del pipeline_snapshot DENTRO del lock.
    select_snap = [c for c in captured_sql if c[0] and "SELECT pipeline_snapshot" in c[0]]
    assert len(select_snap) == 1

    # c) Se ejecutó el UPDATE del tzOffset.
    update_tz = [c for c in captured_sql if c[0] and "form_data,tzOffset" in c[0]]
    assert len(update_tz) >= 1, "Esperaba UPDATE de tzOffset tras detectar drift"

    # d) Push notification disparada (drift real, primera vez).
    assert mock_push.called, "Push notification debe dispararse en drift real primera vez"


# ---------------------------------------------------------------------------
# Test 2: Worker 2 entra al lock con snapshot ya alineado → skip
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_query")
def test_p0_5_second_worker_skips_after_first_resync(mock_query, mock_push):
    """
    El primer worker ya hizo el resync (snapshot tiene tz=-300 en DB). El segundo
    worker llega con su frame local stale (tz_offset_snapshot=-180), entra al
    lock, RE-LEE el snapshot fresh (tz=-300), drift=0 contra live=-300, skip.
    """
    from cron_tasks import _check_chunk_learning_ready

    # Worker 2: frame local tiene snapshot stale (-180), pero la DB ya fue
    # actualizada por el worker 1 a -300.
    snapshot_stale = _build_snapshot(tz_offset=-180)  # frame local
    snapshot_in_db = _build_snapshot(tz_offset=-300)  # ya actualizado en DB
    plan_data = _build_plan_data(days_count=6)

    mock_query.return_value = {"health_profile": {"tz_offset_minutes": -300}}

    captured_sql = []

    class MockCursor:
        def __init__(self):
            self.last_query = None
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, query, params=None):
            captured_sql.append((query, params))
            self.last_query = query
        def fetchone(self):
            if self.last_query and "FROM plan_chunk_queue" in self.last_query:
                # CLAVE: devolvemos el snapshot YA ACTUALIZADO (worker 1 ganó).
                return {"pipeline_snapshot": snapshot_in_db}
            return None

    class MockTransaction:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class MockConn:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def transaction(self): return MockTransaction()
        def cursor(self, **kw): return MockCursor()

    class MockPool:
        def connection(self): return MockConn()

    with patch("db_core.connection_pool", MockPool()):
        if True:
            _check_chunk_learning_ready(
                user_id="user-traveler",
                meal_plan_id="plan-tz-2",
                week_number=2,
                days_offset=3,
                plan_data=plan_data,
                snapshot=snapshot_stale,  # ← frame local stale
            )

    # Validación crítica: NO debe haber UPDATE de tzOffset en este worker.
    update_tz = [c for c in captured_sql if c[0] and "form_data,tzOffset" in c[0]]
    assert len(update_tz) == 0, (
        f"Worker 2 NO debe hacer UPDATE redundante: snapshot ya alineado por worker 1. "
        f"UPDATEs detectados: {len(update_tz)}"
    )

    # Push tampoco debe dispararse (evita doble notificación al usuario).
    assert not mock_push.called, (
        "Worker 2 NO debe disparar push duplicada cuando worker 1 ya hizo el resync."
    )


# ---------------------------------------------------------------------------
# Test 3: Pool no disponible → fallback legacy sin crashear
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_p0_5_pool_unavailable_falls_back_to_legacy_update(mock_query, mock_write, mock_push):
    """
    Si connection_pool es None o lanza al obtener conexión, el código cae al
    UPDATE legacy no-atómico (vulnerable a race) en vez de crashear. Mejor un
    resync no serializado que ningún resync.
    """
    from cron_tasks import _check_chunk_learning_ready

    snapshot = _build_snapshot(tz_offset=-180)
    plan_data = _build_plan_data(days_count=6)

    mock_query.return_value = {"health_profile": {"tz_offset_minutes": -300}}

    # Pool a None — el helper debe caer al fallback legacy.
    with patch("db_core.connection_pool", None):
        if True:
            _check_chunk_learning_ready(
                user_id="user-traveler",
                meal_plan_id="plan-tz-3",
                week_number=2,
                days_offset=3,
                plan_data=plan_data,
                snapshot=snapshot,
            )

    # En el fallback legacy se hace un UPDATE bulk de tzOffset vía execute_sql_write.
    update_tz_calls = [
        c for c in mock_write.call_args_list
        if c.args and "form_data,tzOffset" in (c.args[0] or "")
    ]
    assert len(update_tz_calls) >= 1, (
        "Fallback legacy debe ejecutar el UPDATE de tzOffset cuando pool no disponible"
    )


# ---------------------------------------------------------------------------
# Test 4: Sin drift y sin safety triggers → no entra al bloque resync
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_query")
def test_p0_5_no_drift_no_resync(mock_query, mock_push):
    """
    Si live tz == snapshot tz (drift=0) y no hay deferrals acumulados, el bloque
    de resync NO se ejecuta — sanity check de que no estamos haciendo overhead
    innecesario en el camino feliz.
    """
    from cron_tasks import _check_chunk_learning_ready

    snapshot = _build_snapshot(tz_offset=-300)  # mismo que live
    plan_data = _build_plan_data(days_count=6)
    mock_query.return_value = {"health_profile": {"tz_offset_minutes": -300}}

    captured_sql = []

    class MockCursor:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, query, params=None):
            captured_sql.append((query, params))
        def fetchone(self): return None

    class MockTransaction:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class MockConn:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def transaction(self): return MockTransaction()
        def cursor(self, **kw): return MockCursor()

    class MockPool:
        def connection(self): return MockConn()

    with patch("db_core.connection_pool", MockPool()):
        if True:
            _check_chunk_learning_ready(
                user_id="user-stable",
                meal_plan_id="plan-tz-4",
                week_number=2,
                days_offset=3,
                plan_data=plan_data,
                snapshot=snapshot,
            )

    # Sin drift ni deferrals, NO debe haber lock ni UPDATE de tzOffset.
    lock_calls = [c for c in captured_sql if "pg_advisory_xact_lock" in (c[0] or "")]
    assert len(lock_calls) == 0, (
        "Sin drift: no debe tomarse el advisory lock. Se evita overhead en camino feliz."
    )
    assert not mock_push.called
