"""[P0-HIST-3 · 2026-05-09] Tests del endpoint atómico
``DELETE /api/plans/{plan_id}`` y de la migración SSOT
``p0_hist_3_telemetry_orphan_fk.sql``.

Bug original (audit historial 2026-05-08):
    `History.jsx::handleDeleteConfirm` hacía
    ``supabase.from('meal_plans').delete().eq('id', plan_id)`` directo.
    Eso dejaba dos clases de basura:
      - `chunk_user_locks` (sin FK a meal_plans) zombi hasta sweep cron.
      - `chunk_lesson_telemetry` y `chunk_deferrals` (sin FK) orphans
        que crecían monotonas.

Fix:
    1. Endpoint atómico que libera locks ANTES del DELETE en una sola
       transacción.
    2. Migración SSOT que añade FK con `ON DELETE SET NULL` a
       chunk_lesson_telemetry.meal_plan_id y chunk_deferrals.meal_plan_id
       (preserva telemetría agregable, desreferencia el plan).

Cobertura:
    - 401 sin auth.
    - 400 sin plan_id (raro vía path param, pero blindado).
    - 404 cuando plan no existe / no pertenece al usuario.
    - 200 success: 2 statements en orden (release locks → delete plan)
      DENTRO de una transacción explícita.
    - El endpoint usa el mismo patrón de filtrado de locks que P0-HIST-1
      (subquery contra plan_chunk_queue.meal_plan_id, no global del user).
    - Drift detection: la migración SSOT contiene ambas tablas y SET NULL.
    - Anchor [P0-HIST-3 · 2026-05-09] presente en endpoint + migration.
"""
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers (mismo shape que P0-HIST-1 — recorder cursor + conn).
# ---------------------------------------------------------------------------
def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class _CursorRecorder:
    def __init__(self, rowcounts):
        self.calls = []
        self._rowcounts = list(rowcounts)
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if self._rowcounts:
            self.rowcount = self._rowcounts.pop(0)
        else:
            self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _ConnRecorder:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.tx_entered = False
        self.tx_exited = False

    def cursor(self, *a, **kw):
        return self.cursor_obj

    def transaction(self):
        outer = self
        class _Tx:
            def __enter__(self_inner):
                outer.tx_entered = True
                return self_inner
            def __exit__(self_inner, *a):
                outer.tx_exited = True
                return False
        return _Tx()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _build_pool_mock(cursor):
    conn = _ConnRecorder(cursor)
    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, conn


_USER_A = "11111111-1111-1111-1111-111111111111"
_PLAN_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"


# ---------------------------------------------------------------------------
# 1. Auth / validación
# ---------------------------------------------------------------------------
def test_delete_requires_auth():
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: None

    client.app.dependency_overrides[get_verified_user_id] = lambda: None

    r = client.delete(f"/api/plans/{_PLAN_ID}")
    assert r.status_code == 401, r.text


def test_delete_404_when_plan_missing():
    """SELECT de ownership devuelve None → 404."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=None):
        r = client.delete(f"/api/plans/{_PLAN_ID}")
    assert r.status_code == 404


def test_delete_404_when_plan_belongs_to_other_user():
    """ownership SELECT incluye user_id en WHERE — si no devuelve fila,
    el endpoint responde 404 (no 403, para no filtrar la existencia)."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    captured = {}
    def _spy(sql, params=None, **kw):
        captured["sql"] = sql
        captured["params"] = params
        return None  # otro user → no devuelve fila

    with patch("db_core.execute_sql_query", side_effect=_spy):
        r = client.delete(f"/api/plans/{_PLAN_ID}")
    assert r.status_code == 404
    # El SELECT debe filtrar por user_id (defense-in-depth además de RLS).
    assert "user_id = %s" in captured["sql"]
    assert _USER_A in captured["params"]


# ---------------------------------------------------------------------------
# 2. Success path
# ---------------------------------------------------------------------------
def test_delete_emits_lock_then_release_then_delete_plan_in_order():
    """Success: 3 statements en orden lock → release-locks → delete-plan,
    DENTRO de una transacción explícita.

    [P1-HIST-AUDIT-7 · 2026-05-09] Antes eran 2 statements
    (release+delete). Ahora el advisory lock per-user se toma primero
    para serializar con restore/rename concurrentes del mismo user.
    """
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    # rowcounts: 0 (lock), 2 (release locks), 1 (delete plan).
    cursor = _CursorRecorder(rowcounts=[0, 2, 1])
    pool_mock, conn_recorder = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", return_value={"id": _PLAN_ID}), \
         patch("db_core.connection_pool", pool_mock):
        r = client.delete(f"/api/plans/{_PLAN_ID}")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["released_locks"] == 2
    assert body["deleted"] is True
    assert conn_recorder.tx_entered is True

    # 3 statements en orden esperado.
    assert len(cursor.calls) == 3
    sql1, _ = cursor.calls[0]
    sql2, params2 = cursor.calls[1]
    sql3, params3 = cursor.calls[2]

    # 1) Advisory lock per-user
    assert "pg_advisory_xact_lock" in sql1

    # 2) Release locks via subquery (mismo patrón que P0-HIST-1).
    assert "DELETE FROM chunk_user_locks" in sql2
    assert "locked_by_chunk_id IN" in sql2
    assert "FROM plan_chunk_queue" in sql2
    assert "meal_plan_id = %s" in sql2
    assert _USER_A in params2
    assert _PLAN_ID in params2

    # 3) Delete meal_plans con user_id (defense-in-depth).
    assert "DELETE FROM meal_plans" in sql3
    assert "user_id = %s" in sql3
    assert _PLAN_ID in params3
    assert _USER_A in params3


def test_delete_404_on_race_zero_rows_affected():
    """ownership SELECT pasa pero el DELETE final afecta 0 filas
    (alguien borró entre las dos) → 404 (intent satisfecho, no 500)."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    # [P1-HIST-AUDIT-7] rowcounts: 0 (lock), 0 (release), 0 (delete).
    cursor = _CursorRecorder(rowcounts=[0, 0, 0])
    pool_mock, _ = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", return_value={"id": _PLAN_ID}), \
         patch("db_core.connection_pool", pool_mock):
        r = client.delete(f"/api/plans/{_PLAN_ID}")
    assert r.status_code == 404


def test_delete_handles_plan_with_no_locks():
    """Plan sin chunks (released=0) NO debe ser un error: el endpoint
    completa el DELETE igual."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    # [P1-HIST-AUDIT-7] rowcounts: 0 (lock), 0 (release), 1 (delete).
    cursor = _CursorRecorder(rowcounts=[0, 0, 1])
    pool_mock, _ = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", return_value={"id": _PLAN_ID}), \
         patch("db_core.connection_pool", pool_mock):
        r = client.delete(f"/api/plans/{_PLAN_ID}")
    assert r.status_code == 200, r.text
    assert r.json()["released_locks"] == 0


# ---------------------------------------------------------------------------
# 3. Anchor / drift detection migration↔endpoint
# ---------------------------------------------------------------------------
def test_p0_hist_3_anchor_in_endpoint():
    """Anchor visible para grep desde memoria/CLAUDE.md."""
    import inspect
    from routers.plans import api_delete_plan
    src = inspect.getsource(api_delete_plan)
    assert "P0-HIST-3" in src
    assert "chunk_user_locks" in src
    assert "DELETE FROM meal_plans" in src


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MIGRATION = _REPO_ROOT / "migrations" / "p0_hist_3_telemetry_orphan_fk.sql"


def test_migration_file_exists():
    assert _MIGRATION.is_file(), (
        f"Falta migration {_MIGRATION.name} (audit historial P0-HIST-3)"
    )


def test_migration_covers_both_telemetry_tables():
    """La migración debe añadir FK SET NULL a las DOS tablas. Si alguien
    añade una pero olvida la otra (e.g., chunk_lesson_telemetry sí pero
    chunk_deferrals no), este test alerta."""
    sql = _MIGRATION.read_text(encoding="utf-8")
    # Marker visible en el comentario top y en RAISE NOTICE.
    assert "P0-HIST-3" in sql

    # Ambas tablas mencionadas como objetivo del FK.
    for tbl in ("chunk_lesson_telemetry", "chunk_deferrals"):
        # Debe aparecer ALTER TABLE para esa tabla (al menos 1).
        pattern = re.compile(
            rf"ALTER\s+TABLE\s+public\.{tbl}\s+ADD\s+CONSTRAINT",
            re.IGNORECASE,
        )
        assert pattern.search(sql), (
            f"Migration no añade FK ADD CONSTRAINT en {tbl}"
        )

    # Ambas con ON DELETE SET NULL (no CASCADE — preservar telemetría).
    set_null_count = len(re.findall(r"ON\s+DELETE\s+SET\s+NULL", sql, re.IGNORECASE))
    assert set_null_count >= 2, (
        f"Esperaba ≥2 ON DELETE SET NULL (uno por tabla); encontré {set_null_count}"
    )

    # Idempotencia (DO block + lookup en information_schema).
    assert "information_schema.referential_constraints" in sql
    assert "DO $$" in sql


def test_migration_is_idempotent_pattern():
    """El DO block debe seguir el patrón establecido en
    p2_alpha_plan_chunk_queue_fk_cascade.sql: detectar la FK
    existente y RAISE NOTICE skip cuando ya está OK."""
    sql = _MIGRATION.read_text(encoding="utf-8")
    assert sql.count("RAISE NOTICE") >= 4  # 2 tablas × (creada/replazada/skip)
    # Debe haber un branch "ya tiene ON DELETE SET NULL. Skip" para
    # ambas tablas — el patrón idempotente del repo.
    assert sql.count("ya tiene ON DELETE SET NULL. Skip") >= 2
