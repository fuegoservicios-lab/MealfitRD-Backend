"""[P0-HIST-1 · 2026-05-09] Tests del endpoint atómico
``POST /api/plans/restore``.

Bug original (audit historial 2026-05-08):
    `AssessmentContext.restorePlan` (frontend) hacía
    ``UPDATE meal_plans SET plan_data = ? WHERE id = <latest>``
    via supabase client. Si la fila latest tenía chunks pending/
    processing en `plan_chunk_queue`, los workers seguían
    procesando con el `pipeline_snapshot` del plan anterior y
    luego mergeaban días generados al estilo previo dentro del
    plan_data recién sobrescrito → contaminación silenciosa del
    plan restaurado. Adicionalmente NO actualizaba columnas
    top-level (name/calories/macros/meal_names/ingredients/
    techniques) → el header del Dashboard mostraba metadata stale.

Fix:
    Endpoint backend que hace todo en una sola transacción:
      1. Cancela chunks pending/processing del target con
         dead_letter_reason='restore_overwrite'.
      2. Libera chunk_user_locks asociados a chunks del target.
      3. Sobrescribe plan_data + 6 columnas top-level.
      4. Anota _plan_modified_at + _restored_from_plan_id en
         plan_data.

Cobertura:
    - 401 sin auth.
    - 400 sin source_plan_id.
    - 404 source no existe / no pertenece al usuario.
    - 409 usuario sin planes activos (sin target).
    - 200 noop cuando target.id == source_plan_id.
    - 200 success path: emite los 3 statements esperados, en orden,
      con los parámetros correctos; retorna counts de chunks
      cancelados y locks liberados.
    - Top-level columns (name/calories/macros/meal_names/
      ingredients/techniques) viajan al UPDATE.
    - plan_data enriquecido con _plan_modified_at y
      _restored_from_plan_id.
"""
import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_test_client():
    """TestClient con el router de plans incluido (sin prefix duplicado)."""
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _override_auth(monkeypatch, user_id):
    """Override `verify_api_quota` para devolver el user_id fijo."""
    from auth import verify_api_quota, get_verified_user_id
    from routers.plans import router as plans_router

    def _fake(*args, **kwargs):
        return user_id

    # Override en el TestClient app: aplicamos a través de
    # `dependency_overrides` del FastAPI en cada test (no global).
    return _fake


class _CursorRecorder:
    """Cursor mock que graba (sql, params) por cada execute().

    [P1-HIST-AUDIT-7 · 2026-05-09] Soporta también `fetchone` (para el
    target SELECT que ahora vive dentro de la transacción) y `fetchall`
    (para RETURNING en rename/update).
    """
    def __init__(self, rowcounts, fetchone_returns=None, fetchall_returns=None):
        self.calls = []
        self._rowcounts = list(rowcounts)
        self._fetchone_returns = list(fetchone_returns or [])
        self._fetchall_returns = list(fetchall_returns or [])
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        # Cada execute consume el siguiente rowcount esperado.
        if self._rowcounts:
            self.rowcount = self._rowcounts.pop(0)
        else:
            self.rowcount = 0

    def fetchone(self):
        if self._fetchone_returns:
            return self._fetchone_returns.pop(0)
        return None

    def fetchall(self):
        if self._fetchall_returns:
            return self._fetchall_returns.pop(0)
        return []

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
            def __exit__(self_inner, exc_type, *a):
                outer.tx_exited = True
                # Re-raise cualquier excepción del bloque.
                return False
        return _Tx()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _build_pool_mock(cursor):
    """`connection_pool.connection()` → context manager con `_ConnRecorder`."""
    conn = _ConnRecorder(cursor)
    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, conn


# ---------------------------------------------------------------------------
# Test fixtures: source row stub
# ---------------------------------------------------------------------------
_USER_A = "11111111-1111-1111-1111-111111111111"
_USER_B = "22222222-2222-2222-2222-222222222222"
_SOURCE_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_TARGET_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

_SOURCE_ROW = {
    "id": _SOURCE_ID,
    "plan_data": {
        "name": "Plan Histórico A",
        "days": [{"meals": [{"name": "Mangú"}]}],
        "totalDays": 7,
    },
    "name": "Plan Histórico A",
    "calories": 2200,
    "macros": {"protein": 140, "carbs": 250, "fats": 60},
    "meal_names": ["Mangú", "Pollo guisado"],
    "ingredients": ["plátano", "pollo"],
    "techniques": ["hervido", "estofado"],
}


# ---------------------------------------------------------------------------
# 1. Auth / input validation
# ---------------------------------------------------------------------------
def test_restore_requires_auth():
    """Sin auth (verify_api_quota → None) → 401."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: None

    client.app.dependency_overrides[get_verified_user_id] = lambda: None

    r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 401, r.text


def test_restore_requires_source_plan_id():
    """Sin `source_plan_id` en body → 400."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    # Body vacío.
    r = client.post("/api/plans/restore", json={})
    assert r.status_code == 400

    # Tipo no-string.
    r = client.post("/api/plans/restore", json={"source_plan_id": 123})
    assert r.status_code == 400


def test_restore_404_when_source_missing():
    """SELECT del source devuelve None → 404."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", return_value=None):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 404


def test_restore_409_when_no_target():
    """Source existe pero usuario sin planes (target=None) → 409.

    [P1-HIST-AUDIT-7 · 2026-05-09] El target SELECT ahora vive DENTRO
    de la transacción (después del advisory lock). El cursor retorna
    None en fetchone → 409 raised, transacción rollback.
    """
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    # source viene de execute_sql_query (fuera de tx), target de cursor.fetchone().
    cursor = _CursorRecorder(rowcounts=[0, 0], fetchone_returns=[None])
    pool_mock, _ = _build_pool_mock(cursor)
    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 409


def test_restore_noop_when_source_equals_target():
    """target.id == source_plan_id → no-op idempotente.

    [P1-HIST-AUDIT-7 · 2026-05-09] Ahora la transacción SÍ se abre
    (porque el lock + target SELECT viven dentro). En noop, solo
    se ejecutan 2 statements: el advisory_lock y el target SELECT.
    NO se hace cancel/release/update.
    """
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    cursor = _CursorRecorder(rowcounts=[0, 0], fetchone_returns=[{"id": _SOURCE_ID}])
    pool_mock, conn_recorder = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["noop"] is True
    assert body["cancelled_chunks"] == 0
    assert body["released_locks"] == 0
    # Transacción SÍ se abre (lock vive dentro), pero solo 2
    # statements en cursor: advisory_lock + target SELECT.
    assert conn_recorder.tx_entered is True
    assert len(cursor.calls) == 2
    assert "pg_advisory_xact_lock" in cursor.calls[0][0]
    assert "SELECT id" in cursor.calls[1][0]


# ---------------------------------------------------------------------------
# 2. Success path: contrato de los 3 statements
# ---------------------------------------------------------------------------
def test_restore_emits_six_statements_in_order():
    """Success path: 6 SQL emitidos en orden:
        lock → target SELECT → cancel target → cancel source →
        release locks → UPDATE meal_plans.

    [P1-HIST-AUDIT-7 · 2026-05-09] Antes eran 3 (cancel+release+update).
    Ahora el lock y el target SELECT viven dentro de la transacción.
    [P2-HIST-AUDIT-5 · 2026-05-09] Ahora también cancela chunks
    pending del SOURCE (`restore_source_archived`) y el release de
    locks cubre AMBOS plans.
    """
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    # rowcounts: lock, target SELECT, cancel target=2, cancel source=3,
    # release locks=1, update=1.
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 2, 3, 1, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock, conn_recorder = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["noop"] is False
    assert body["cancelled_chunks"] == 2
    assert body["cancelled_source_chunks"] == 3
    assert body["released_locks"] == 1
    assert body["target_plan_id"] == _TARGET_ID
    assert body["source_plan_id"] == _SOURCE_ID

    # 6 statements en orden esperado.
    assert len(cursor.calls) == 6
    sql1, _ = cursor.calls[0]
    sql2, _ = cursor.calls[1]
    sql3, params3 = cursor.calls[2]
    sql4, params4 = cursor.calls[3]
    sql5, params5 = cursor.calls[4]
    sql6, params6 = cursor.calls[5]

    # 1) Advisory lock per-user
    assert "pg_advisory_xact_lock" in sql1
    # 2) Target SELECT
    assert "SELECT id" in sql2
    assert "FROM meal_plans" in sql2
    assert "GREATEST" in sql2

    # 3) Cancel target chunks
    assert "UPDATE plan_chunk_queue" in sql3
    assert "status = 'cancelled'" in sql3
    assert "'pending'" in sql3 and "'processing'" in sql3
    assert "restore_overwrite" in params3
    assert _TARGET_ID in params3

    # 4) Cancel source chunks (P2-HIST-AUDIT-5)
    assert "UPDATE plan_chunk_queue" in sql4
    assert "restore_source_archived" in params4
    assert _SOURCE_ID in params4

    # 5) Release locks (cubre target Y source)
    assert "DELETE FROM chunk_user_locks" in sql5
    assert "locked_by_chunk_id" in sql5
    assert _USER_A in params5
    assert _TARGET_ID in params5
    assert _SOURCE_ID in params5

    # 6) Update meal_plans (top-level + plan_data)
    assert "UPDATE meal_plans" in sql6
    assert "plan_data = %s::jsonb" in sql6
    assert "name = %s" in sql6
    assert "calories = %s" in sql6
    assert "macros = %s::jsonb" in sql6
    assert "meal_names = %s" in sql6
    assert "ingredients = %s" in sql6
    assert "techniques = %s" in sql6


def test_restore_top_level_columns_match_source():
    """Las 6 columnas top-level del UPDATE deben venir del source row.
    Cierra P0-HIST-2 acoplado: el UPDATE no debe quedarse solo con
    `plan_data` (legacy bug).

    [P1-HIST-AUDIT-7] El UPDATE es el 5º statement (lock+target+cancel
    +release+update).
    """
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    cursor = _CursorRecorder(
        rowcounts=[0, 0, 0, 0, 0, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock, _ = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200, r.text

    # [P2-HIST-AUDIT-5 · 2026-05-09] UPDATE meal_plans ahora es el
    # 6º statement (lock + target SELECT + cancel target + cancel
    # source + release locks + UPDATE).
    _, params6 = cursor.calls[5]
    # params6 es la tupla del UPDATE: (plan_data_json, name, calories,
    # macros_json, meal_names, ingredients, techniques, target_id, user_id)
    plan_data_json = params6[0]
    name = params6[1]
    calories = params6[2]
    macros_json = params6[3]
    meal_names = params6[4]
    ingredients = params6[5]
    techniques = params6[6]
    target_id = params6[7]
    user_id_param = params6[8]

    assert name == "Plan Histórico A"
    assert calories == 2200
    assert json.loads(macros_json) == {"protein": 140, "carbs": 250, "fats": 60}
    assert meal_names == ["Mangú", "Pollo guisado"]
    assert ingredients == ["plátano", "pollo"]
    assert techniques == ["hervido", "estofado"]
    assert target_id == _TARGET_ID
    assert user_id_param == _USER_A

    # plan_data debe traer las claves enriquecidas.
    pd = json.loads(plan_data_json)
    assert pd["_restored_from_plan_id"] == _SOURCE_ID
    assert "_plan_modified_at" in pd
    # Y preservar los campos del source.
    assert pd["totalDays"] == 7
    assert pd["days"][0]["meals"][0]["name"] == "Mangú"


def test_restore_opens_explicit_transaction():
    """El UPDATE de las 5 sentencias debe ocurrir DENTRO de
    `conn.transaction()` para que un fallo en cualquiera revierta las
    anteriores. [P1-HIST-AUDIT-7] el lock y el target SELECT también
    viven dentro de la transacción ahora."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    cursor = _CursorRecorder(
        rowcounts=[0, 0, 0, 0, 0, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock, conn_recorder = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200, r.text

    assert conn_recorder.tx_entered is True
    assert conn_recorder.tx_exited is True


def test_restore_handles_empty_top_level_arrays():
    """Si el source tiene meal_names/ingredients/techniques NULL (filas
    legacy), el endpoint los normaliza a [] sin romper."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    src = dict(_SOURCE_ROW)
    src["meal_names"] = None
    src["ingredients"] = None
    src["techniques"] = None
    src["macros"] = None

    cursor = _CursorRecorder(
        rowcounts=[0, 0, 0, 0, 0, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock, _ = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[src]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200, r.text

    # [P2-HIST-AUDIT-5] UPDATE meal_plans es el 6º statement.
    _, params6 = cursor.calls[5]
    assert json.loads(params6[3]) == {}  # macros normalizado
    assert params6[4] == []  # meal_names
    assert params6[5] == []  # ingredients
    assert params6[6] == []  # techniques


# ---------------------------------------------------------------------------
# 3. Documentación / contrato visible al equipo
# ---------------------------------------------------------------------------
def test_p0_hist_1_marker_present_in_endpoint():
    """El endpoint debe tener el marker [P0-HIST-1 · 2026-05-09] en
    su docstring o cuerpo, para que un grep desde memoria/CLAUDE.md
    encuentre el cierre."""
    import inspect
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    assert "P0-HIST-1" in src
    assert "restore_overwrite" in src
    assert "_plan_modified_at" in src
    assert "_restored_from_plan_id" in src
