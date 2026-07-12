"""[P1-HIST-5 · 2026-05-09] Tests del endpoint atómico
``PATCH /api/plans/{plan_id}/name`` para renombrado.

Bug original (audit historial 2026-05-08):
    `History.jsx::handleEditSave` hacía
    ``supabase.from('meal_plans').update({ name: trimmed })`` directo.
    Eso dejaba `plan_data.name` (jsonb top-level) con el valor viejo;
    cualquier flujo que copiara plan_data después (swap, shift_plan,
    restore pre-P0-HIST-1, serializaciones) propagaba el nombre stale
    a otro contexto.

Fix:
    Endpoint atómico que en un solo UPDATE actualiza la columna
    `name` Y `plan_data` via `jsonb_set(plan_data, '{name}',
    to_jsonb(?::text), true)`. El `create_missing=true` cubre planes
    legacy sin la key `name` en plan_data.

Cobertura:
    - 401 sin auth.
    - 400 sin name / name no-string / name vacío post-trim / name >200ch.
    - 404 cuando plan no existe / pertenece a otro user (RETURNING vacío).
    - 200 success: SQL hace UPDATE atómico con jsonb_set + name column.
    - Anchor [P1-HIST-5] presente en endpoint.
"""
import inspect
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


_USER_A = "11111111-1111-1111-1111-111111111111"
_PLAN_ID = "dddddddd-dddd-dddd-dddd-dddddddddddd"


# [P1-HIST-AUDIT-7 · 2026-05-09] Mocks para connection_pool + cursor.
# El endpoint rename ahora usa transacción explícita con advisory lock,
# así que los tests interceptan el cursor en lugar de execute_sql_write.
class _CursorRecorder:
    def __init__(self, fetchall_returns=None):
        self.calls = []
        self._fetchall_returns = list(fetchall_returns or [])

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

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

    def cursor(self, *a, **kw):
        return self.cursor_obj

    def transaction(self):
        class _Tx:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, exc_type, *a):
                return False
        return _Tx()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _build_pool_mock(cursor):
    from unittest.mock import MagicMock
    pool = MagicMock()
    pool.connection.return_value = _ConnRecorder(cursor)
    return pool


# ---------------------------------------------------------------------------
# 1. Auth / validación
# ---------------------------------------------------------------------------
def test_rename_requires_auth():
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: None

    client.app.dependency_overrides[get_verified_user_id] = lambda: None

    r = client.patch(f"/api/plans/{_PLAN_ID}/name", json={"name": "Nuevo"})
    assert r.status_code == 401, r.text


def test_rename_requires_name_field():
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    # Body vacío.
    r = client.patch(f"/api/plans/{_PLAN_ID}/name", json={})
    assert r.status_code == 400


def test_rename_rejects_non_string_name():
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    for bad in [123, None, [], {"x": 1}, True]:
        r = client.patch(f"/api/plans/{_PLAN_ID}/name", json={"name": bad})
        assert r.status_code == 400, f"non-string name {bad!r} debió rechazar; got {r.status_code}"


def test_rename_rejects_empty_post_trim():
    """Strings whitespace-only se rechazan (400). Sin esto, un usuario
    podría dejar el nombre como espacios y el frontend mostraría el
    campo vacío sin tooltip de ayuda."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    for empty in ["", "   ", "\t\n", "  \n  \t  "]:
        r = client.patch(f"/api/plans/{_PLAN_ID}/name", json={"name": empty})
        assert r.status_code == 400, f"name empty post-trim {empty!r}: {r.status_code}"


def test_rename_caps_length():
    """Cap defensivo a 200 chars. Un nombre de 10K caracteres es bug
    cliente, no caso legítimo."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    too_long = "X" * 201
    r = client.patch(f"/api/plans/{_PLAN_ID}/name", json={"name": too_long})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 2. Success / not found
# ---------------------------------------------------------------------------
def test_rename_404_when_plan_missing_or_other_user():
    """RETURNING vacío → 404 (no leak de existencia)."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    cursor = _CursorRecorder(fetchall_returns=[[]])  # RETURNING vacío
    pool_mock = _build_pool_mock(cursor)
    with patch("db_core.connection_pool", pool_mock):
        r = client.patch(
            f"/api/plans/{_PLAN_ID}/name",
            json={"name": "Plan Renombrado"},
        )
    assert r.status_code == 404


def test_rename_emits_atomic_update_with_jsonb_set():
    """Success path: el UPDATE debe actualizar `name` Y plan_data
    via jsonb_set en el mismo statement.

    [P1-HIST-AUDIT-7 · 2026-05-09] El endpoint ahora hace 2 statements
    en transacción: lock advisory + UPDATE...RETURNING.
    """
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    cursor = _CursorRecorder(fetchall_returns=[[{"id": _PLAN_ID}]])
    pool_mock = _build_pool_mock(cursor)
    with patch("db_core.connection_pool", pool_mock):
        r = client.patch(
            f"/api/plans/{_PLAN_ID}/name",
            json={"name": "  Plan con espacios  "},
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    # Trim aplicado server-side.
    assert body["name"] == "Plan con espacios"

    # lock + SELECT-latest (P2-HIST-RENAME-NO-PROMOTE) + UPDATE. El recorder no
    # implementa fetchone → el check cae al fail-open legacy (sella), que es la
    # rama que este test cubre.
    assert 2 <= len(cursor.calls) <= 3
    assert "pg_advisory_xact_lock" in cursor.calls[0][0]

    # [P2-HIST-RENAME-NO-PROMOTE] SELECT de latest entre lock y UPDATE → por contenido.
    _upd = [c for c in cursor.calls if "UPDATE meal_plans" in c[0]]
    assert _upd, f"sin UPDATE en calls: {[c[0][:40] for c in cursor.calls]}"
    sql, params = _upd[0]
    # Atomic: ambos en el mismo UPDATE.
    assert "UPDATE meal_plans" in sql
    assert "name = %s" in sql
    assert "jsonb_set(" in sql
    assert "'{name}'" in sql
    assert "to_jsonb(%s::text)" in sql
    # create_missing=true: planes legacy sin `name` en plan_data
    # también lo ganan.
    assert "true" in sql
    # Defense-in-depth: WHERE incluye user_id (RLS también, pero
    # explícito previene leak si RLS falla).
    assert "user_id = %s" in sql
    # RETURNING para confirmar match.
    assert "RETURNING id" in sql

    # [P1-HIST-AUDIT-2 · 2026-05-09] Params en orden:
    # name (col), name (jsonb_set name), modified_at (jsonb_set ts),
    # plan_id, user_id.
    assert len(params) == 5, (
        f"Se esperaban 5 params (col + 2× jsonb_set + plan_id + user_id), "
        f"got {len(params)}: {params}"
    )
    assert params[0] == "Plan con espacios"
    assert params[1] == "Plan con espacios"
    assert isinstance(params[2], str) and "T" in params[2]
    assert params[3] == _PLAN_ID
    assert params[4] == _USER_A


def test_rename_trims_name_in_response():
    """Backend devuelve el nombre POST-trim, frontend lo usa para mirror
    del state. Sin trim, el state local guardaría espacios y next read
    desde DB devolvería trimmed → drift en re-renders."""
    client = _build_test_client()
    from auth import verify_api_quota, get_verified_user_id
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    cursor = _CursorRecorder(fetchall_returns=[[{"id": _PLAN_ID}]])
    pool_mock = _build_pool_mock(cursor)
    with patch("db_core.connection_pool", pool_mock):
        r = client.patch(
            f"/api/plans/{_PLAN_ID}/name",
            json={"name": "\t  Mi Plan  \n"},
        )
    assert r.status_code == 200
    assert r.json()["name"] == "Mi Plan"


# ---------------------------------------------------------------------------
# 3. Anchor / drift detection
# ---------------------------------------------------------------------------
def test_p1_hist_5_anchor_in_endpoint():
    from routers.plans import api_rename_plan
    src = inspect.getsource(api_rename_plan)
    assert "P1-HIST-5" in src
    # Contrato clave: jsonb_set en el endpoint. Si alguien revierte a
    # solo `UPDATE meal_plans SET name=?` el drift vuelve.
    assert "jsonb_set" in src
    assert "to_jsonb" in src
    # create_missing=true es load-bearing para planes legacy.
    assert "true" in src
