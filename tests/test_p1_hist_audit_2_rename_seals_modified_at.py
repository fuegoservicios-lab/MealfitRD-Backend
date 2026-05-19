"""[P1-HIST-AUDIT-2 · 2026-05-09] Tests: ``api_rename_plan`` debe sellar
``_plan_modified_at`` para mantener SSOT con el sort del Historial.

Bug original (audit historial 2026-05-08):
    Post-P1-HIST-4, el frontend ordena el listado del Historial por
    ``max(created_at, _plan_modified_at)``
    (`History.jsx::_effectiveModifiedAt`). Post-P1-HIST-AUDIT-1, el
    SELECT target del restore replica esa lógica en SQL. Pero
    ``api_rename_plan`` solo actualizaba la columna ``name`` y la
    key ``plan_data->>name`` — NO sellaba ``_plan_modified_at``.

    Consecuencia: al renombrar un plan, la card optimistic-updated
    aparece con el nuevo nombre Y "primera" (asumida modificada),
    pero al próximo ``fetchHistory`` el sort la baja porque
    ``_plan_modified_at`` no fue actualizado en DB. Drift visible
    entre optimistic update y refetch — friction UX y inconsistencia
    con los otros ~6 paths del backend que sí sellan en cada
    mutación de plan_data (cron_tasks.py:284-377, :13337, :16636,
    :16742, :20278, :20759 y routers/plans.py:3661).

Fix:
    El UPDATE usa ``jsonb_set`` ANIDADO: la capa interior actualiza
    ``name``, la exterior actualiza ``_plan_modified_at`` con
    ``datetime.now(timezone.utc).isoformat()``. Ambas con
    ``create_missing=true``.

Cobertura:
    - Anchor del marker en el endpoint.
    - SQL contiene ``jsonb_set`` ANIDADO (doble open-paren).
    - SQL referencia path ``{_plan_modified_at}``.
    - Param 3 es timestamp ISO parseable como UTC.
    - El timestamp emitido está dentro de una ventana razonable
      (< 5s antes de la llamada).
    - El response NO expone el timestamp (privado, solo persistido).
"""
from __future__ import annotations

import inspect
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "11111111-1111-1111-1111-111111111111"
_PLAN_ID = "dddddddd-dddd-dddd-dddd-dddddddddddd"


# [P1-HIST-AUDIT-7 · 2026-05-09] El endpoint rename ahora usa
# transacción explícita con advisory lock; los tests interceptan el
# cursor en lugar de execute_sql_write.
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


def _client_with_auth():
    from auth import verify_api_quota, get_verified_user_id
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER
    return client


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_api_rename_plan():
    from routers.plans import api_rename_plan
    src = inspect.getsource(api_rename_plan)
    assert "P1-HIST-AUDIT-2" in src, (
        "api_rename_plan debe mencionar el marker P1-HIST-AUDIT-2 para "
        "que un grep desde memoria/CLAUDE.md encuentre el cierre."
    )


# ---------------------------------------------------------------------------
# 2. Contrato SQL: jsonb_set anidado + path _plan_modified_at
# ---------------------------------------------------------------------------
def test_update_uses_nested_jsonb_set_for_modified_at():
    from routers.plans import api_rename_plan
    src = inspect.getsource(api_rename_plan)

    # Aislar el UPDATE (entre execute_sql_write( y la cláusula
    # RETURNING). Esto descarta menciones en docstrings/comentarios.
    update_match = re.search(
        r"UPDATE\s+meal_plans[\s\S]*?RETURNING\s+id",
        src,
        re.IGNORECASE,
    )
    assert update_match is not None, (
        "No se encontró el UPDATE atómico con `RETURNING id` en "
        "api_rename_plan. ¿Fue refactorizado?"
    )
    update_sql = update_match.group(0)

    # jsonb_set anidado: dos open-parens consecutivos.
    nested_calls = re.findall(r"jsonb_set\s*\(", update_sql, re.IGNORECASE)
    assert len(nested_calls) >= 2, (
        f"Se esperaban ≥2 llamadas a `jsonb_set` (anidado: name + "
        f"_plan_modified_at), encontradas {len(nested_calls)}. Sin el "
        f"sello, el plan renombrado no sube en el listado tras refetch."
    )

    assert "'{_plan_modified_at}'" in update_sql, (
        "UPDATE no referencia el path `'{_plan_modified_at}'`. P1-HIST-AUDIT-2 "
        "requiere sellar ese key en el mismo statement."
    )


# ---------------------------------------------------------------------------
# 3. Comportamiento: el timestamp emitido es ISO + dentro de ventana
# ---------------------------------------------------------------------------
def test_emitted_timestamp_is_recent_iso_utc():
    """El 3er param del UPDATE debe ser un ISO timestamptz cercano al
    momento de la llamada (< 5s atrás). Esto demuestra que el endpoint
    usa `datetime.now(timezone.utc).isoformat()` en vivo, no un literal
    estático que se quedaría stale.

    [P1-HIST-AUDIT-7] El cursor ejecuta 2 statements: lock + UPDATE.
    Capturamos los params del UPDATE (índice 1).
    """
    client = _client_with_auth()

    cursor = _CursorRecorder(fetchall_returns=[[{"id": _PLAN_ID}]])
    pool_mock = _build_pool_mock(cursor)

    before = datetime.now(timezone.utc)
    with patch("db_core.connection_pool", pool_mock):
        r = client.patch(
            f"/api/plans/{_PLAN_ID}/name",
            json={"name": "Plan Renombrado"},
        )
    after = datetime.now(timezone.utc)

    assert r.status_code == 200, r.text

    # cursor.calls[0] es lock, cursor.calls[1] es UPDATE.
    assert "pg_advisory_xact_lock" in cursor.calls[0][0]
    _, params = cursor.calls[1]
    assert len(params) == 5, (
        f"Se esperaban 5 params, got {len(params)}: {params}"
    )
    iso_str = params[2]
    assert isinstance(iso_str, str), f"Param 3 no es string: {iso_str!r}"

    # Parse y verificación de ventana.
    try:
        ts = datetime.fromisoformat(iso_str)
    except ValueError as e:
        pytest.fail(
            f"Timestamp emitido no parsea como ISO: {iso_str!r}: {e}"
        )

    # tz-aware.
    assert ts.tzinfo is not None, (
        f"Timestamp debe ser tz-aware (UTC), got naive: {iso_str!r}"
    )
    # Dentro de la ventana de la llamada (con margen).
    assert before - timedelta(seconds=5) <= ts <= after + timedelta(seconds=5), (
        f"Timestamp {ts.isoformat()} fuera de la ventana de la llamada "
        f"({before.isoformat()} .. {after.isoformat()}). ¿Hardcodeado?"
    )


# ---------------------------------------------------------------------------
# 4. Response NO expone el timestamp (privado interno)
# ---------------------------------------------------------------------------
def test_response_does_not_leak_modified_at():
    """El timestamp es metadata interna. El response público sigue
    siendo `{success, name}` — sin `modified_at` ni similares.
    Cualquier consumidor del response no debería depender de ese valor.
    """
    client = _client_with_auth()
    cursor = _CursorRecorder(fetchall_returns=[[{"id": _PLAN_ID}]])
    pool_mock = _build_pool_mock(cursor)
    with patch("db_core.connection_pool", pool_mock):
        r = client.patch(
            f"/api/plans/{_PLAN_ID}/name",
            json={"name": "X"},
        )
    assert r.status_code == 200
    body = r.json()
    # Contrato del response: solo success + name.
    assert set(body.keys()) == {"success", "name"}, (
        f"Response shape cambió post P1-HIST-AUDIT-2: {body}. El "
        f"timestamp es interno y no debe filtrarse al cliente."
    )


# ---------------------------------------------------------------------------
# 5. Justificación textual: el comentario debe explicar POR QUÉ
# ---------------------------------------------------------------------------
def test_comment_references_history_sort_or_effective_modified_at():
    """El comentario inline debe explicar la motivación (SSOT con el
    sort del Historial / `_effectiveModifiedAt`). Sin esto, un refactor
    cosmético podría borrar el sello sin entender que es load-bearing.
    """
    from routers.plans import api_rename_plan
    src = inspect.getsource(api_rename_plan)
    anchors = ("_effectiveModifiedAt", "_plan_modified_at", "P1-HIST-AUDIT-1")
    assert any(a in src for a in anchors), (
        f"Comentario sobre sello `_plan_modified_at` no menciona ninguno "
        f"de {anchors}. Sin anchor, el load-bearing del sello es invisible."
    )
