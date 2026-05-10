"""[P2-HIST-AUDIT-5 · 2026-05-09] Tests: ``api_restore_plan`` cancela
chunks pending del SOURCE además del TARGET, y el release de locks
cubre AMBOS plans.

Bug original (audit historial 2026-05-08):
    `api_restore_plan` cancelaba pending/processing del TARGET pero
    NO del SOURCE. Si el usuario reactivaba un plan archivado con
    workers vivos, esos seguían corriendo y escribían al row source
    tras el copy → row source con plan_data modificado post-archivo
    (debería ser snapshot inmutable). UX confusa al hacer
    post-mortem y consumo desperdiciado de slots LLM.

Fix:
    - Segundo `UPDATE plan_chunk_queue` con
      `dead_letter_reason='restore_source_archived'` y filtro
      `meal_plan_id = source_plan_id`.
    - Subquery del DELETE de chunk_user_locks extendida con
      `meal_plan_id IN (target, source)` para cubrir locks de
      cualquiera de los dos planes.
    - Response expone `cancelled_source_chunks` separately.

Cobertura:
    - Anchor del marker.
    - 4º statement es UPDATE plan_chunk_queue con
      `restore_source_archived` + source_plan_id.
    - Razón distinta `restore_source_archived` separa audit trail
      del `restore_overwrite` del target.
    - Release locks subquery menciona ambos plan_ids.
    - Response shape: nuevo field `cancelled_source_chunks`.
    - Logger info incluye cancelled_source_chunks para diagnóstico.
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "11111111-1111-1111-1111-111111111111"
_SOURCE_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_TARGET_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

_SOURCE_ROW = {
    "id": _SOURCE_ID,
    "plan_data": {"name": "Plan A", "days": [{"meals": [{"name": "X"}]}]},
    "name": "Plan A",
    "calories": 2000,
    "macros": {},
    "meal_names": [],
    "ingredients": [],
    "techniques": [],
}


# ---------------------------------------------------------------------------
# Mocks (mismos que test_p0_hist_1)
# ---------------------------------------------------------------------------
class _CursorRecorder:
    def __init__(self, rowcounts, fetchone_returns=None):
        self.calls = []
        self._rowcounts = list(rowcounts)
        self._fetchone_returns = list(fetchone_returns or [])
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        self.rowcount = self._rowcounts.pop(0) if self._rowcounts else 0

    def fetchone(self):
        return self._fetchone_returns.pop(0) if self._fetchone_returns else None

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
    pool = MagicMock()
    pool.connection.return_value = _ConnRecorder(cursor)
    return pool


def _client():
    from auth import verify_api_quota
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER
    return client


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_endpoint():
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    assert "P2-HIST-AUDIT-5" in src


# ---------------------------------------------------------------------------
# 2. Cancel del SOURCE: 4º statement con razón distinta
# ---------------------------------------------------------------------------
def test_cancel_source_statement_emitted_with_correct_reason():
    """El 4º statement (post target SELECT, post cancel target) es un
    UPDATE plan_chunk_queue con `dead_letter_reason='restore_source_archived'`
    y `meal_plan_id = source_plan_id`."""
    client = _client()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 2, 5, 1, 1],  # cancel_source = 5 chunks
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cancelled_source_chunks"] == 5

    assert len(cursor.calls) == 6
    sql4, params4 = cursor.calls[3]
    assert "UPDATE plan_chunk_queue" in sql4
    assert "status = 'cancelled'" in sql4
    assert "'pending'" in sql4 and "'processing'" in sql4
    # Razón distinta del cancel del target.
    assert "restore_source_archived" in params4
    assert _SOURCE_ID in params4
    # NO debe tener target_id en el cancel del source.
    assert _TARGET_ID not in params4


def test_cancel_source_uses_distinct_reason_from_target():
    """Razones distintas (`restore_overwrite` vs
    `restore_source_archived`) separan el audit trail. Si SRE agrupa
    por dead_letter_reason, puede medir cuántos chunks se cancelan
    por reactivación de un plan archivado vs sobreescritura del
    activo."""
    client = _client()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 1, 1, 0, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200

    _, params_target = cursor.calls[2]
    _, params_source = cursor.calls[3]
    # Razones distintas.
    assert "restore_overwrite" in params_target
    assert "restore_overwrite" not in params_source
    assert "restore_source_archived" in params_source
    assert "restore_source_archived" not in params_target


# ---------------------------------------------------------------------------
# 3. Release locks cubre AMBOS plans
# ---------------------------------------------------------------------------
def test_release_locks_subquery_covers_target_and_source():
    """El DELETE FROM chunk_user_locks debe filtrar por
    `meal_plan_id IN (target, source)` — sino, un lock vivo del
    source quedaría colgado tras el cancel.
    """
    client = _client()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 0, 0, 1, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200

    sql5, params5 = cursor.calls[4]
    assert "DELETE FROM chunk_user_locks" in sql5
    # Subquery con IN (target, source).
    assert re.search(
        r"meal_plan_id\s+IN\s*\(\s*%s\s*,\s*%s\s*\)",
        sql5,
        re.IGNORECASE,
    ), (
        f"Subquery del release locks debe usar `meal_plan_id IN (%s, %s)`. "
        f"Got: {sql5!r}"
    )
    # Ambos plan_ids en los params.
    assert _USER in params5
    assert _TARGET_ID in params5
    assert _SOURCE_ID in params5


# ---------------------------------------------------------------------------
# 4. Response shape incluye cancelled_source_chunks
# ---------------------------------------------------------------------------
def test_response_includes_cancelled_source_chunks_field():
    """El field `cancelled_source_chunks` es parte del contrato
    público del response — clientes (frontend, monitoring) pueden
    consumirlo. Verificamos su presencia siempre, no solo cuando
    > 0."""
    client = _client()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 0, 0, 0, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200
    body = r.json()
    assert "cancelled_source_chunks" in body
    assert isinstance(body["cancelled_source_chunks"], int)


def test_noop_response_includes_zero_cancelled_source_chunks():
    """En el caso noop (target == source), no se cancela nada (no
    se entra al else block). El field debe seguir siendo 0, no
    missing — frontends que asumen el shape completo no rompen."""
    client = _client()
    cursor = _CursorRecorder(
        rowcounts=[0, 0],
        fetchone_returns=[{"id": _SOURCE_ID}],  # target == source → noop
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200
    body = r.json()
    assert body["noop"] is True
    assert body["cancelled_chunks"] == 0
    assert body["cancelled_source_chunks"] == 0


# ---------------------------------------------------------------------------
# 5. Logger info incluye cancelled_source_chunks
# ---------------------------------------------------------------------------
def test_logger_info_format_includes_cancelled_source():
    """El logger.info al final del success path debe incluir
    `cancelled_source_chunks=%d` para que SRE pueda diagnosticar
    via grep en producción."""
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    # El logger.info debe mencionar cancelled_source_chunks.
    assert "cancelled_source_chunks" in src
    assert re.search(
        r"cancelled_source_chunks\s*=\s*%d",
        src,
    ), (
        "logger.info debe formatear `cancelled_source_chunks=%d` "
        "para que el log sea grep-able. Sin esto, el field del response "
        "es invisible en producción."
    )


# ---------------------------------------------------------------------------
# 6. Comentario load-bearing: explica POR QUÉ cancelamos source
# ---------------------------------------------------------------------------
def test_comment_explains_source_cancel_motivation():
    """El comentario inline debe explicar la motivación (workers
    vivos del source escribirían al row archivado tras restore).
    Sin esto, un refactor cosmético podría borrar el statement sin
    entender que es load-bearing.
    """
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    # Buscamos referencias clave en los comentarios cerca del cancel
    # del source.
    cancel_idx = src.find("restore_source_archived")
    assert cancel_idx > -1
    block = src[max(0, cancel_idx - 1500):cancel_idx]
    # Debe explicar al menos uno de los conceptos: snapshot inmutable,
    # workers vivos, o post-archivo.
    motivations = ("snapshot", "workers", "post-archivo", "archivado", "inmutable")
    assert any(m.lower() in block.lower() for m in motivations), (
        f"El comentario sobre el cancel del source debe explicar la "
        f"motivación. Esperado uno de {motivations}; got block sin match."
    )
