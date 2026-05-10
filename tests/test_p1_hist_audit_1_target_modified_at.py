"""[P1-HIST-AUDIT-1 · 2026-05-09] Tests del target del endpoint
``POST /api/plans/restore``: debe usar ``GREATEST(created_at,
_plan_modified_at)`` para mantener SSOT con el helper del frontend
``History.jsx::_effectiveModifiedAt``.

Bug original (audit historial 2026-05-08):
    El SELECT target ordenaba solo por ``created_at DESC``. Si el
    usuario ya había restaurado un plan archivado B (con
    ``created_at`` anterior al plan A), B quedaba con
    ``_plan_modified_at`` sellado en plan_data. La UI del Historial
    (que ordena por ``max(created_at, _plan_modified_at)``) mostraba
    B "primero" como activo, pero el backend seguía tratando A como
    target. Próximo restore sobrescribía A en lugar del activo
    visible → drift entre activo-UI y activo-backend.

Fix:
    El SELECT target del endpoint replica el cálculo del helper JS
    en SQL: ``GREATEST(created_at, COALESCE((plan_data->>
    '_plan_modified_at')::timestamptz, created_at))``. Tie-breaker
    secundario por ``created_at DESC`` para determinismo.

Cobertura:
    - Anchor textual del marker en el endpoint.
    - Contrato SQL del SELECT target: tokens `GREATEST`,
      `_plan_modified_at`, `COALESCE`, tie-breaker `created_at DESC`.
    - Comportamiento: mockeando el SELECT target con un id distinto
      al "primero por created_at", el endpoint procede con ESE id
      como target_plan_id en el response.
    - Paridad de intención con el helper JS: para un set de pares
      (created_at, _plan_modified_at), el helper Python equivalente
      coincide con la lógica del frontend.
"""
from __future__ import annotations

import inspect
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers compartidos con test_p0_hist_1_restore_plan_atomic
# ---------------------------------------------------------------------------
_USER = "11111111-1111-1111-1111-111111111111"
_SOURCE_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_TARGET_BY_CREATED = "cccccccc-cccc-cccc-cccc-cccccccccccc"  # newest created_at
_TARGET_BY_MODIFIED = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"  # newest _plan_modified_at

_SOURCE_ROW = {
    "id": _SOURCE_ID,
    "plan_data": {
        "name": "Plan Histórico",
        "days": [{"meals": [{"name": "Mangú"}]}],
    },
    "name": "Plan Histórico",
    "calories": 2200,
    "macros": {"protein": 140, "carbs": 250, "fats": 60},
    "meal_names": ["Mangú"],
    "ingredients": ["plátano"],
    "techniques": ["hervido"],
}


class _CursorRecorder:
    def __init__(self, rowcounts, fetchone_returns=None, fetchall_returns=None):
        self.calls = []
        self._rowcounts = list(rowcounts)
        self._fetchone_returns = list(fetchone_returns or [])
        self._fetchall_returns = list(fetchall_returns or [])
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        self.rowcount = self._rowcounts.pop(0) if self._rowcounts else 0

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


def _client_with_auth():
    from auth import verify_api_quota
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER
    return client


# ---------------------------------------------------------------------------
# 1. Anchor del marker en el endpoint
# ---------------------------------------------------------------------------
def test_marker_present_in_endpoint():
    """El endpoint debe mencionar `P1-HIST-AUDIT-1` para que un grep desde
    memoria/CLAUDE.md encuentre el cierre."""
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    assert "P1-HIST-AUDIT-1" in src
    # Justificación textual: que el comentario explique POR QUÉ no es solo
    # `created_at DESC` (defensa contra refactor que borre la complejidad).
    assert "_effectiveModifiedAt" in src or "_plan_modified_at" in src


# ---------------------------------------------------------------------------
# 2. Contrato SQL del SELECT target
# ---------------------------------------------------------------------------
def test_target_select_uses_greatest_modified_at():
    """El SQL del SELECT target debe mencionar GREATEST + COALESCE +
    `_plan_modified_at` + tie-breaker `created_at DESC`. Este test bloquea
    regresiones donde alguien restaure el `ORDER BY created_at DESC` simple.

    [P1-HIST-AUDIT-7 · 2026-05-09] El SELECT target ahora vive dentro
    de la transacción (post-advisory_lock); seguimos siendo el único
    SELECT con `LIMIT 1` en el endpoint.
    """
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)

    # Aislamos el bloque del SELECT target para no matchear contra la
    # cláusula de cancel-pending que también puede usar created_at.
    # El SELECT target es el ÚNICO lugar del endpoint con `LIMIT 1`.
    target_block_match = re.search(
        r"SELECT\s+id\s+FROM\s+meal_plans[^;]*?LIMIT\s+1",
        src,
        re.IGNORECASE | re.DOTALL,
    )
    assert target_block_match is not None, (
        "No se encontró el SELECT target con `LIMIT 1` en api_restore_plan. "
        "¿Fue refactorizado a otra función?"
    )
    block = target_block_match.group(0)
    assert "GREATEST" in block, (
        "SELECT target no usa GREATEST. P1-HIST-AUDIT-1 requiere "
        "max(created_at, _plan_modified_at) para SSOT con el frontend."
    )
    assert "_plan_modified_at" in block, (
        "SELECT target no referencia `_plan_modified_at`. Sin esto, el "
        "target diverge de History.jsx::_effectiveModifiedAt."
    )
    assert "COALESCE" in block, (
        "SELECT target no envuelve el cast en COALESCE. Sin fallback a "
        "created_at, planes legacy sin la key se ordenan inconsistente."
    )
    # Tie-breaker secundario por created_at DESC: aparece DESPUÉS del
    # GREATEST. La regex valida que el bloque tenga al menos dos
    # menciones de created_at (una en GREATEST, otra como tie-breaker).
    assert block.count("created_at") >= 3, (
        "SELECT target no incluye tie-breaker `created_at DESC` después "
        "del GREATEST. Sin tie-breaker, ties sobre _plan_modified_at "
        "pueden alternar entre llamadas (no determinismo)."
    )


# ---------------------------------------------------------------------------
# 3. Comportamiento: el endpoint procede con el id que devuelva el SELECT,
#    no el id "más reciente por created_at" si difieren
# ---------------------------------------------------------------------------
def test_endpoint_uses_target_returned_by_select():
    """Mockeando que el SELECT target devuelve `_TARGET_BY_MODIFIED` (un id
    distinto al "newest by created_at" simulado), el response del endpoint
    refleja ese id como `target_plan_id`. Demuestra que el endpoint NO
    hardcodea otra resolución y respeta el ORDER BY del SELECT.

    [P1-HIST-AUDIT-7 · 2026-05-09] El target ahora viene del cursor
    dentro de la transacción (no de execute_sql_query).
    [P2-HIST-AUDIT-5 · 2026-05-09] 6 statements totales: lock +
    target SELECT + cancel target + cancel source + release locks
    + UPDATE.
    """
    client = _client_with_auth()

    cursor = _CursorRecorder(
        rowcounts=[0, 0, 1, 0, 0, 1],
        fetchone_returns=[{"id": _TARGET_BY_MODIFIED}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post(
            "/api/plans/restore", json={"source_plan_id": _SOURCE_ID}
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_plan_id"] == _TARGET_BY_MODIFIED, (
        "El endpoint debe usar el id devuelto por el SELECT target, no "
        "computar otro internamente."
    )
    # Sanity: el UPDATE meal_plans (6º statement) debe haber recibido
    # ese mismo target_id.
    _, params_update = cursor.calls[5]
    assert _TARGET_BY_MODIFIED in params_update


# ---------------------------------------------------------------------------
# 4. Paridad lógica con History.jsx::_effectiveModifiedAt
# ---------------------------------------------------------------------------
def _python_effective_modified_at(created_at: datetime, modified_iso: str | None) -> datetime:
    """Mirror Python del helper JS para validar paridad de intención.

    History.jsx:35 hace `Math.max(parseDate(_plan_modified_at), parseDate(created_at))`
    con fallback a created_at si parsing falla. Esto replica esa lógica.
    """
    if modified_iso:
        try:
            mod = datetime.fromisoformat(modified_iso.replace("Z", "+00:00"))
            return max(created_at, mod)
        except (ValueError, TypeError):
            return created_at
    return created_at


def test_helper_parity_picks_modified_when_newer():
    """Plan B (older created_at, newer _plan_modified_at) gana sobre A."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    A_created = base + timedelta(days=10)  # más reciente por created_at
    A_modified = None
    B_created = base + timedelta(days=5)
    B_modified = (base + timedelta(days=20)).isoformat()  # restaurado después

    eff_A = _python_effective_modified_at(A_created, A_modified)
    eff_B = _python_effective_modified_at(B_created, B_modified)

    assert eff_B > eff_A, (
        "Helper falló: B (con _plan_modified_at más reciente) debería "
        "ordenar después de A. Esto confirma la INTENCIÓN del SELECT "
        "SQL en api_restore_plan."
    )


def test_helper_parity_falls_back_to_created_when_modified_missing():
    """Sin _plan_modified_at, el efectivo == created_at."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    eff = _python_effective_modified_at(base, None)
    assert eff == base


def test_helper_parity_falls_back_when_modified_corrupt():
    """_plan_modified_at no parseable → fallback a created_at (no crash)."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    eff = _python_effective_modified_at(base, "not-an-iso-date")
    assert eff == base


def test_helper_parity_max_when_created_newer():
    """created_at > _plan_modified_at → efectivo == created_at.

    Caso real: plan recién creado que aún no se ha modificado posterior;
    el _plan_modified_at podría estar sellado durante la creación misma
    pero ser ligeramente anterior por timing del INSERT.
    """
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    created = base + timedelta(days=5)
    modified = (base + timedelta(days=3)).isoformat()
    eff = _python_effective_modified_at(created, modified)
    assert eff == created
