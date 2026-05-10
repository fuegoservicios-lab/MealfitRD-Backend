"""[P1-HIST-AUDIT-3 · 2026-05-09] Tests: ``api_restore_plan`` debe
limpiar las flags de fallo del SOURCE antes de persistir el plan_data
en el target.

Bug original (audit historial 2026-05-08):
    El ``enriched_pd`` build en `api_restore_plan` solo agregaba
    `_plan_modified_at` y `_restored_from_plan_id`, pero arrastraba
    intactas las flags `_user_action_required` y
    `_recovery_exhausted_chunks` del plan SOURCE. Resultado: tras
    reactivar un plan archivado que originalmente había fallado un
    chunk (banner CTA "regenera chunks fallados"), el banner aparecía
    en el plan recién reactivado y `getStatusInfo` lo clasificaba
    como `failed`/`action_required` — aunque la decisión explícita
    del usuario de reactivar implica que acepta el plan tal-cual.

Fix:
    Pop de ambas keys en `enriched_pd` antes del UPDATE. Si la
    generación del target vuelve a fallar después del restore, los
    crons re-setearán las flags con los chunks REALES del nuevo
    intento — limpieza no destructiva del estado histórico.

Cobertura:
    - Anchor del marker en el endpoint.
    - Cuando el SOURCE tiene `_user_action_required`, el plan_data
      persistido en el UPDATE (params del jsonb) NO la contiene.
    - Idem para `_recovery_exhausted_chunks` (lista no vacía).
    - Limpieza es no-destructiva: keys NO relacionadas con fallo
      (ej: `name`, `days`, `_shopping_coherence_block_history`,
      `_user_forced_simplified_weeks`) sobreviven al restore.
    - `_plan_modified_at` y `_restored_from_plan_id` siguen siendo
      sellados (sin regresión vs P0-HIST-1).
"""
from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "11111111-1111-1111-1111-111111111111"
_SOURCE_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_TARGET_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


# ---------------------------------------------------------------------------
# Mocks compartidos con test_p0_hist_1
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
        if self._fetchone_returns:
            return self._fetchone_returns.pop(0)
        return None

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


def _source_with_failure_flags() -> dict:
    """Source row con flags de fallo + payload sano para cross-check."""
    return {
        "id": _SOURCE_ID,
        "plan_data": {
            "name": "Plan Histórico con Fallo",
            "days": [{"meals": [{"name": "Mangú"}]}],
            "totalDays": 7,
            # Flags de fallo del SOURCE — DEBEN limpiarse en el target.
            "_user_action_required": {
                "title": "Acción requerida",
                "body": "Regenera los chunks fallados",
                "reason": "unrecoverable_missing_anchor",
                "chunk_id": "ccccccc",
            },
            "_recovery_exhausted_chunks": [
                {"week_number": 2, "chunk_id": "ccccccc"},
            ],
            # Keys NO relacionadas con fallo — DEBEN sobrevivir.
            "_shopping_coherence_block_history": [
                {"action_taken": "reject_minor", "ts": "2026-05-01T00:00:00+00:00"},
            ],
            "_user_forced_simplified_weeks": {"2": "2026-05-01T00:00:00+00:00"},
            "assessment": {"diet_preference": "omnivorous"},
        },
        "name": "Plan Histórico con Fallo",
        "calories": 2200,
        "macros": {"protein": 140, "carbs": 250, "fats": 60},
        "meal_names": ["Mangú"],
        "ingredients": ["plátano"],
        "techniques": ["hervido"],
    }


def _captured_plan_data(cursor) -> dict:
    """Extrae el plan_data persistido en el UPDATE meal_plans.

    [P1-HIST-AUDIT-7 · 2026-05-09] El UPDATE es el 5º statement (lock +
    target SELECT + cancel + release + update). Localizamos por
    contenido (`UPDATE meal_plans`) en lugar de índice fijo para
    robustez ante reordenamientos futuros.
    """
    update_calls = [
        (i, c) for i, c in enumerate(cursor.calls)
        if "UPDATE meal_plans" in c[0]
    ]
    assert len(update_calls) == 1, (
        f"Se esperaba exactamente 1 `UPDATE meal_plans`, got "
        f"{len(update_calls)} en: {[c[0][:60] for c in cursor.calls]}"
    )
    _, (_, update_params) = update_calls[0]
    plan_data_json = update_params[0]
    return json.loads(plan_data_json)


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_api_restore_plan():
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    assert "P1-HIST-AUDIT-3" in src, (
        "api_restore_plan debe mencionar P1-HIST-AUDIT-3 para que un "
        "grep desde memoria/CLAUDE.md encuentre el cierre."
    )


# ---------------------------------------------------------------------------
# 2. _user_action_required del SOURCE NO persiste en el target
# ---------------------------------------------------------------------------
def test_user_action_required_cleared_in_persisted_plan_data():
    client = _client()
    src = _source_with_failure_flags()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 1, 1, 1, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[src]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200, r.text

    persisted_pd = _captured_plan_data(cursor)
    assert "_user_action_required" not in persisted_pd, (
        "P1-HIST-AUDIT-3: `_user_action_required` debe limpiarse en restore. "
        "Arrastrar la flag deja al target con banner CTA del SOURCE aunque "
        "la decisión de reactivar implica aceptar el plan tal-cual."
    )


# ---------------------------------------------------------------------------
# 3. _recovery_exhausted_chunks del SOURCE NO persiste en el target
# ---------------------------------------------------------------------------
def test_recovery_exhausted_chunks_cleared_in_persisted_plan_data():
    client = _client()
    src = _source_with_failure_flags()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 0, 0, 0, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[src]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200, r.text

    persisted_pd = _captured_plan_data(cursor)
    assert "_recovery_exhausted_chunks" not in persisted_pd, (
        "P1-HIST-AUDIT-3: `_recovery_exhausted_chunks` debe limpiarse en "
        "restore. Mantenerla causa que History.jsx::getStatusInfo "
        "clasifique el target como `failed` post-restore."
    )


# ---------------------------------------------------------------------------
# 4. Limpieza NO es over-zealous: keys no relacionadas con fallo persisten
# ---------------------------------------------------------------------------
def test_unrelated_keys_survive_restore():
    """Solo `_user_action_required` y `_recovery_exhausted_chunks` se
    limpian. Otras keys del plan_data (datos del plan, telemetría no-fallo,
    metadata UX) deben transferirse al target intactas.
    """
    client = _client()
    src = _source_with_failure_flags()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 0, 0, 0, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[src]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200, r.text

    persisted_pd = _captured_plan_data(cursor)
    # Datos del plan.
    assert persisted_pd["name"] == "Plan Histórico con Fallo"
    assert persisted_pd["totalDays"] == 7
    assert persisted_pd["days"][0]["meals"][0]["name"] == "Mangú"
    # Metadata UX (P2-HIST-3, P2-HIST-4) — NO son flags de fallo.
    assert persisted_pd["_user_forced_simplified_weeks"] == {
        "2": "2026-05-01T00:00:00+00:00"
    }
    assert persisted_pd["_shopping_coherence_block_history"] == [
        {"action_taken": "reject_minor", "ts": "2026-05-01T00:00:00+00:00"}
    ]
    assert persisted_pd["assessment"] == {"diet_preference": "omnivorous"}


# ---------------------------------------------------------------------------
# 5. Sin regresión P0-HIST-1: enriched_pd sigue sellando
#    `_plan_modified_at` y `_restored_from_plan_id`
# ---------------------------------------------------------------------------
def test_p0_hist_1_invariants_preserved():
    client = _client()
    src = _source_with_failure_flags()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 0, 0, 0, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[src]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200, r.text

    persisted_pd = _captured_plan_data(cursor)
    # P0-HIST-1: dos keys load-bearing del enriched_pd.
    assert "_plan_modified_at" in persisted_pd, (
        "Regresión P0-HIST-1: el enriched_pd debe sellar "
        "`_plan_modified_at` (consumido por crons que filtran "
        "'planes editados últimas 24h')."
    )
    assert persisted_pd["_restored_from_plan_id"] == _SOURCE_ID, (
        "Regresión P0-HIST-1: `_restored_from_plan_id` permite "
        "correlación post-mortem entre target y source."
    )


# ---------------------------------------------------------------------------
# 6. Source SIN flags: limpieza es no-op (idempotente, no rompe planes
#    sanos)
# ---------------------------------------------------------------------------
def test_source_without_failure_flags_is_noop_for_cleanup():
    client = _client()
    src = _source_with_failure_flags()
    src["plan_data"].pop("_user_action_required")
    src["plan_data"].pop("_recovery_exhausted_chunks")

    cursor = _CursorRecorder(
        rowcounts=[0, 0, 0, 0, 0, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[src]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post("/api/plans/restore", json={"source_plan_id": _SOURCE_ID})
    assert r.status_code == 200, r.text

    persisted_pd = _captured_plan_data(cursor)
    # No-op: keys siguen ausentes (no las inserta, no las invierte).
    assert "_user_action_required" not in persisted_pd
    assert "_recovery_exhausted_chunks" not in persisted_pd
    # Y el resto del plan_data sigue intacto.
    assert persisted_pd["name"] == "Plan Histórico con Fallo"
