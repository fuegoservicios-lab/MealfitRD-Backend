"""[P0-AUDIT-HIST-1 · 2026-05-09] Tests: ``api_restore_plan`` cancela
chunks en TODOS los estados "vivos" del SSOT — no solo
``('pending', 'processing')``.

Bug original (audit Historial 2026-05-09):
    El restore atómico cancelaba solo `pending`/`processing`. El SSOT
    del resto del backend cubre 5 estados:
      ('pending', 'processing', 'stale', 'pending_user_action', 'failed')
    (ver db_plans.py:573, services.py:222, routers/plans.py:2059).

    Restar `stale`/`pending_user_action`/`failed` dejaba zombis: un
    chunk pausado por pantry/tz/missing-lessons (status =
    `pending_user_action` por crons en cron_tasks.py:5977/6038/6114/
    10636/12091/16798) o uno con heartbeat expirado (status='stale')
    podía despertarse tras el restore con `pipeline_snapshot` del
    plan PREVIO al target y escribir días al `plan_data` recién
    sobrescrito → exactamente la corrupción silenciosa que P0-HIST-1
    fue diseñado a prevenir.

    Confirmado en producción al inspeccionar la DB del proyecto
    MealFitRD: chunk en `pending_user_action` con `plan_data` SIN
    `_user_action_required`/`_recovery_exhausted_chunks` →
    inconsistencia entre la fuente operativa (queue) y la fuente
    consumida por el Historial (jsonb flags).

Fix:
    Ambos UPDATE (target y source) extienden el filtro
    `WHERE status IN (...)` a los 5 estados del SSOT.

Cobertura:
    1. Anchor del marker en el endpoint.
    2. Statement target cubre los 5 estados.
    3. Statement source cubre los 5 estados.
    4. Drift guard: las 5 strings literales DEBEN aparecer en cada UPDATE.
    5. Drift guard: el filtro NO debe incluir `cancelled` ni `completed`
       (sería destructivo: re-cancelar histórico o revertir completados).
    6. SSOT alignment: los 5 estados del restore == set canónico de
       db_plans.py / services.py / routers/plans.py:2059.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
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

# SSOT canónico de "estados vivos" del chunk queue. Si esta lista
# diverge del filtro real del restore, el contrato SSOT está roto.
# Sincronizar con: db_plans.py:573, services.py:222,
# routers/plans.py:2059.
_LIVE_CHUNK_STATES = (
    "pending",
    "processing",
    "stale",
    "pending_user_action",
    "failed",
)


# ---------------------------------------------------------------------------
# Mocks (mismos que test_p0_hist_1 / test_p2_hist_audit_5)
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


def _normalize_sql(sql: str) -> str:
    """Colapsa whitespace para que la búsqueda de literales funcione
    sin importar indentación/saltos de línea de la query."""
    return re.sub(r"\s+", " ", sql).strip()


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_endpoint():
    """El marker `P0-AUDIT-HIST-1` debe estar citado en el endpoint
    para que un grep + git blame lleve directo al fix."""
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    assert "P0-AUDIT-HIST-1" in src, (
        "El endpoint api_restore_plan debe citar `P0-AUDIT-HIST-1` "
        "en el comentario load-bearing del cancel — sin el anchor, "
        "un futuro refactor que reduzca el filtro pierde la motivación."
    )


# ---------------------------------------------------------------------------
# 2. Statement target cubre los 5 estados (runtime check)
# ---------------------------------------------------------------------------
def test_target_cancel_filter_includes_all_five_live_states():
    """El primer UPDATE plan_chunk_queue (cancela chunks del TARGET)
    debe filtrar por los 5 estados del SSOT, no solo
    pending/processing como antes del fix."""
    client = _client()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 3, 0, 0, 1],  # cancel_target = 3
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post(
            "/api/plans/restore", json={"source_plan_id": _SOURCE_ID}
        )

    assert r.status_code == 200, r.text
    # El cancel del target es el 3er statement (advisory_lock=0,
    # SELECT target=1, cancel target=2 en la lista 0-indexed).
    sql_target, params_target = cursor.calls[2]
    assert "UPDATE plan_chunk_queue" in sql_target
    assert "restore_overwrite" in params_target
    norm = _normalize_sql(sql_target)
    for state in _LIVE_CHUNK_STATES:
        assert f"'{state}'" in norm, (
            f"El cancel del target debe filtrar `'{state}'` (estado "
            f"vivo del SSOT). Filtro actual: {norm!r}"
        )


# ---------------------------------------------------------------------------
# 3. Statement source cubre los 5 estados (runtime check)
# ---------------------------------------------------------------------------
def test_source_cancel_filter_includes_all_five_live_states():
    """El segundo UPDATE plan_chunk_queue (cancela chunks del SOURCE)
    debe filtrar por los 5 estados — un chunk del source en
    pending_user_action puede resucitar tras restore si nunca se
    cancela."""
    client = _client()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 1, 2, 0, 1],  # cancel_source = 2
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post(
            "/api/plans/restore", json={"source_plan_id": _SOURCE_ID}
        )

    assert r.status_code == 200, r.text
    sql_source, params_source = cursor.calls[3]
    assert "UPDATE plan_chunk_queue" in sql_source
    assert "restore_source_archived" in params_source
    norm = _normalize_sql(sql_source)
    for state in _LIVE_CHUNK_STATES:
        assert f"'{state}'" in norm, (
            f"El cancel del source debe filtrar `'{state}'` (estado "
            f"vivo del SSOT). Filtro actual: {norm!r}"
        )


# ---------------------------------------------------------------------------
# 4. Drift guard: filtro NO debe incluir `cancelled` ni `completed`
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("forbidden", ["cancelled", "completed"])
def test_cancel_filter_does_not_include_terminal_states(forbidden):
    """`cancelled` y `completed` son estados TERMINALES — incluirlos
    en el filtro re-cancelaría chunks ya cancelados (idempotente pero
    ruidoso) o, peor, re-marcaría completados como cancelled
    (destructivo: pierde días generados del plan_data restaurado)."""
    client = _client()
    cursor = _CursorRecorder(
        rowcounts=[0, 0, 0, 0, 0, 1],
        fetchone_returns=[{"id": _TARGET_ID}],
    )
    pool_mock = _build_pool_mock(cursor)

    with patch("db_core.execute_sql_query", side_effect=[_SOURCE_ROW]), \
         patch("db_core.connection_pool", pool_mock):
        r = client.post(
            "/api/plans/restore", json={"source_plan_id": _SOURCE_ID}
        )
    assert r.status_code == 200

    for idx in (2, 3):  # cancel_target, cancel_source
        sql, _params = cursor.calls[idx]
        norm = _normalize_sql(sql)
        # Buscar el bloque del IN para no fallar por menciones
        # incidentales del literal en otros lugares (e.g.,
        # `SET status = 'cancelled'`). El filtro está en el
        # `WHERE ... status IN (...)`.
        m = re.search(r"status\s+IN\s*\(([^)]*)\)", norm, re.IGNORECASE)
        assert m is not None, f"No encontré el filtro IN en {norm!r}"
        in_list = m.group(1)
        assert f"'{forbidden}'" not in in_list, (
            f"El filtro IN del cancel NO debe incluir `'{forbidden}'` "
            f"(estado terminal). Filtro actual: IN ({in_list})"
        )


# ---------------------------------------------------------------------------
# 5. SSOT alignment: el set del restore == el set de los demás call sites
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_ROOT = _REPO_ROOT / "backend"


def _extract_in_states(text: str, anchor: str) -> set[str]:
    """Busca un fragmento `status IN ( 'a', ... )` que contenga
    `'pending_user_action'` Y aparezca arriba del literal `anchor`
    en el texto (ese anchor suele ser el dead_letter_reason del
    params tuple, ej. ``"restore_overwrite"``). Devuelve el set de
    literales del bloque IN, o None si no hay match.

    Estrategia: buscar TODOS los bloques `status IN (...)` del archivo
    con regex multi-línea, descartar los que no mencionen
    `pending_user_action` (no son del SSOT) y devolver el último cuya
    posición precede al primer match del anchor — eso es el WHERE del
    UPDATE que cancelan ese flujo.
    """
    idx = text.find(anchor)
    if idx < 0:
        return None
    # `re.DOTALL` permite que `.` cubra newlines del bloque multi-línea.
    in_blocks = list(
        re.finditer(
            r"status\s+IN\s*\(([^)]*'pending_user_action'[^)]*)\)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )
    # Filtrar por bloques cuya posición está antes del anchor (el
    # WHERE precede al params tuple que contiene el anchor).
    preceding = [m for m in in_blocks if m.end() < idx]
    if not preceding:
        return None
    raw = preceding[-1].group(1)
    return set(re.findall(r"'([^']+)'", raw))


def test_restore_ssot_matches_db_plans_and_services():
    """Drift detection cross-archivo: el set de estados del restore
    debe coincidir con los call sites canónicos del SSOT
    (db_plans.py:573, services.py:222, routers/plans.py:2059).

    Si alguien añade un estado nuevo al SSOT (e.g. 'preempted')
    pero olvida añadirlo aquí, el restore deja zombis. Este test
    falla loud para forzar la actualización en lock-step.
    """
    plans_text = (
        _BACKEND_ROOT / "routers" / "plans.py"
    ).read_text(encoding="utf-8")

    # Anchor del fix actual (el primer cancel del target).
    restore_states = _extract_in_states(plans_text, "restore_overwrite")
    assert restore_states is not None, (
        "No pude extraer los estados del cancel `restore_overwrite`. "
        "Verifica que el anchor sigue presente en el endpoint."
    )

    # Comparar contra el SSOT canónico de db_plans.py — usado por
    # los crons de chunk pickup y por services.py (mismo set en ambos).
    db_plans_text = (_BACKEND_ROOT / "db_plans.py").read_text(
        encoding="utf-8"
    )
    # Tomar el primer bloque `status IN (...)` que mencione los 5
    # estados — aceptamos cualquier orden de literales.
    ssot_matches = re.findall(
        r"status\s+IN\s*\(([^)]*'pending_user_action'[^)]*)\)",
        db_plans_text,
        re.IGNORECASE,
    )
    assert ssot_matches, (
        "No pude extraer el set SSOT de estados desde db_plans.py "
        "(buscando `status IN (...) ... 'pending_user_action'`). "
        "Verifica que el SSOT sigue documentado allí."
    )
    ssot_states_self = set(re.findall(r"'([^']+)'", ssot_matches[0]))

    assert restore_states == ssot_states_self, (
        f"DRIFT del SSOT detectado.\n"
        f"  Restore:        {sorted(restore_states)}\n"
        f"  SSOT canónico:  {sorted(ssot_states_self)}\n"
        f"Si añadiste un estado nuevo a uno, debes añadirlo a TODOS "
        f"los call sites — sino el restore deja chunks zombis."
    )


# ---------------------------------------------------------------------------
# 6. Comentario load-bearing explica POR QUÉ los 5 estados
# ---------------------------------------------------------------------------
def test_comment_explains_five_state_motivation():
    """El comentario del fix debe explicar por qué los 5 estados son
    necesarios — sino, un refactor "limpiador" podría simplificar
    de vuelta a 2 sin entender la consecuencia."""
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    anchor_idx = src.find("P0-AUDIT-HIST-1")
    assert anchor_idx > -1
    block = src[anchor_idx:anchor_idx + 1800]
    # Al menos uno de estos conceptos clave debe estar mencionado:
    motivations = (
        "pending_user_action",
        "stale",
        "zombi",
        "pipeline_snapshot",
        "P0-HIST-1",
        "SSOT",
    )
    matches = [m for m in motivations if m in block]
    assert len(matches) >= 3, (
        f"El comentario debe citar al menos 3 conceptos clave de la "
        f"motivación. Encontrados: {matches}. Esperado del set: "
        f"{motivations}."
    )
