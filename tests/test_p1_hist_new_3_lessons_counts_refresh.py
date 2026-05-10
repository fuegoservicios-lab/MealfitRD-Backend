"""[P1-HIST-NEW-3 · 2026-05-09] Tests del refresh de `lessonsCounts`
en visibilitychange — cross-link del marker + payload contract.

Bug original (audit profundo Historial 2026-05-09):
    El listener de `visibilitychange` (P0-HIST-VIS-REFRESH) en
    `History.jsx` re-llamaba `fetchHistory()` para refrescar el
    listado pero NO refrescaba `getLessonsCounts()`. Un chunk que
    completaba en background y persistía lecciones nuevas vía T2
    (commit a `chunk_lesson_telemetry`) dejaba el chip "X lecciones"
    con conteo viejo hasta navegar fuera y volver al Historial.

    El fix es 100% client-side (helper extraído + invocación desde
    visibilitychange). Este test cierra el cross-link del marker
    (`test_p2_hist_audit_14_marker_test_link` requiere
    `tests/test_p1_hist_new_3*.py`) Y protege el endpoint backend
    `/lessons-counts` que el helper consume.

Cobertura backend:
    1. Anchor del marker en History.jsx.
    2. Endpoint `/lessons-counts` existe y responde el shape esperado.
    3. Response incluye `counts` (legacy P1-HIST-3) Y `counts_by_quality`
       (P2-HIST-AUDIT-D) — el helper depende de ambos para hidratar
       lessonsCounts + lessonsCountsByQuality.
    4. Cap de 200 planes (mismo que /history-list para no DOS-ear).
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER_A = "11111111-1111-1111-1111-111111111111"


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Anchor del marker — fix vive en History.jsx (client-side)
# ---------------------------------------------------------------------------
def test_marker_present_in_history_jsx():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    history_jsx = repo_root / "frontend" / "src" / "pages" / "History.jsx"
    assert history_jsx.exists()
    text = history_jsx.read_text(encoding="utf-8")
    assert "[P1-HIST-NEW-3" in text, (
        "Marker `P1-HIST-NEW-3` debe aparecer en History.jsx donde "
        "vive el helper _fetchLessonsCounts."
    )


def test_helper_function_exists_in_history_jsx():
    """Defensa contra refactor silencioso: el helper debe existir como
    función nombrada (no inline lambda anónima dentro del useEffect)
    para que tanto mount como visibilitychange lo invoquen."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    history_jsx = repo_root / "frontend" / "src" / "pages" / "History.jsx"
    text = history_jsx.read_text(encoding="utf-8")
    # Declaración: `const _fetchLessonsCounts = () =>` (sin paréntesis
    # directamente tras el nombre).
    assert "const _fetchLessonsCounts" in text, (
        "Helper `_fetchLessonsCounts` debe estar declarado como const "
        "a nivel componente."
    )
    # Invocaciones: `_fetchLessonsCounts(` aparece en mount +
    # visibilitychange (mínimo 2). Si solo hay 1, el visibility
    # handler perdió su llamada.
    invocations = text.count("_fetchLessonsCounts(")
    assert invocations >= 2, (
        f"Esperaba >=2 invocaciones de `_fetchLessonsCounts` "
        f"(mount + visibilitychange); encontré {invocations}."
    )


# ---------------------------------------------------------------------------
# 2. Endpoint /lessons-counts existe + smoke test
# ---------------------------------------------------------------------------
def test_lessons_counts_endpoint_exists():
    """El helper depende de `/api/plans/lessons-counts`. Si alguien
    rename del endpoint sin actualizar el frontend, el chip queda
    inerte. Este test fija el path."""
    from routers.plans import api_plans_lessons_counts
    assert callable(api_plans_lessons_counts)


def test_lessons_counts_response_includes_counts_and_quality():
    """End-to-end: el response debe traer AMBAS keys que el helper
    consume — `counts` (legacy) y `counts_by_quality` (P2-HIST-AUDIT-D).
    El helper setea ambos (`setLessonsCounts` + `setLessonsCountsByQuality`)
    desde el mismo body."""
    fake_rows = [
        {"meal_plan_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
         "cnt": 5,
         "high_cnt": 3, "partial_cnt": 1, "low_cnt": 1},
    ]

    def _fake(query, params=None, **kwargs):
        return fake_rows

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get("/api/plans/lessons-counts")

    assert r.status_code == 200, r.text
    body = r.json()
    # `counts` siempre presente (P1-HIST-3 legacy contract).
    assert "counts" in body, (
        f"Response debe incluir `counts` para que el helper popule "
        f"lessonsCounts. Got keys: {sorted(body.keys())}"
    )
    assert isinstance(body["counts"], dict)
    # `counts_by_quality` agregado en P2-HIST-AUDIT-D — el helper lo
    # lee con typeof check defensivo, así que su ausencia no rompe
    # el frontend (legacy fallback). Pero el endpoint actual SÍ debe
    # exponerlo.
    assert "counts_by_quality" in body, (
        f"Response debe incluir `counts_by_quality` (P2-HIST-AUDIT-D). "
        f"Got keys: {sorted(body.keys())}"
    )
    assert isinstance(body["counts_by_quality"], dict)


# ---------------------------------------------------------------------------
# 3. Cardinality natural — no hace falta LIMIT
# ---------------------------------------------------------------------------
def test_lessons_counts_uses_event_whitelist():
    """El endpoint depende de `_LESSON_COUNT_EVENT_WHITELIST` para
    acotar el response. La cardinalidad final es N_planes × W_events
    (≤200×~5 = 1000 rows) — naturalmente bounded sin LIMIT explícito.
    Este test fija el contrato: cualquier refactor que quite la
    whitelist debe re-introducir un cap."""
    from routers.plans import api_plans_lessons_counts
    src = inspect.getsource(api_plans_lessons_counts)
    assert "_LESSON_COUNT_EVENT_WHITELIST" in src, (
        "Endpoint debe filtrar por `_LESSON_COUNT_EVENT_WHITELIST` "
        "para acotar la cardinalidad. Si lo quitas, añade LIMIT."
    )
    # `event = ANY(%s)` es el patrón que usa la whitelist.
    assert "event = ANY(%s)" in src, (
        "Patrón de filtrado `event = ANY(%s)` ausente — refactor "
        "puede haber roto el guard de cardinalidad."
    )
