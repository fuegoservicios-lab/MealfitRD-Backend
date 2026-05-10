"""[P1-HIST-LIFETIME-LESSONS · 2026-05-09] Tests del endpoint
``GET /api/plans/{id}/lifetime-lessons`` que surface el aprendizaje
continuo lifetime (3 estructuras de plan_data) en el modal del
Historial.

Bug original (audit Historial 2026-05-09 · gap P1-1):
    El tab "Lecciones" del modal solo leía `chunk_lesson_telemetry`
    (telemetría sobre el aprendizaje, no el aprendizaje en sí). Las 3
    estructuras reales (`_lifetime_lessons_summary`,
    `_lifetime_lessons_history`, `_critical_lessons_permanent`) viven
    en `meal_plans.plan_data` y eran invisibles al usuario en planes
    archivados.

Fix:
    Endpoint nuevo `/api/plans/{plan_id}/lifetime-lessons` que devuelve
    las 3 estructuras con caps defensivos (history y critical_permanent
    ≤50) en una sola request. Frontend extiende el tab "Lecciones" con
    sub-secciones que consumen este endpoint en paralelo a `/lessons`.

Cobertura:
    - Anchor del marker en el endpoint.
    - Auth: 401 sin verified_user_id.
    - Validación: 400 plan_id missing/invalid.
    - Ownership: 404 plan no existe O plan de otro user.
    - SELECT extrae las 3 keys del plan_data en una pasada.
    - Response shape: {plan_id, summary, history, critical_permanent, counts}.
    - Summary whitelist de keys numéricas + listas; coerción defensiva.
    - History reverse + cap ≤50.
    - Critical permanent cap ≤50.
    - Plan legacy sin las keys → 200 con summary=null + arrays vacíos.
    - Tipos corruptos (history dict en vez de list) → defaults seguros.
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "22222222-2222-2222-2222-222222222222"
_PLAN_ID = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"


def _client():
    from auth import verify_api_quota
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER
    return client


def _client_no_auth():
    from auth import verify_api_quota
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: None
    return client


# ---------------------------------------------------------------------------
# 1. Anchor + estructura del endpoint
# ---------------------------------------------------------------------------
def test_marker_present_in_endpoint():
    from routers.plans import api_plan_lifetime_lessons
    src = inspect.getsource(api_plan_lifetime_lessons)
    assert "P1-HIST-LIFETIME-LESSONS · 2026-05-09" in src


def test_endpoint_route_decorator():
    """Verifica que el endpoint está montado en la ruta correcta.
    El router de plans tiene prefix `/api/plans` aplicado en
    `routers/__init__.py`, así que la ruta efectiva es
    `/api/plans/{plan_id}/lifetime-lessons`."""
    from routers.plans import router
    paths = [r.path for r in router.routes if hasattr(r, "path")]
    assert "/api/plans/{plan_id}/lifetime-lessons" in paths


def test_select_extracts_three_lifetime_keys():
    """SELECT debe extraer las 3 estructuras del plan_data en una sola
    pasada (no 3 queries — single roundtrip)."""
    from routers.plans import api_plan_lifetime_lessons
    src = inspect.getsource(api_plan_lifetime_lessons)
    # SELECT plan_data->'_lifetime_lessons_summary' ...
    assert "plan_data->'_lifetime_lessons_summary'" in src
    assert "plan_data->'_lifetime_lessons_history'" in src
    assert "plan_data->'_critical_lessons_permanent'" in src


# ---------------------------------------------------------------------------
# 2. Auth + validación
# ---------------------------------------------------------------------------
def test_requires_auth():
    client = _client_no_auth()
    r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 401


def test_404_when_plan_missing_or_other_user():
    """Ownership SELECT devuelve None → 404 sin DOS-able discovery."""
    client = _client()
    with patch("db_core.execute_sql_query", return_value=None):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 3. Response shape canónico (datos completos)
# ---------------------------------------------------------------------------
def _full_payload_row():
    """Row simulado con las 3 estructuras pobladas (mirror del shape
    que `cron_tasks.py:~20720` produce)."""
    return {
        "summary": {
            "total_rejection_violations": 3,
            "total_allergy_violations": 1,
            "top_rejection_hits": ["Pollo guisado", "Lasaña"],
            "top_repeated_meal_names": ["Arroz con pollo", "Habichuelas"],
            "top_repeated_bases": ["arroz", "pollo"],
            "permanent_meal_blocklist": ["Arroz con habichuelas"],
            "_lifetime_window_days": 60,
            "_lifetime_proxy_ratio": 0.62,
            "_lifetime_user_logs_count": 8,
            "_lifetime_proxy_count": 13,
        },
        "history": [
            {"chunk": 1, "rejection_violations": 1, "allergy_violations": 0,
             "rejected_meals_that_reappeared": ["Pollo"]},
            {"chunk": 2, "rejection_violations": 2, "allergy_violations": 1,
             "repeated_meal_names": ["Habichuelas"]},
        ],
        "critical_permanent": [
            {"allergy_violations": 1, "rejection_violations": 0,
             "last_validated_at": "2026-05-08T12:00:00Z"},
        ],
    }


def test_full_payload_returns_canonical_shape():
    client = _client()
    with patch("db_core.execute_sql_query", return_value=_full_payload_row()):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    body = r.json()
    assert body["plan_id"] == _PLAN_ID
    # Summary debe ser dict no-null.
    assert isinstance(body["summary"], dict)
    assert body["summary"]["total_rejection_violations"] == 3
    assert body["summary"]["_lifetime_proxy_ratio"] == pytest.approx(0.62)
    assert body["summary"]["permanent_meal_blocklist"] == ["Arroz con habichuelas"]
    # History reversed (más reciente primero) — chunk 2 antes de chunk 1.
    assert isinstance(body["history"], list)
    assert len(body["history"]) == 2
    assert body["history"][0]["chunk"] == 2
    assert body["history"][1]["chunk"] == 1
    # Critical permanent.
    assert isinstance(body["critical_permanent"], list)
    assert len(body["critical_permanent"]) == 1
    # Counts.
    assert body["counts"]["history_total"] == 2
    assert body["counts"]["history_returned"] == 2
    assert body["counts"]["critical_permanent_total"] == 1
    assert body["counts"]["critical_permanent_returned"] == 1


# ---------------------------------------------------------------------------
# 4. Plan legacy sin las keys → defaults seguros
# ---------------------------------------------------------------------------
def test_legacy_plan_without_keys_returns_empty_defaults():
    """Plan archivado pre-rollout del aprendizaje continuo: no tiene
    las 3 keys en plan_data. El endpoint NO debe 500 — devuelve summary
    null + arrays vacíos para que el frontend renderice el tab limpio."""
    client = _client()
    legacy_row = {"summary": None, "history": None, "critical_permanent": None}
    with patch("db_core.execute_sql_query", return_value=legacy_row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"] is None
    assert body["history"] == []
    assert body["critical_permanent"] == []
    assert body["counts"]["history_total"] == 0
    assert body["counts"]["critical_permanent_total"] == 0


def test_corrupted_history_type_falls_to_empty_list():
    """history llega como dict (corrupción jsonb) → tratado como []."""
    client = _client()
    bad_row = {
        "summary": None,
        "history": {"not": "a list"},
        "critical_permanent": None,
    }
    with patch("db_core.execute_sql_query", return_value=bad_row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    body = r.json()
    assert body["history"] == []
    assert body["counts"]["history_total"] == 0


def test_corrupted_critical_type_falls_to_empty_list():
    """critical_permanent llega como string → tratado como []."""
    client = _client()
    bad_row = {
        "summary": None,
        "history": None,
        "critical_permanent": "not a list",
    }
    with patch("db_core.execute_sql_query", return_value=bad_row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    body = r.json()
    assert body["critical_permanent"] == []


# ---------------------------------------------------------------------------
# 5. Caps defensivos (history y critical_permanent ≤ 50)
# ---------------------------------------------------------------------------
def test_history_cap_applies_to_50_most_recent():
    """history con 80 entries → response trae las 50 más recientes
    (slice del tail) en orden reverso (más reciente primero)."""
    client = _client()
    # Productor (cron_tasks.py) hace append-only — los más recientes
    # están al FINAL del array. El endpoint hace slice del tail y
    # luego reverse. Generamos 80 con chunk numerado 1..80.
    history = [{"chunk": i, "rejection_violations": 0} for i in range(1, 81)]
    row = {"summary": None, "history": history, "critical_permanent": None}
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    body = r.json()
    assert len(body["history"]) == 50
    # El primero del response debe ser el último del array original
    # (chunk=80) tras reverse.
    assert body["history"][0]["chunk"] == 80
    # El último del response debe ser chunk=31 (los 50 más recientes
    # del tail eran chunks 31..80; reverse los pone 80..31).
    assert body["history"][-1]["chunk"] == 31
    assert body["counts"]["history_total"] == 80
    assert body["counts"]["history_returned"] == 50


def test_critical_permanent_cap_applies_to_50():
    client = _client()
    critical = [
        {"allergy_violations": 1, "rejection_violations": 0,
         "id": i} for i in range(60)
    ]
    row = {"summary": None, "history": None, "critical_permanent": critical}
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    body = r.json()
    assert len(body["critical_permanent"]) == 50
    assert body["counts"]["critical_permanent_total"] == 60
    assert body["counts"]["critical_permanent_returned"] == 50


# ---------------------------------------------------------------------------
# 6. Summary whitelist (keys + tipos)
# ---------------------------------------------------------------------------
def test_summary_whitelist_keys_numeric_and_lists():
    """Summary devuelto debe contener TODAS las keys whitelisted (con
    valor o null/[]). Drift detection: si el productor añade una key
    nueva, el endpoint la propagaría como key extra (no en whitelist)
    pero las whitelisted SIEMPRE están presentes."""
    client = _client()
    minimal_summary = {
        "total_rejection_violations": 0,
        "top_rejection_hits": [],
    }
    row = {"summary": minimal_summary, "history": None, "critical_permanent": None}
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    body = r.json()
    s = body["summary"]
    # Keys numéricas siempre presentes con valor o None.
    for k in ("total_rejection_violations", "total_allergy_violations",
              "_lifetime_window_days", "_lifetime_proxy_ratio",
              "_lifetime_user_logs_count", "_lifetime_proxy_count"):
        assert k in s, f"summary debe declarar key whitelisted '{k}'"
    # Lists con default [].
    for k in ("top_rejection_hits", "top_repeated_bases",
              "top_repeated_meal_names", "permanent_meal_blocklist"):
        assert isinstance(s[k], list), (
            f"summary['{k}'] debe ser lista; got {type(s[k])}"
        )
    # Valor presente preserva tipo.
    assert s["total_rejection_violations"] == 0
    # Valor ausente cae a None.
    assert s["total_allergy_violations"] is None


def test_summary_rejects_invalid_numeric_types():
    """Si el productor escribió un str en una key numérica (corrupción),
    el endpoint la cae a None — el frontend renderiza "—" en la UI."""
    client = _client()
    bad_summary = {
        "total_rejection_violations": "not a number",  # string
        "total_allergy_violations": True,  # bool — explicit reject
        "_lifetime_proxy_ratio": 0.5,  # válido
    }
    row = {"summary": bad_summary, "history": None, "critical_permanent": None}
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    body = r.json()
    s = body["summary"]
    assert s["total_rejection_violations"] is None
    # bool NO debe contar como número (isinstance(True, int) es True
    # en Python por herencia, pero el endpoint excluye bool explícitamente).
    assert s["total_allergy_violations"] is None
    assert s["_lifetime_proxy_ratio"] == pytest.approx(0.5)


def test_summary_list_keys_coerce_strings():
    """Las listas son sanitizadas: cada elemento se convierte a str
    para evitar render de objetos anidados que rompan el frontend."""
    client = _client()
    summary = {
        "permanent_meal_blocklist": ["Pollo", 123, None, {"meal": "tacos"}],
    }
    row = {"summary": summary, "history": None, "critical_permanent": None}
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    body = r.json()
    items = body["summary"]["permanent_meal_blocklist"]
    # None se filtra; números → str; dicts → str (el frontend renderiza
    # el repr — preferible a crash por intentar `<span>{obj}</span>`).
    assert all(isinstance(x, str) for x in items)
    assert "Pollo" in items
    assert "123" in items
    # None excluido.
    assert None not in items


# ---------------------------------------------------------------------------
# 7. Counts shape
# ---------------------------------------------------------------------------
def test_counts_shape_complete():
    """`counts` siempre presente con los 4 contadores aunque las
    estructuras estén vacías."""
    client = _client()
    legacy_row = {"summary": None, "history": None, "critical_permanent": None}
    with patch("db_core.execute_sql_query", return_value=legacy_row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    body = r.json()
    counts = body["counts"]
    for k in ("history_total", "history_returned",
              "critical_permanent_total", "critical_permanent_returned"):
        assert k in counts
        assert isinstance(counts[k], int)
        assert counts[k] >= 0


# ---------------------------------------------------------------------------
# 8. Caps numéricos exportados — drift detection
# ---------------------------------------------------------------------------
def test_caps_constants_match_50():
    """Los caps están en constantes módulo-level para que un futuro
    cambio sea explícito (vs hardcoded en 4+ sitios). Drift detection:
    si alguien sube el cap a 100 sin justificación, el test falla y
    pide revisión."""
    from routers.plans import _LIFETIME_HISTORY_CAP, _LIFETIME_CRITICAL_CAP
    assert _LIFETIME_HISTORY_CAP == 50
    assert _LIFETIME_CRITICAL_CAP == 50
