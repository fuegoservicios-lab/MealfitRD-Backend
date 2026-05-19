"""[P0-HIST-LEARN-1 · 2026-05-09] Surface de `_last_chunk_learning` en
``GET /api/plans/{id}/lifetime-lessons``.

Bug original (audit Historial 2026-05-09 · gap P0):
    `_last_chunk_learning` es el SSOT de qué aprendió el chunk anterior
    y se inyecta como semilla del prompt al PRÓXIMO chunk del cron
    rolling_refill (ver `_persist_last_chunk_learning` en cron_tasks.py).
    Sin embargo, ningún endpoint del Historial lo exponía — diagnosticar
    "por qué el chunk N+1 generó X" requería SQL al jsonb. La data
    real del plan 98d902e3 confirma que la key tiene 18 sub-keys ricas
    (learning_signal_strength, rebuilt_from_pipeline_failure,
    rejected_meals_that_reappeared, …) — todas invisibles al user.

Fix:
    Extender el endpoint `/lifetime-lessons` para incluir
    `last_chunk_learning` con coerción defensiva por tipo (numeric /
    bool / str / list). Plan legacy sin la key responde
    `last_chunk_learning: None` y la sub-sección queda oculta.

Cobertura:
    - SELECT extrae la key del plan_data en la misma pasada que las
      otras 3 estructuras lifetime (single roundtrip).
    - Whitelist de keys split por tipo (numeric / bool / str / list).
    - Coerción tolerante a corrupciones: keys con tipo inesperado
      caen al default (None / []) sin romper la sub-sección.
    - Plan legacy sin la key → response con `last_chunk_learning: None`.
    - Bool keys NO se confunden con int (Python `isinstance(True, int)`
      es True — el coerce numeric debe excluirlas explícitamente).
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "33333333-3333-3333-3333-333333333333"
_PLAN_ID = "ffffffff-ffff-ffff-ffff-ffffffffffff"


def _client():
    from auth import verify_api_quota, get_verified_user_id
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER
    return client


# ---------------------------------------------------------------------------
# 1. Anchor + SELECT extiende a la 4ª key
# ---------------------------------------------------------------------------
def test_marker_present():
    from routers.plans import api_plan_lifetime_lessons
    src = inspect.getsource(api_plan_lifetime_lessons)
    assert "P0-HIST-LEARN-1" in src


def test_select_extracts_last_chunk_learning():
    from routers.plans import api_plan_lifetime_lessons
    src = inspect.getsource(api_plan_lifetime_lessons)
    assert "plan_data->'_last_chunk_learning'" in src
    assert "AS last_chunk_learning" in src


def test_response_includes_last_chunk_learning_key():
    """Response shape SIEMPRE declara la key (puede ser None) para que
    el frontend no tenga que detectar 'feature flag' por presencia."""
    client = _client()
    legacy_row = {
        "summary": None, "history": None, "critical_permanent": None,
        "last_chunk_learning": None,
    }
    with patch("db_core.execute_sql_query", return_value=legacy_row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    body = r.json()
    assert "last_chunk_learning" in body
    assert body["last_chunk_learning"] is None


# ---------------------------------------------------------------------------
# 2. Coerción defensiva por tipo
# ---------------------------------------------------------------------------
def _row_with_lcl(lcl):
    return {
        "summary": None, "history": None, "critical_permanent": None,
        "last_chunk_learning": lcl,
    }


def test_full_payload_normalizes_all_keys():
    """Payload realista (mirror del shape que produce
    `_persist_last_chunk_learning` en cron_tasks.py)."""
    lcl = {
        "chunk": 2,
        "timestamp": "2026-05-09T12:34:56Z",
        "repeat_pct": 0.18,
        "ingredient_base_repeat_pct": 0.42,
        "allergy_violations": 1,
        "rejection_violations": 2,
        "fatigued_violations": 0,
        "low_confidence": True,
        "metrics_unavailable": False,
        "rebuilt_from_queue": True,
        "rebuilt_from_preflight": False,
        "rebuilt_from_pipeline_failure": False,
        "rebuilt_source_status": "completed",
        "learning_signal_strength": "medium",
        "repeated_meal_names": ["Arroz con pollo", "Habichuelas"],
        "repeated_bases": ["arroz", "pollo"],
        "allergy_hits": ["maní"],
        "rejected_meals_that_reappeared": ["Pollo guisado"],
    }
    client = _client()
    with patch("db_core.execute_sql_query", return_value=_row_with_lcl(lcl)):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    body = r.json()
    out = body["last_chunk_learning"]
    assert isinstance(out, dict)
    assert out["chunk"] == 2
    assert out["repeat_pct"] == pytest.approx(0.18)
    assert out["ingredient_base_repeat_pct"] == pytest.approx(0.42)
    assert out["allergy_violations"] == 1
    assert out["low_confidence"] is True
    assert out["metrics_unavailable"] is False
    assert out["rebuilt_from_queue"] is True
    assert out["timestamp"] == "2026-05-09T12:34:56Z"
    assert out["learning_signal_strength"] == "medium"
    assert out["repeated_meal_names"] == ["Arroz con pollo", "Habichuelas"]
    assert out["allergy_hits"] == ["maní"]


def test_legacy_plan_without_key_returns_none():
    """Plan archivado pre-aprendizaje continuo: plan_data sin la key.
    El SELECT devuelve None para esa columna; el endpoint debe pasar
    None directo (NO {} con todas las keys nulled)."""
    client = _client()
    with patch("db_core.execute_sql_query", return_value=_row_with_lcl(None)):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    assert r.json()["last_chunk_learning"] is None


def test_corrupted_lcl_type_falls_to_none():
    """LCL llega como lista (corrupción jsonb) → None defensivo."""
    client = _client()
    with patch("db_core.execute_sql_query", return_value=_row_with_lcl([1, 2, 3])):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    assert r.json()["last_chunk_learning"] is None


def test_numeric_keys_reject_bool_values():
    """Python `isinstance(True, int)` es True. La whitelist numeric debe
    excluir bools — sino `chunk: True` se renderizaría como "1" en la
    UI, confundiendo al user. Sin este check, un escritor bug-y del
    cron contaminaría el chip."""
    lcl = {
        "chunk": True,  # bool, NO numeric — debe caer a None
        "repeat_pct": 0.5,  # válido
        "rejection_violations": False,  # bool — None
    }
    client = _client()
    with patch("db_core.execute_sql_query", return_value=_row_with_lcl(lcl)):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    out = r.json()["last_chunk_learning"]
    assert out["chunk"] is None
    assert out["repeat_pct"] == pytest.approx(0.5)
    assert out["rejection_violations"] is None


def test_str_keys_reject_empty_or_whitespace():
    """str whitelist debe rechazar empty + whitespace-only — render
    `Origen: ` con valor vacío sería ruido."""
    lcl = {
        "rebuilt_source_status": "   ",  # whitespace → None
        "learning_signal_strength": "",  # empty → None
        "timestamp": "2026-05-09T00:00:00Z",  # válido
    }
    client = _client()
    with patch("db_core.execute_sql_query", return_value=_row_with_lcl(lcl)):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    out = r.json()["last_chunk_learning"]
    assert out["rebuilt_source_status"] is None
    assert out["learning_signal_strength"] is None
    assert out["timestamp"] == "2026-05-09T00:00:00Z"


def test_list_keys_coerce_to_string_and_drop_nulls():
    """list whitelist: items None se descartan, otros se castean a str
    (mirror del patrón usado para `permanent_meal_blocklist` en summary)."""
    lcl = {
        "repeated_meal_names": ["Pollo", None, 42, "Arroz"],
        "allergy_hits": "no es lista",  # corrupción → []
    }
    client = _client()
    with patch("db_core.execute_sql_query", return_value=_row_with_lcl(lcl)):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    out = r.json()["last_chunk_learning"]
    assert out["repeated_meal_names"] == ["Pollo", "42", "Arroz"]
    assert out["allergy_hits"] == []


def test_bool_keys_reject_non_bool():
    """`low_confidence: 1` (int truthy) NO debe interpretarse como True
    — el productor SOLO escribe bool en estas keys, un int es bug del
    writer y no debe propagarse silenciosamente al user."""
    lcl = {
        "low_confidence": 1,  # int — None defensivo
        "metrics_unavailable": "true",  # str — None
        "rebuilt_from_queue": True,  # válido
    }
    client = _client()
    with patch("db_core.execute_sql_query", return_value=_row_with_lcl(lcl)):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    out = r.json()["last_chunk_learning"]
    assert out["low_confidence"] is None
    assert out["metrics_unavailable"] is None
    assert out["rebuilt_from_queue"] is True


# ---------------------------------------------------------------------------
# 3. Backward compat: la nueva key NO rompe el shape existente
# ---------------------------------------------------------------------------
def test_existing_three_keys_still_present_with_new_key():
    """Smoke test del shape canónico existente: el endpoint sigue
    devolviendo summary + history + critical_permanent + counts +
    plan_id, ahora con last_chunk_learning añadido."""
    row = {
        "summary": {"total_rejection_violations": 1},
        "history": [{"chunk": 1}],
        "critical_permanent": [{"allergy_violations": 1}],
        "last_chunk_learning": {"chunk": 1, "low_confidence": False},
    }
    client = _client()
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    body = r.json()
    for k in ("plan_id", "summary", "history", "critical_permanent",
              "last_chunk_learning", "counts"):
        assert k in body
