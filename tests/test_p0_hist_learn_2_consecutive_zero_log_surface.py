"""[P0-HIST-LEARN-2 · 2026-05-09] Surface de
``_consecutive_zero_log_chunks`` en ``/lifetime-lessons`` y
``/history-list``.

Bug original (audit Historial 2026-05-09 · gap P0):
    El counter `_consecutive_zero_log_chunks` (en `meal_plans.plan_data`)
    es bumpeado por `cron_tasks.py:_bump_zero_log` cada vez que un
    rolling_refill corre sin signal del usuario (sin consumed_meals ni
    interacciones). A partir de ≥3 (cron_tasks.py:17487) el cron:

      1. Flippea `generation_status` a 'degraded_pending_engagement'.
      2. Dispara push notification con copy alarmante ("Tu plan se está
         generando sin tu feedback").

    Pero el modal del Historial NO lo surfaceaba — un user que recibió
    el push y abrió el Historial no encontraba forma de verificar
    "¿este plan se generó sin mi feedback?". El plan real
    `98d902e3-56f0-4f54-a4f6-cb454b23d4de` tiene
    `_consecutive_zero_log_chunks=1` ahora mismo, completamente
    invisible al user.

Fix:
    `/lifetime-lessons` extiende SELECT a la key + `generation_status`
    para que el modal pueda diferenciar "1-2 (info)" de "≥3 + degradado
    (alarm)". `/history-list` extiende projection a la key para que la
    card del listado muestre el chip SIN abrir el modal.

Cobertura:
    - SELECT extrae la key + generation_status (single roundtrip).
    - Coerción defensiva: NULLIF + ::int en SQL cubre key ausente
      o string vacío. Python valida isinstance(int).
    - Plan legacy sin la key → consecutive_zero_log_chunks: None.
    - Plan con counter degraded → response trae generation_status para
      severity tiering frontend.
    - /history-list devuelve la key como int o None.
    - Backward-compat shape: las keys existentes siguen presentes.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "44444444-4444-4444-4444-444444444444"
_PLAN_ID = "12121212-1212-1212-1212-121212121212"


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
# 1. /lifetime-lessons — SELECT + response shape
# ---------------------------------------------------------------------------
def test_marker_present_in_lifetime_endpoint():
    from routers.plans import api_plan_lifetime_lessons
    src = inspect.getsource(api_plan_lifetime_lessons)
    assert "P0-HIST-LEARN-2" in src


def test_select_extracts_zero_log_counter_and_status():
    from routers.plans import api_plan_lifetime_lessons
    src = inspect.getsource(api_plan_lifetime_lessons)
    assert "_consecutive_zero_log_chunks" in src
    assert "generation_status" in src


def _row_with_zero_log(czl=None, gen_status=None):
    return {
        "summary": None, "history": None, "critical_permanent": None,
        "last_chunk_learning": None,
        "consecutive_zero_log_chunks": czl,
        "generation_status": gen_status,
    }


def test_response_includes_keys_when_value_present():
    """Counter+status canónicos: ambas keys en response, valor preservado."""
    client = _client()
    row = _row_with_zero_log(czl=2, gen_status="active")
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    assert r.status_code == 200
    body = r.json()
    assert body["consecutive_zero_log_chunks"] == 2
    assert body["generation_status"] == "active"


def test_response_handles_degraded_status():
    """Plan con counter ≥3 + status degraded — el frontend escala el
    chip a 'alarm' por la combinación de ambos."""
    client = _client()
    row = _row_with_zero_log(czl=3, gen_status="degraded_pending_engagement")
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    body = r.json()
    assert body["consecutive_zero_log_chunks"] == 3
    assert body["generation_status"] == "degraded_pending_engagement"


def test_legacy_plan_without_keys_returns_nulls():
    """Plan archivado pre-engagement-tracking: keys ausentes → None
    sin romper response."""
    client = _client()
    with patch("db_core.execute_sql_query", return_value=_row_with_zero_log()):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    body = r.json()
    assert body["consecutive_zero_log_chunks"] is None
    assert body["generation_status"] is None


def test_corrupted_counter_type_falls_to_none():
    """Si el SELECT devuelve un str (driver bug, no canónico — el SQL
    NULLIF + ::int debería castear), Python coerce defensivo a None."""
    client = _client()
    row = _row_with_zero_log(czl="3", gen_status="active")  # str en lugar de int
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    body = r.json()
    assert body["consecutive_zero_log_chunks"] is None
    assert body["generation_status"] == "active"


def test_corrupted_status_empty_string_falls_to_none():
    """generation_status whitespace-only → None (el frontend NO debe
    confundir empty con un status real)."""
    client = _client()
    row = _row_with_zero_log(czl=1, gen_status="   ")
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    body = r.json()
    assert body["generation_status"] is None


def test_lifetime_response_keeps_existing_keys_intact():
    """La nueva key NO desplaza/oculta las 5 keys canónicas existentes."""
    client = _client()
    row = {
        "summary": {"total_rejection_violations": 1},
        "history": [{"chunk": 1}],
        "critical_permanent": [{"allergy_violations": 1}],
        "last_chunk_learning": {"chunk": 1, "low_confidence": False},
        "consecutive_zero_log_chunks": 2,
        "generation_status": "active",
    }
    with patch("db_core.execute_sql_query", return_value=row):
        r = client.get(f"/api/plans/{_PLAN_ID}/lifetime-lessons")
    body = r.json()
    for k in ("plan_id", "summary", "history", "critical_permanent",
              "last_chunk_learning", "counts",
              "consecutive_zero_log_chunks", "generation_status"):
        assert k in body, f"key '{k}' missing from response"


# ---------------------------------------------------------------------------
# 2. /history-list — projection + response
# ---------------------------------------------------------------------------
def test_marker_present_in_history_list_endpoint():
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "P0-HIST-LEARN-2" in src


def test_history_list_select_extracts_counter():
    """SELECT extiende projection con la key; sigue extrayendo solo
    keys necesarias (no `select *` del jsonb)."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "_consecutive_zero_log_chunks" in src


def test_history_list_response_propagates_counter():
    """Card del listado recibe el counter por plan; el frontend pinta
    el chip sin abrir modal."""
    client = _client()
    rows = [
        {
            "id": _PLAN_ID, "name": "Test plan",
            "created_at": None, "calories": 2000, "macros": None,
            "plan_modified_at": None, "generation_status": None,
            "total_days_requested": 30, "days_generated": 7,
            "user_action_required": None, "recovery_exhausted_count": 0,
            "user_forced_simplified_weeks": None,
            "shift_days_accumulated": None,
            "consecutive_zero_log_chunks": 4,
            "coherence_history": [],
            "preview_meals_raw": [],
            "goal_root": None, "goal_assessment": None,
            "diet_root": None, "diet_assessment_snake": None,
            "diet_assessment_camel": None, "diet_assessment_type": None,
            "allergies": None,
            "chunk_pending_user_action_count": 0,
            "chunk_failed_count": 0,
            "chunk_failed_unreplaced_count": 0,
            "chunk_in_flight_count": 0,
            "chunk_completed_count": 0,
            "chunk_pantry_degraded_count": 0,
            "chunk_pantry_degraded_reasons": None,
            "chunk_tier_breakdown": None,
            "primary_action_reason_code": None,
        },
    ]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = client.get("/api/plans/history-list")
    assert r.status_code == 200
    body = r.json()
    assert len(body["plans"]) == 1
    assert body["plans"][0]["consecutive_zero_log_chunks"] == 4


def test_history_list_legacy_plan_returns_none():
    """Plan sin la key (ausente del SELECT) → response devuelve None."""
    client = _client()
    rows = [
        {
            "id": _PLAN_ID, "name": "Legacy plan",
            "created_at": None, "calories": None, "macros": None,
            "plan_modified_at": None, "generation_status": None,
            "total_days_requested": None, "days_generated": 0,
            "user_action_required": None, "recovery_exhausted_count": 0,
            "user_forced_simplified_weeks": None,
            "shift_days_accumulated": None,
            "consecutive_zero_log_chunks": None,  # key ausente
            "coherence_history": [],
            "preview_meals_raw": [],
            "goal_root": None, "goal_assessment": None,
            "diet_root": None, "diet_assessment_snake": None,
            "diet_assessment_camel": None, "diet_assessment_type": None,
            "allergies": None,
            "chunk_pending_user_action_count": 0,
            "chunk_failed_count": 0,
            "chunk_failed_unreplaced_count": 0,
            "chunk_in_flight_count": 0,
            "chunk_completed_count": 0,
            "chunk_pantry_degraded_count": 0,
            "chunk_pantry_degraded_reasons": None,
            "chunk_tier_breakdown": None,
            "primary_action_reason_code": None,
        },
    ]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = client.get("/api/plans/history-list")
    body = r.json()
    assert body["plans"][0]["consecutive_zero_log_chunks"] is None


# ---------------------------------------------------------------------------
# 3. Drift detection: threshold ≥3 (SSOT cron_tasks.py)
# ---------------------------------------------------------------------------
def test_threshold_ssot_in_cron():
    """El threshold ≥3 vive en cron_tasks.py (`>= 3` literal). El
    frontend lo hardcodea con cita explícita. Si alguien cambia el
    threshold del cron sin actualizar el frontend, el chip aparecería
    al threshold viejo. Este test asserta el literal ≥3 sigue presente
    en cron_tasks — falla si el cron cambia (alerta cross-language)."""
    import re
    with open("cron_tasks.py", encoding="utf-8") as f:
        src = f.read()
    # Dos call sites: _build_zero_log_push_payload + el bloque en
    # _record_zero_log que dispara push + flip de status. Ambos deben
    # comparar `>= 3`.
    matches = re.findall(r"_consecutive_zero_log_chunks?\s*\)?\s*>=\s*3\b", src)
    # Tolerante a variaciones de spacing; mínimo 1 occurrencia. El
    # patrón también matchea el bump in-memory del cron (línea 17526).
    assert len(matches) >= 1, (
        "Threshold ≥3 ya no aparece en cron_tasks.py — frontend hardcodea "
        "este valor con cita SSOT, debe actualizarse si el cron cambia."
    )
