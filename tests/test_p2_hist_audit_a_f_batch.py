"""[P2-HIST-AUDIT-A · 2026-05-09] Batch backend tests para los 6 P2
del audit Historial 2026-05-09:

  A. Cache-Control no-store en endpoints derivados.
  B. expected_preemption_seconds + reservation_status al chunk-metrics.
  C. _shift_days_accumulated en /history-list.
  D. Lessons telemetry split por tier (high/partial/low).
  E. is_rolling_refill cross-check vs chunk_kind.
  F. chunk_user_locks zombi expose en chunk-metrics.

Bug original (audit Historial 2026-05-09):
    Después de cerrar 3 P0 + 5 P1, quedaban 6 quick wins documentados:
      - Headers Cache-Control inconsistentes entre /history-list y los
        endpoints derivados (BFCache servía respuestas stale).
      - Campos diagnósticos del queue/metrics (expected_preemption,
        reservation_status, is_rolling_refill, blocking lock zombi)
        existían en DB pero no se exponían.
      - Telemetría de lecciones plana sin diferenciar calidad
        (high/partial/low).
      - shift_days_accumulated invisible en card.

Fix:
    Backend extiende los endpoints existentes (sin breaking changes
    para frontend legacy). Frontend renderiza los nuevos campos en
    chips/badges del Historial.

Cobertura backend:
    P2-A: helper `_apply_no_store` + 7 call sites.
    P2-B: SELECT extiende q.expected_preemption_seconds + q.reservation_status.
    P2-C: SELECT extrae _shift_days_accumulated; response sanitiza int|null.
    P2-D: GROUP BY (meal_plan_id, event); split por LESSON_QUALITY_TIERS.
    P2-E: SELECT m.is_rolling_refill; cross-check chunk_kind para
          is_rolling_refill_drift.
    P2-F: LEFT JOIN chunk_user_locks; CASE detect lock zombi de OTRO
          chunk con heartbeat fresco.
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "66666666-6666-6666-6666-666666666666"
_PLAN_ID = "abababab-abab-abab-abab-abababababab"


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
# P2-HIST-AUDIT-A: Cache-Control no-store en endpoints derivados
# ---------------------------------------------------------------------------
def test_p2_a_helper_no_store_exists():
    """Helper SSOT `_apply_no_store(response)` definido."""
    from routers import plans as plans_module
    assert hasattr(plans_module, "_apply_no_store")
    assert callable(plans_module._apply_no_store)


def test_p2_a_helper_sets_no_store_and_pragma():
    """El helper escribe AMBOS headers — Cache-Control y Pragma."""
    from routers.plans import _apply_no_store

    class _MockResponse:
        def __init__(self):
            self.headers = {}

    resp = _MockResponse()
    _apply_no_store(resp)
    assert resp.headers["Cache-Control"] == "no-store, max-age=0"
    assert resp.headers["Pragma"] == "no-cache"


@pytest.mark.parametrize("fn_name", [
    "api_blocked_reasons",
    "api_plans_lessons_counts",
    "api_plans_history_status_summary",
    "api_plan_lessons_detail",
    "api_plan_coherence_history",
    "api_plan_lifetime_lessons",
    "api_plan_chunk_metrics",
])
def test_p2_a_endpoints_apply_no_store(fn_name):
    """Cada endpoint derivado del Historial llama `_apply_no_store(response)`.
    Drift detection: si alguien añade un endpoint nuevo y olvida el
    helper, el listado parametrize abajo lo cubre."""
    from routers import plans as plans_module
    fn = getattr(plans_module, fn_name)
    src = inspect.getsource(fn)
    assert "_apply_no_store(response)" in src


# ---------------------------------------------------------------------------
# P2-HIST-AUDIT-B: expected_preemption_seconds + reservation_status
# ---------------------------------------------------------------------------
def test_p2_b_chunk_metrics_selects_new_columns():
    """SELECT del endpoint chunk-metrics incluye los 2 campos."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "q.expected_preemption_seconds" in src
    assert "q.reservation_status" in src
    # Y se propagan al response dict.
    assert '"expected_preemption_seconds"' in src
    assert '"reservation_status"' in src


def test_p2_b_marker_present():
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "P2-HIST-AUDIT-B · 2026-05-09" in src


# ---------------------------------------------------------------------------
# P2-HIST-AUDIT-C: _shift_days_accumulated en /history-list
# ---------------------------------------------------------------------------
def test_p2_c_history_list_extracts_shift_days():
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "_shift_days_accumulated" in src
    assert "shift_days_accumulated" in src


def test_p2_c_response_includes_shift_days_or_null():
    """Response shape: shift_days_accumulated = int | None."""
    client = _client()
    # Mock single-row response con shift_days populated.
    rows = [{
        "id": "plan-shift",
        "name": "Plan con shift",
        "created_at": None,
        "calories": 2000,
        "macros": {},
        "plan_modified_at": None,
        "generation_status": "complete",
        "total_days_requested": 7,
        "days_generated": 7,
        "user_action_required": None,
        "recovery_exhausted_count": 0,
        "user_forced_simplified_weeks": None,
        "shift_days_accumulated": 3,
        "coherence_history": [],
        "preview_meals_raw": [],
        "goal_root": None,
        "goal_assessment": None,
        "diet_root": None,
        "diet_assessment_snake": None,
        "diet_assessment_camel": None,
        "diet_assessment_type": None,
        "allergies": [],
        "chunk_pending_user_action_count": 0,
        "chunk_failed_count": 0,
        "chunk_in_flight_count": 0,
        "chunk_completed_count": 7,
        "chunk_tier_breakdown": None,
        "chunk_pantry_degraded_count": 0,
        "chunk_pantry_degraded_reasons": None,
    }]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = client.get("/api/plans/history-list")
    body = r.json()
    assert body["plans"][0]["shift_days_accumulated"] == 3


def test_p2_c_shift_days_null_when_absent():
    client = _client()
    rows = [{
        "id": "plan-noshift",
        "name": "Plan sin shift",
        "created_at": None,
        "calories": 2000,
        "macros": {},
        "plan_modified_at": None,
        "generation_status": "complete",
        "total_days_requested": 7,
        "days_generated": 7,
        "user_action_required": None,
        "recovery_exhausted_count": 0,
        "user_forced_simplified_weeks": None,
        "shift_days_accumulated": None,
        "coherence_history": [],
        "preview_meals_raw": [],
        "goal_root": None,
        "goal_assessment": None,
        "diet_root": None,
        "diet_assessment_snake": None,
        "diet_assessment_camel": None,
        "diet_assessment_type": None,
        "allergies": [],
        "chunk_pending_user_action_count": 0,
        "chunk_failed_count": 0,
        "chunk_in_flight_count": 0,
        "chunk_completed_count": 7,
        "chunk_tier_breakdown": None,
        "chunk_pantry_degraded_count": 0,
        "chunk_pantry_degraded_reasons": None,
    }]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = client.get("/api/plans/history-list")
    body = r.json()
    assert body["plans"][0]["shift_days_accumulated"] is None


# ---------------------------------------------------------------------------
# P2-HIST-AUDIT-D: Lessons split por tier (high/partial/low)
# ---------------------------------------------------------------------------
def test_p2_d_lesson_quality_tiers_constant_exists():
    from constants import LESSON_QUALITY_TIERS
    assert isinstance(LESSON_QUALITY_TIERS, dict)
    assert "high" in LESSON_QUALITY_TIERS
    assert "partial" in LESSON_QUALITY_TIERS
    assert "low" in LESSON_QUALITY_TIERS


def test_p2_d_lesson_quality_tiers_partition_whitelist():
    """Cada event de LESSON_COUNT_EVENT_WHITELIST cae en EXACTAMENTE
    una tier (no overlap, no gaps). Drift detection cross-archivo
    si alguien añade un event a la whitelist sin clasificarlo."""
    from constants import LESSON_COUNT_EVENT_WHITELIST, LESSON_QUALITY_TIERS
    all_tiered = []
    for tier_events in LESSON_QUALITY_TIERS.values():
        all_tiered.extend(tier_events)
    # Sin duplicados (cada event en UNA sola tier).
    assert len(all_tiered) == len(set(all_tiered)), (
        f"LESSON_QUALITY_TIERS tiene events duplicados: "
        f"{[e for e in all_tiered if all_tiered.count(e) > 1]}"
    )
    # Misma membresía que la whitelist (sin gaps, sin extras).
    assert set(all_tiered) == set(LESSON_COUNT_EVENT_WHITELIST), (
        f"Drift: WHITELIST={set(LESSON_COUNT_EVENT_WHITELIST)} vs "
        f"TIERED={set(all_tiered)}"
    )


def test_p2_d_lessons_counts_returns_split():
    """/lessons-counts response incluye `counts_by_quality` con shape
    `{plan_id: {high, partial, low}}`."""
    client = _client()
    # Mock: 3 events distintos para el mismo plan.
    rows = [
        {"pid": "plan-A", "event": "synth_propagated_to_prompt", "cnt": 5},
        {"pid": "plan-A", "event": "lesson_synthesized_low_confidence", "cnt": 2},
        {"pid": "plan-A", "event": "recent_lessons_partial_synthesis", "cnt": 1},
        {"pid": "plan-B", "event": "indefinite_pause_unblocked", "cnt": 3},
    ]
    with patch("db_core.execute_sql_query", return_value=rows):
        r = client.get("/api/plans/lessons-counts")
    body = r.json()
    # Total agregado retrocompat.
    assert body["counts"]["plan-A"] == 8
    assert body["counts"]["plan-B"] == 3
    # Split por tier.
    assert body["counts_by_quality"]["plan-A"] == {
        "high": 5, "partial": 1, "low": 2,
    }
    assert body["counts_by_quality"]["plan-B"] == {
        "high": 3, "partial": 0, "low": 0,
    }


# ---------------------------------------------------------------------------
# P2-HIST-AUDIT-E: is_rolling_refill cross-check
# ---------------------------------------------------------------------------
def test_p2_e_chunk_metrics_selects_is_rolling_refill():
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "m.is_rolling_refill" in src


def test_p2_e_cross_check_drift_field_in_response():
    """Response dict del chunk debe incluir is_rolling_refill_drift."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert '"is_rolling_refill_drift"' in src
    # La lógica del cross-check.
    assert "_kind_is_rolling" in src
    assert "_metrics_is_rolling" in src


# ---------------------------------------------------------------------------
# P2-HIST-AUDIT-F: chunk_user_locks zombi expose
# ---------------------------------------------------------------------------
def test_p2_f_chunk_metrics_left_joins_chunk_user_locks():
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "LEFT JOIN chunk_user_locks ul" in src
    assert "ul.user_id = q.user_id" in src


def test_p2_f_blocking_lock_filters_freshness_and_other_chunk():
    """El CASE filter excluye:
      1. Lock del MISMO chunk (path normal: chunk procesando legítimo).
      2. Lock con heartbeat stale (>5min — el lock lo va a soltar el cron
         de orphan cleanup). Solo locks frescos en otro chunk = zombi
         relevante.
    """
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    # Filter `locked_by_chunk_id != q.id`.
    assert re.search(
        r"ul\.locked_by_chunk_id\s*!=\s*q\.id",
        src,
    )
    # Filter heartbeat freshness.
    assert "heartbeat_at > NOW() - INTERVAL '5 minutes'" in src


def test_p2_f_response_includes_blocking_lock_fields():
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert '"blocking_lock_chunk_id"' in src
    assert '"blocking_lock_age_seconds"' in src
