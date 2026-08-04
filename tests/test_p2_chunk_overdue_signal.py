"""[P2-CHUNK-OVERDUE-SIGNAL] Predicado SSOT + payload de /chunk-status + cron + coach."""
from datetime import date
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import chat_history_context as chc


def _plan(days_dates, total):
    return {"total_days_requested": total,
            "days": [{"date": d, "meals": []} for d in days_dates]}


def test_overdue_cuando_hoy_supera_el_ultimo_dia_y_nada_corre():
    plan = _plan(["2026-08-01", "2026-08-02", "2026-08-03"], 15)
    overdue, since = chc.compute_chunk_overdue(plan, in_flight_count=0, today=date(2026, 8, 4))
    assert overdue is True and since == "2026-08-04"


def test_no_overdue_si_algo_esta_corriendo():
    plan = _plan(["2026-08-01", "2026-08-02", "2026-08-03"], 15)
    assert chc.compute_chunk_overdue(plan, in_flight_count=1, today=date(2026, 8, 4)) == (False, None)


def test_no_overdue_si_hoy_es_el_ultimo_dia_generado():
    plan = _plan(["2026-08-02", "2026-08-03", "2026-08-04"], 15)
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 4)) == (False, None)


def test_no_overdue_si_el_plan_ya_entrego_todos_los_dias():
    plan = _plan(["2026-08-01", "2026-08-02", "2026-08-03"], 3)
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 9)) == (False, None)


def test_fail_open_plan_legacy_sin_dates():
    plan = {"total_days_requested": 15,
            "days": [{"day_name": "Lunes", "meals": []}, {"day_name": "Martes", "meals": []}]}
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 9)) == (False, None)


def test_fail_open_ante_basura():
    assert chc.compute_chunk_overdue({"days": "no-una-lista"}, 0, today=date(2026, 8, 9)) == (False, None)


# ---------------------------------------------------------------------------
# Payload de /chunk-status — mismo patrón (TestClient + dependency_overrides +
# patch("db_core.execute_sql_query", side_effect=...) por-query) que
# tests/test_p2_hist_new_3_chunk_days_range_payload.py y
# tests/test_p1_audit_hist_6_tier_breakdown.py (hermanos reales del endpoint
# que sí invocan `api_chunk_status` vía TestClient; test_p0_dash_chip_honesty
# es whitebox/source-inspection puro y no monta un fixture de request).
# ---------------------------------------------------------------------------

_USER_A = "11111111-1111-1111-1111-111111111111"
_PLAN_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_CHUNK_A = "cccccccc-cccc-cccc-cccc-cccccccccccc"


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _fake_chunk_status_queries(plan_data, upcoming_rows):
    """Dispatcher por-query: cada SELECT del endpoint tiene una firma
    textual distinta que permite enrutar sin tocar orden de invocación."""
    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"user_id": _USER_A, "plan_data": plan_data}
        if "id::text AS chunk_id" in query and "status IN ('pending', 'processing')" in query:
            # [P2-CHUNK-OVERDUE-SIGNAL] query nueva de upcoming_chunks.
            return upcoming_rows
        if "SELECT execute_after FROM plan_chunk_queue" in query:
            return None  # next_chunk_eta (fetch_one), sin next chunk aquí
        if "status = 'pending_user_action'" in query:
            return []  # paused_rows
        if "COUNT(*) FILTER" in query:
            return {
                "in_flight_count": 0,
                "pending_user_action_count": 0,
                "failed_count": 0,
                "completed_count": 3,
            }
        if "FROM user_profiles" in query:
            return None
        if "quality_tier, COUNT" in query:
            return []  # tier_breakdown
        return None
    return _fake


def _plan_3_dias_viejo():
    return {
        "total_days_requested": 15,
        "days": [
            {"date": "2026-08-01", "meals": []},
            {"date": "2026-08-02", "meals": []},
            {"date": "2026-08-03", "meals": []},
        ],
    }


def _upcoming_rows_1_pending():
    return [{
        "chunk_id": _CHUNK_A,
        "week_number": 2,
        "days_offset": 3,
        "days_count": 4,
        "status": "pending",
        "execute_after": None,
    }]


def test_chunk_status_expone_upcoming_y_overdue(monkeypatch):
    """Plan de 3 días con dates viejas, total 15, cola con 1 chunk pending
    → upcoming_chunks con {chunk_id, week_number, days_offset, days_count,
    status, execute_after} → overdue True (in_flight_count del
    counters_row = 0)."""
    # rd_today() fijo para determinismo — el predicado usa `last_day + 1`
    # para `overdue_since`, no `today`, pero fijamos igual para que el
    # test no dependa del reloj de la máquina que lo corre.
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    plan_data = _plan_3_dias_viejo()
    upcoming_rows = _upcoming_rows_1_pending()

    client = _build_test_client()
    from auth import get_verified_user_id
    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch(
        "db_core.execute_sql_query",
        side_effect=_fake_chunk_status_queries(plan_data, upcoming_rows),
    ):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["overdue"] is True
    assert body["overdue_since"] == "2026-08-04"
    assert body["upcoming_chunks"] == upcoming_rows


def test_chunk_status_knob_off_no_manda_los_campos(monkeypatch):
    monkeypatch.setenv("MEALFIT_UPCOMING_DAYS_UI", "false")
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    plan_data = _plan_3_dias_viejo()
    upcoming_rows = _upcoming_rows_1_pending()

    client = _build_test_client()
    from auth import get_verified_user_id
    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch(
        "db_core.execute_sql_query",
        side_effect=_fake_chunk_status_queries(plan_data, upcoming_rows),
    ):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert "upcoming_chunks" not in body
    assert "overdue" not in body
    assert "overdue_since" not in body
