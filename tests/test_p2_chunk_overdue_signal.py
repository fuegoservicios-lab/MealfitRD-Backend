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


# ---------------------------------------------------------------------------
# Cron horario `_chunk_overdue_alert_job` (Task 2) — mismo patrón de
# monkeypatch por-función sobre `ct.execute_sql_query` / `ct.execute_sql_write`
# (side_effect posicional consumido con un iterator) que
# tests/test_p3_b_coherence_block_metrics_cron.py, el cron hermano más simple.
# ---------------------------------------------------------------------------

_CRON_PLAN_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_CRON_USER_ID = "11111111-1111-1111-1111-111111111111"


def _cron_overdue_plan_row(plan_id=_CRON_PLAN_ID, user_id=_CRON_USER_ID):
    """Plan con 3 días de dates viejas y total=15: overdue si hoy > 2026-08-03
    y la cola in-flight está vacía."""
    return {
        "plan_id": plan_id,
        "user_id": user_id,
        "plan_data": {
            "total_days_requested": 15,
            "days": [
                {"date": "2026-08-01", "meals": []},
                {"date": "2026-08-02", "meals": []},
                {"date": "2026-08-03", "meals": []},
            ],
        },
    }


def _cron_complete_plan_row(plan_id=_CRON_PLAN_ID, user_id=_CRON_USER_ID):
    """Mismo plan pero ya completo (total_days_requested == len(days)) →
    compute_chunk_overdue debe devolver (False, None)."""
    days = [{"date": f"2026-08-0{i + 1}", "meals": []} for i in range(3)]
    return {"plan_id": plan_id, "user_id": user_id,
            "plan_data": {"total_days_requested": 3, "days": days}}


def test_cron_overdue_upsertea_y_autoresuelve(monkeypatch):
    """Corrida 1: 1 plan overdue (dates viejas, total 15, cola vacía) → upsert
    con alert_key='chunk_overdue:<id>'. Corrida 2 con el plan ya completo
    (total == len(days)) → UPDATE resolved_at (auto-resuelve, modelo Auto
    implicit: el job re-emite mientras la condición exista)."""
    import cron_tasks as ct
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    captured = []

    def _fake_write(sql, params=None, **kwargs):
        captured.append({"sql": sql, "params": params})
        return None

    monkeypatch.setattr(ct, "execute_sql_write", _fake_write)

    # --- Corrida 1: plan overdue, cola in-flight vacía (count=0) ---
    def _fake_query_overdue(sql, params=None, **kwargs):
        if "FROM meal_plans" in sql:
            return [_cron_overdue_plan_row()]
        if "FROM plan_chunk_queue" in sql:
            return {"c": 0}
        raise AssertionError(f"query inesperada: {sql}")

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query_overdue)
    ct._chunk_overdue_alert_job()

    assert len(captured) == 1, captured
    upsert = captured[0]
    assert "INSERT INTO system_alerts" in upsert["sql"]
    assert "ON CONFLICT" in upsert["sql"]
    assert upsert["params"][0] == f"chunk_overdue:{_CRON_PLAN_ID}"

    # --- Corrida 2: mismo plan ahora completo → resolve ---
    captured.clear()

    def _fake_query_complete(sql, params=None, **kwargs):
        if "FROM meal_plans" in sql:
            return [_cron_complete_plan_row()]
        if "FROM plan_chunk_queue" in sql:
            return {"c": 0}
        raise AssertionError(f"query inesperada: {sql}")

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query_complete)
    ct._chunk_overdue_alert_job()

    assert len(captured) == 1, captured
    resolve = captured[0]
    assert "UPDATE system_alerts" in resolve["sql"]
    assert "resolved_at = NOW()" in resolve["sql"]
    assert resolve["params"][0] == f"chunk_overdue:{_CRON_PLAN_ID}"


def test_cron_overdue_fail_open_por_plan(monkeypatch):
    """El primer plan lanza excepción en su COUNT in-flight; el segundo es
    overdue. El job NO debe abortar: el segundo plan recibe su alerta igual
    (fail-open POR PLAN, no fail-open global)."""
    import cron_tasks as ct
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    _plan_boom_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    plan_boom = _cron_overdue_plan_row(plan_id=_plan_boom_id)
    plan_ok = _cron_overdue_plan_row(plan_id=_CRON_PLAN_ID)

    captured = []

    def _fake_write(sql, params=None, **kwargs):
        captured.append({"sql": sql, "params": params})
        return None

    _count_results = iter([RuntimeError("boom: COUNT falló"), {"c": 0}])

    def _fake_query(sql, params=None, **kwargs):
        if "FROM meal_plans" in sql:
            return [plan_boom, plan_ok]
        if "FROM plan_chunk_queue" in sql:
            nxt = next(_count_results)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        raise AssertionError(f"query inesperada: {sql}")

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query)
    monkeypatch.setattr(ct, "execute_sql_write", _fake_write)

    ct._chunk_overdue_alert_job()  # no debe lanzar

    assert len(captured) == 1, captured
    assert captured[0]["params"][0] == f"chunk_overdue:{_CRON_PLAN_ID}"


def test_alert_key_documentada_en_la_tabla():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")
    assert "chunk_overdue:" in doc
