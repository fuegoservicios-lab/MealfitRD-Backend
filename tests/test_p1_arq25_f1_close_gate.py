"""[P1-ARQ25-F1-CLOSE · 2026-09-02] El gate de la Fase 1 se LEE, no se supone.

`GET /api/system/admin/arq25-gate` (admin) mide en DB lo que el roadmap pide para el flip:
runs por la cola, duplicados, commits stale (`fencing_rejected`), `pending_pipeline`, kills
recuperados, alertas desde el canary y días sin alerta nueva (el soak de 7 días es
observación, no código; aquí se cuenta).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
_SYS = (BACKEND / "routers" / "system.py").read_text(encoding="utf-8")


def _fake_query(rows):
    def q(sql, params=None, fetch_one=False, fetch_all=False, **kw):
        for key, val in rows:
            if key in sql:
                return val
        return {} if fetch_one else []
    return q


def _rows(*, runs=10, dup_runs=0, dup_plans=0, fencing=0, pending=0, kills=2, alerts=None, since=None):
    since = since or datetime.now(timezone.utc) - timedelta(days=8)
    return [
        ("min(created_at) AS t FROM plan_generation_runs", {"t": since}),
        ("count(DISTINCT user_id) AS u", {"n": runs, "u": 1}),
        ("jsonb_array_length(coalesce(mp.plan_data->'days'", {"n": runs}),
        ("GROUP BY plan_id HAVING count(*) > 1", {"n": dup_runs}),
        ("HAVING count(DISTINCT plan_id) > 1", {"n": dup_plans}),
        ("node = 'arq25_fencing_rejected'", {"n": fencing}),
        ("status = 'pending_pipeline'", {"n": pending}),
        ("coalesce(attempts, 0) >= 1", {"n": kills}),
        ("FROM system_alerts WHERE triggered_at >= %s", alerts or []),
    ]


def test_gate_ready_when_counts_ok_and_soak_elapsed():
    sysmod = pytest.importorskip("routers.system")
    out = sysmod.arq25_gate_status(_fake_query(_rows()))
    assert out["counts_ok"] is True and out["soak_ok"] is True and out["ready_to_flip"] is True
    assert out["runs"] == 10 and out["kills_recovered"] == 2 and out["soak_days_required"] == 7
    assert out["days_since_last_lifecycle_alert"] >= 7.9


def test_gate_not_ready_before_soak_even_with_perfect_counts():
    sysmod = pytest.importorskip("routers.system")
    out = sysmod.arq25_gate_status(_fake_query(_rows(since=datetime.now(timezone.utc) - timedelta(hours=6))))
    assert out["counts_ok"] is True and out["soak_ok"] is False and out["ready_to_flip"] is False


def test_lifecycle_alert_resets_the_soak_but_quality_alert_does_not():
    sysmod = pytest.importorskip("routers.system")
    now = datetime.now(timezone.utc)
    quality = [{"alert_key": "plan_quality_degraded:u:p", "severity": "warning", "triggered_at": now - timedelta(hours=1)}]
    out = sysmod.arq25_gate_status(_fake_query(_rows(alerts=quality)))
    assert out["alerts_since_canary_count"] == 1 and out["lifecycle_alerts_since_canary"] == 0
    assert out["days_since_last_alert_any"] < 1 and out["soak_ok"] is True, "una alerta de calidad no es del lifecycle"
    lifecycle = [{"alert_key": "zombie_chunk_rescued:x", "severity": "warning", "triggered_at": now - timedelta(days=2)}]
    out2 = sysmod.arq25_gate_status(_fake_query(_rows(alerts=lifecycle)))
    assert out2["lifecycle_alerts_since_canary"] == 1 and out2["soak_ok"] is False


@pytest.mark.parametrize("field,value", [("dup_runs", 1), ("dup_plans", 1), ("fencing", 1), ("pending", 1), ("kills", 1), ("runs", 9)])
def test_each_count_can_block_the_gate(field, value):
    sysmod = pytest.importorskip("routers.system")
    out = sysmod.arq25_gate_status(_fake_query(_rows(**{field: value})))
    assert out["counts_ok"] is False and out["ready_to_flip"] is False


def test_endpoint_is_admin_gated_and_reads_from_db_facade():
    i = _SYS.find('@router.get("/admin/arq25-gate")')
    assert i != -1
    body = _SYS[i:i + 600]
    assert '_verify_admin_token(request.headers.get("authorization"))' in body
    assert "from db import execute_sql_query" in body and "arq25_gate_status(execute_sql_query)" in body
