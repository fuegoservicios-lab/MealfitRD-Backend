# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1001 · G-01 + G-06 · 2026-11-14] Observabilidad de modelos efectivos y fases por turno.

Verifica sin llamadas de red ni IA:
- G-06 (`model_visibility`): snapshot efectivo, detección de drift entre arranques, alerta
  en `system_alerts` (`model_effective_changed`), campos en `/health/version`, knob registrado
  en `knobs` y documentación actualizada en `docs/llm_tier_routing.md`.
- G-01 (`chat_phase_metrics`): tracker único por turno (`chat_turn_phases`), agregación de
  llamadas LLM y herramientas (incluyendo guard_blocked), emisión única (`emit_once`),
  telemetría del cliente (`chat_client_perf`), consulta p50/p90 (`query_chat_phase_percentiles`)
  y wiring en `/api/chat/stream`, `/api/chat/client-metrics` y `/admin/chat-perf`.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace
from unittest.mock import patch

import chat_phase_metrics as cpm
import knobs
import model_visibility as mv


BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent


def _snapshot_stub(**overrides):
    base = {
        "coach_model": "gpt-6-luna",
        "coach_fallback_model": "gemini-4-hydrogen",
        "vision_model": "gemini-4-hydrogen",
        "vision_provider": "openai_compatible",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# G-06 · model_visibility
# ---------------------------------------------------------------------------


def test_g06_knob_registered_in_knobs():
    mv.model_drift_alert_enabled()
    reg = knobs._KNOBS_REGISTRY
    assert "MEALFIT_MODEL_DRIFT_ALERT_ENABLED" in reg
    assert reg["MEALFIT_MODEL_DRIFT_ALERT_ENABLED"]["default"] is True


def test_g06_compute_model_drift_cases():
    curr = _snapshot_stub()
    assert mv.compute_model_drift(None, curr) == {}
    assert mv.compute_model_drift({}, curr) == {}
    assert mv.compute_model_drift({"recorded_at": "2026-11-14T00:00:00Z"}, curr) == {}
    assert mv.compute_model_drift(_snapshot_stub(), curr) == {}

    prev_drift = _snapshot_stub(coach_model="gemini-4-hydrogen")
    diff = mv.compute_model_drift(prev_drift, curr)
    assert diff == {
        "coach_model": {
            "previous": "gemini-4-hydrogen",
            "current": "gpt-6-luna",
        }
    }

    prev_vision = _snapshot_stub(vision_model="gpt-5.4")
    diff_vis = mv.compute_model_drift(prev_vision, curr)
    assert diff_vis == {
        "vision_model": {
            "previous": "gpt-5.4",
            "current": "gemini-4-hydrogen",
        }
    }


def test_g06_startup_drift_first_boot_and_no_drift_skips_alert():
    writes = []

    def fake_write(sql, params=None):
        writes.append((sql, params))
        return 1

    with (
        patch.object(mv, "get_effective_models_snapshot", return_value=_snapshot_stub()),
        patch("db_core.connection_pool", object()),
        patch("db_core.execute_sql_query", return_value=None),
        patch("db_core.execute_sql_write", side_effect=fake_write),
    ):
        res_first = mv.detect_and_record_model_drift_on_startup()

    assert res_first["checked"] is True
    assert res_first["drift_detected"] is False
    assert len(writes) == 1
    assert "app_kv_store" in writes[0][0]
    assert writes[0][1][0] == "effective_models_snapshot"

    writes.clear()
    with (
        patch.object(mv, "get_effective_models_snapshot", return_value=_snapshot_stub()),
        patch("db_core.connection_pool", object()),
        patch("db_core.execute_sql_query", return_value={"value": _snapshot_stub()}),
        patch("db_core.execute_sql_write", side_effect=fake_write),
    ):
        res_same = mv.detect_and_record_model_drift_on_startup()

    assert res_same["checked"] is True
    assert res_same["drift_detected"] is False
    assert len(writes) == 1
    assert "app_kv_store" in writes[0][0]


def test_g06_startup_drift_emits_system_alert_when_model_changes():
    writes = []

    def fake_write(sql, params=None):
        writes.append((sql, params))
        return 1

    prev = json.dumps(_snapshot_stub(coach_model="gemini-4-hydrogen"))
    with (
        patch.object(mv, "get_effective_models_snapshot", return_value=_snapshot_stub()),
        patch("db_core.connection_pool", object()),
        patch("db_core.execute_sql_query", return_value={"value": prev}),
        patch("db_core.execute_sql_write", side_effect=fake_write),
    ):
        res = mv.detect_and_record_model_drift_on_startup()

    assert res["checked"] is True
    assert res["drift_detected"] is True
    assert "coach_model" in res["changed"]
    assert len(writes) == 2
    assert "system_alerts" in writes[0][0]
    assert writes[0][1][0] == "model_effective_changed"
    assert writes[0][1][1] == "model_drift"
    assert "app_kv_store" in writes[1][0]

    fields = mv.get_health_version_model_fields()
    assert fields["model_drift_detected"] is True


def test_g06_startup_drift_fail_open_without_pool():
    with (
        patch.object(mv, "get_effective_models_snapshot", return_value=_snapshot_stub()),
        patch("db_core.connection_pool", None),
    ):
        res = mv.detect_and_record_model_drift_on_startup()

    assert res["checked"] is False
    assert res["drift_detected"] is False


def test_g06_docs_and_pfix_updated():
    docs = (BACKEND_DIR / "docs" / "llm_tier_routing.md").read_text(encoding="utf-8")
    assert "MEALFIT_CHAT_AGENT_MODEL=gpt-6-luna" in docs
    assert "gemini-4-hydrogen" in docs
    assert "model_effective_changed" in docs

    import app as app_mod

    assert app_mod._LAST_KNOWN_PFIX.startswith("P1-PLAN-LOTE-1001")


# ---------------------------------------------------------------------------
# G-01 · chat_phase_metrics
# ---------------------------------------------------------------------------


def test_g01_knob_registered_in_knobs():
    cpm.phase_metrics_enabled()
    reg = knobs._KNOBS_REGISTRY
    assert "MEALFIT_CHAT_PHASE_METRICS" in reg
    assert reg["MEALFIT_CHAT_PHASE_METRICS"]["default"] is True


def test_g01_tracker_single_emit_and_context_binding():
    writes = []

    def fake_write(sql, params=None):
        writes.append((sql, params))
        return 1

    tracker, tok = cpm.start_turn_tracker(
        user_id="guest",
        session_id="sess_g01_test",
        corr="corr_g01_abc",
        owned_by_router=True,
    )
    try:
        assert cpm.get_active_turn_tracker("sess_g01_test") is tracker
        tracker.set_phase_ms("preflight_ms", -50)
        tracker.set_phase_ms("sentiment_rag_router_ms", 120)
        tracker.set_phase_ms("rag_ms", 999_999)
        tracker.mark_context_ready()
        tracker.record_llm_call(
            model="gpt-6-luna",
            duration_ms=1450,
            fallback_used=False,
            input_tokens=33000,
            cached_tokens=0,
            output_tokens=45,
        )
        tracker.record_tool("log_consumed_meal", 85)
        tracker.record_tool("update_form_field", 12, guard_blocked=True)
        tracker.mark_first_word("stream_chunk")
        tracker.mark_first_word("stream_chunk")
        tracker.set_phase_ms("persist_ms", 42)
        tracker.mark_done(outcome="ok", model="gpt-6-luna")

        with patch("db_core.execute_sql_write", side_effect=fake_write):
            first = tracker.emit_once(outcome="ok", model_name="gpt-6-luna")
            second = tracker.emit_once(outcome="ok", model_name="gpt-6-luna")
    finally:
        cpm.release_turn_tracker(tracker, tok)

    assert cpm.get_active_turn_tracker("sess_g01_test") is None
    assert first is True
    assert second is False
    assert len(writes) == 1

    sql, params = writes[0]
    assert "INSERT INTO pipeline_metrics" in sql
    user_param, session_param, node_param, dur_param, meta_raw = params
    assert user_param is None  # guest se normaliza a NULL
    assert session_param == "sess_g01_test"
    assert node_param == "chat_turn_phases"
    assert isinstance(dur_param, int) and dur_param >= 0

    meta = json.loads(meta_raw)
    assert meta["corr"] == "corr_g01_abc"
    assert meta["preflight_ms"] == 0
    assert meta["sentiment_rag_router_ms"] == 120
    assert meta["rag_ms"] == 600_000
    assert meta["chunks_yielded"] == 2
    assert meta["first_word_source"] == "stream_chunk"
    assert meta["persist_ms"] == 42
    assert meta["llm_calls_count"] == 1
    assert meta["llm_total_ms"] == 1450
    assert meta["llm_calls"][0]["cached_tokens"] == 0
    assert meta["tools_count"] == 2
    assert meta["tools_total_ms"] == 97
    assert meta["tools"][1]["guard_blocked"] is True


def test_g01_extract_usage_counts():
    msg = SimpleNamespace(
        usage_metadata={
            "input_tokens": 1200,
            "output_tokens": 80,
            "input_token_details": {"cache_read": 1000},
        }
    )
    assert cpm.extract_usage_counts(msg) == (1200, 1000, 80)
    assert cpm.extract_usage_counts(SimpleNamespace()) == (None, None, None)


def test_g01_record_client_chat_perf_sanitizes_and_writes():
    writes = []

    def fake_write(sql, params=None):
        writes.append((sql, params))
        return 1

    with patch("db_core.execute_sql_write", side_effect=fake_write):
        res = cpm.record_client_chat_perf(
            user_id="11111111-1111-4111-8111-111111111111",
            session_id="sess_client_1",
            payload={
                "corr": "  corr_xyz_9  ",
                "first_chunk_ms": 940.7,
                "first_paint_ms": 1020.2,
                "stream_total_ms": 3400,
                "chunk_count": 14,
                "is_call_mode": False,
                "platform": "web",
            },
        )

    assert res == {"recorded": True, "corr": "corr_xyz_9"}
    assert len(writes) == 1
    user_param, session_param, node_param, primary_ms, meta_raw = writes[0][1]
    assert user_param == "11111111-1111-4111-8111-111111111111"
    assert session_param == "sess_client_1"
    assert node_param == "chat_client_perf"
    assert primary_ms == 1020
    meta = json.loads(meta_raw)
    assert meta["corr"] == "corr_xyz_9"
    assert meta["first_chunk_ms"] == 941
    assert meta["first_paint_ms"] == 1020
    assert meta["stream_total_ms"] == 3400
    assert meta["platform"] == "web"

    # Corr malicioso se descarta en lugar de romper el insert
    assert cpm.sanitize_corr("bad corr; DROP TABLE") is None


def test_g01_query_chat_phase_percentiles_sql_shape():
    queries = []

    def fake_query(sql, params=None, fetch_one=False, fetch_all=False):
        queries.append((sql, params, fetch_one, fetch_all))
        if fetch_all:
            return [{"tool_name": "log_consumed_meal", "calls": 5, "p50_ms": 80, "p90_ms": 140}]
        return {"turns": 5}

    with patch("db_core.execute_sql_query", side_effect=fake_query):
        out = cpm.query_chat_phase_percentiles(hours=72)

    assert out["window_hours"] == 72
    assert out["server"] == {"turns": 5}
    assert out["client"] == {"turns": 5}
    assert len(out["by_tool"]) == 1
    assert len(queries) == 3
    # Verificar casts explícitos para PostgreSQL y guardia de jsonb_typeof
    assert "::numeric)::int" in queries[0][0]
    assert "::numeric)::int" in queries[1][0]
    assert "jsonb_typeof(metadata->'tools') = 'array'" in queries[2][0]


def test_g01_and_g06_endpoints_and_wiring_present():
    from fastapi.testclient import TestClient
    import app as app_mod

    route_paths = {getattr(r, "path", None) for r in app_mod.app.routes}
    assert "/health/version" in route_paths
    assert any(p and p.endswith("/chat/client-metrics") for p in route_paths)
    assert any(p and p.endswith("/chat-perf") for p in route_paths)

    client = TestClient(app_mod.app)
    with patch.object(
        mv,
        "get_health_version_model_fields",
        return_value={
            "coach_model": "gpt-6-luna",
            "coach_fallback_model": "gemini-4-hydrogen",
            "vision_model": "gemini-4-hydrogen",
            "model_drift_detected": False,
        },
    ):
        resp = client.get("/health/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["coach_model"] == "gpt-6-luna"
    assert body["coach_fallback_model"] == "gemini-4-hydrogen"
    assert body["vision_model"] == "gemini-4-hydrogen"
    assert body["model_drift_detected"] is False

    # Agent source wiring check (corr en evento done y mark_first_word)
    agent_src = (BACKEND_DIR / "agent.py").read_text(encoding="utf-8")
    assert "'corr': _pt.corr" in agent_src
    assert "_pt.mark_first_word" in agent_src
    assert "_pt_call.record_llm_call" in agent_src
    assert "_pt_tools.record_tool" in agent_src

    # Endpoints protegidos: UUID inválido -> 400, sesión ajena -> 403, /admin/chat-perf sin token -> 401/403/503
    bad_uuid_resp = client.post(
        "/api/chat/client-metrics",
        json={"session_id": "not-a-uuid", "first_chunk_ms": 100},
    )
    assert bad_uuid_resp.status_code == 400
    with patch("routers.chat.get_session_owner", return_value="22222222-2222-4222-8222-222222222222"):
        idor_resp = client.post(
            "/api/chat/client-metrics",
            json={"session_id": "11111111-1111-4111-8111-111111111111", "first_chunk_ms": 100},
        )
    assert idor_resp.status_code == 403
    assert "/api/system/admin/chat-perf" in route_paths
    unauth_admin_perf = client.get("/api/system/admin/chat-perf")
    assert unauth_admin_perf.status_code in (401, 403, 503)
