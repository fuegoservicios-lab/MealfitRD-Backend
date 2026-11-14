"""[P1-PLAN-LOTE-1011 · 2026-11-14] P2/P3 Observability, Parity & Idempotency:
- G-16: Feedback negativo de usuarios en `/admin/chat-insights` y `?include_insights=true`
- G-27: Rendimiento de nudges internos y proactivos (`nudge_outcomes`)
- G-29: Cola lenta (>15s) con desglose por fase
- G-26: Paridad contractual entre voz live/voz TTS y texto (`coach_live` ↔ `prompts/chat_agent`)
- G-28: Idempotencia ante reintentos (`log_consumed_meal`, `mark_shopping_list_purchased`, `client_message_id`)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# ---------------------------------------------------------------------------
# G-16 / G-27 / G-29: query_chat_ops_insights + endpoints admin
# ---------------------------------------------------------------------------

def test_g16_g27_g29_query_chat_ops_insights_and_admin_endpoints(monkeypatch):
    """`query_chat_ops_insights` consulta feedback negativo, turnos >15s y nudges,
    y `/admin/chat-insights` + `/admin/chat-perf?include_insights=true` exigen admin."""
    import chat_phase_metrics as cpm
    import db_core
    from routers import system as sys_router

    captured_calls: list[tuple[str, tuple]] = []

    def _fake_sql(sql: str, params=(), fetch_all=False, fetch_one=False):
        captured_calls.append((sql, tuple(params)))
        if "FROM public.agent_messages" in sql:
            return [
                {
                    "message_id": "m-neg-1",
                    "session_id": "s-1",
                    "user_id": "u-1",
                    "preview": "No detectó el pan ni las kcal reales.",
                    "created_at": "2026-11-14T08:00:00Z",
                }
            ]
        if "duration_ms > 15000" in sql:
            return [
                {
                    "corr": "corr-slow-1",
                    "model": "gpt-6-luna",
                    "outcome": "ok",
                    "duration_ms": 18420,
                    "context_build_ms": 320,
                    "first_word_ms": 1950,
                    "llm_total_ms": 15200,
                    "tools_total_ms": 840,
                    "llm_calls_count": 3,
                    "created_at": "2026-11-14T07:50:00Z",
                }
            ]
        if "jsonb_array_elements_text(metadata->'nudges')" in sql:
            return [
                {
                    "nudge_name": "nudge_diary_tool",
                    "turns": 12,
                    "ok_rate": 0.917,
                    "avg_llm_calls": 2.42,
                }
            ]
        if "FROM public.nudge_outcomes" in sql:
            return [
                {
                    "nudge_type": "dinner_reminder",
                    "sent": 10,
                    "responded": 4,
                    "meal_logged": 3,
                    "response_rate": 0.4,
                    "meal_log_rate": 0.3,
                }
            ]
        return []

    monkeypatch.setattr(db_core, "execute_sql_query", _fake_sql)

    out = cpm.query_chat_ops_insights(hours=9999, limit=500)
    assert out["window_hours"] == 720
    assert out["limit"] == 100
    assert len(out["negative_feedback"]) == 1
    assert out["negative_feedback"][0]["message_id"] == "m-neg-1"
    assert len(out["slow_turns"]) == 1
    assert out["slow_turns"][0]["duration_ms"] == 18420
    assert out["slow_turns"][0]["context_build_ms"] == 320
    assert out["nudge_outcomes"][0]["ok_rate"] == pytest.approx(0.917)
    assert out["proactive_nudge_outcomes"][0]["meal_log_rate"] == pytest.approx(0.3)
    assert any("feedback = 'down'" in s for s, _ in captured_calls)
    assert any("duration_ms > 15000" in s for s, _ in captured_calls)

    # Endpoint protection: missing/invalid admin auth fails closed before running query
    with pytest.raises(HTTPException) as exc_info:
        sys_router.admin_chat_insights(SimpleNamespace(headers={}), hours=24, limit=10)
    assert exc_info.value.status_code in (401, 403, 503)

    monkeypatch.setattr(sys_router, "_verify_admin_token", lambda auth: None)
    monkeypatch.setattr(sys_router, "_check_admin_rate_limit", lambda req: None)

    resp_insights = sys_router.admin_chat_insights(
        SimpleNamespace(headers={"authorization": "Bearer ok"}), hours=48, limit=5
    )
    assert resp_insights["success"] is True
    assert resp_insights["window_hours"] == 48
    assert resp_insights["limit"] == 5
    assert resp_insights["slow_turns"][0]["corr"] == "corr-slow-1"

    with (
        patch(
            "chat_phase_metrics.query_chat_phase_percentiles",
            return_value={"window_hours": 24, "server": {}, "client": {}, "by_tool": []},
        ),
        patch(
            "chat_phase_metrics.query_chat_efficiency_breakdown",
            return_value={"window_hours": 24, "llm_summary": {"turns": 7}, "by_nudge": []},
        ),
    ):
        resp_perf = sys_router.admin_chat_perf(
            SimpleNamespace(headers={"authorization": "Bearer ok"}),
            hours=24,
            include_insights=True,
        )
    assert resp_perf["success"] is True
    assert resp_perf["insights"]["negative_feedback"][0]["session_id"] == "s-1"
    assert resp_perf["efficiency"]["llm_summary"]["turns"] == 7


# ---------------------------------------------------------------------------
# G-28: Idempotencia ante reintentos y doble envío en tools de escritura
# ---------------------------------------------------------------------------

def test_g28_log_consumed_meal_idempotent_dedup_and_force_bypass(monkeypatch):
    """Un reintento dentro de la ventana dedup NO inserta otra fila ni descuenta Nevera doble;
    solo `force=True` vuelve a registrar cuando el usuario confirma otra porción."""
    import db_inventory
    import tools

    deduct_mock = MagicMock(return_value="descontado")
    log_calls: list[dict] = []

    def _fake_db_log(*args, **kwargs):
        log_calls.append({"allow_repeat": kwargs.get("allow_repeat"), "args": args})
        if not kwargs.get("allow_repeat") and len(log_calls) == 2:
            return "deduped"
        return f"cm-id-{len(log_calls)}"

    monkeypatch.setattr(tools, "db_log_consumed_meal", _fake_db_log)
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: True)
    monkeypatch.setattr(db_inventory, "deduct_consumed_meal_from_inventory", deduct_mock)
    monkeypatch.setattr(tools, "_nota_total_del_dia", lambda uid, d=0: " Total: 420 kcal.")
    monkeypatch.setattr(tools, "_nota_comidas_sin_registrar", lambda uid, d=0: "")
    monkeypatch.setattr(tools, "_hora_local_float", lambda uid: 9.0)
    monkeypatch.setattr(tools, "_buscar_cena_principal_hoy", lambda uid: None)
    monkeypatch.setattr(tools, "_rescue_dinner_slot", lambda u, mt, cal, da: mt)
    monkeypatch.setattr(tools, "_rescue_dinner_from_unlabeled_desayuno", lambda u, mt, cal, da: mt)
    monkeypatch.setattr(tools, "_rescue_stale_desayuno_after_noon", lambda u, mt, cal, da: (mt, False))
    monkeypatch.setattr(tools, "_rescue_breakfast_from_premature_almuerzo", lambda u, mt, mn: mt)

    # Neutralize optional nutrition catalog/profile side-effects
    import db as _db_mod

    monkeypatch.setattr(_db_mod, "get_user_profile", lambda uid: {}, raising=False)
    monkeypatch.setattr(tools, "assert_valid_uuid", lambda v: None, raising=False)

    base_payload = {
        "user_id": "u-idem-1",
        "meal_name": "Huevos revueltos con pan",
        "calories": 380,
        "protein": 22,
        "carbs": 30,
        "healthy_fats": 18,
        "ingredients": ["2 huevos (≈110 g)", "2 rebanadas de pan (≈60 g)"],
        "meal_type": "desayuno",
    }

    first = tools.log_consumed_meal.invoke(base_payload)
    assert "Éxito" in first
    assert deduct_mock.call_count == 1

    # Retry with same payload (force=False): deduped -> no second inventory deduction!
    second = tools.log_consumed_meal.invoke(base_payload)
    assert "NO REGISTRADO DE NUEVO" in second
    assert "force=true" in second
    assert deduct_mock.call_count == 1
    assert log_calls[1]["allow_repeat"] is False

    # Explicit confirmation (force=True): allows repeat and deducts once more
    third = tools.log_consumed_meal.invoke({**base_payload, "force": True})
    assert "Éxito" in third
    assert deduct_mock.call_count == 2
    assert log_calls[2]["allow_repeat"] is True


def test_g28_log_consumed_meal_source_and_mark_shopping_list_purchased_idempotent(monkeypatch):
    """`mark_shopping_list_purchased` salta ciclos ya registrados sin duplicar inventario y
    `save_message_with_attachments` mantiene `ON CONFLICT (session_id, client_message_id)`."""
    import db_inventory
    import restock_cycle
    import tools

    restock_mock = MagicMock(return_value=(True, ["Pollo"]))
    monkeypatch.setattr(tools, "_usuario_en_modo_contador", lambda uid: False)
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan_with_id", lambda uid: {"id": "plan-1", "plan_data": {"days": []}})
    monkeypatch.setattr(
        "shopping_calculator.get_shopping_list_delta",
        lambda uid, plan, structured=True: [{"name": "Pollo", "quantity": 2, "unit": "lb"}],
    )
    monkeypatch.setattr(
        restock_cycle,
        "filter_purchase_for_cycle",
        lambda uid, plan, items: {"filtered": [], "skipped": ["Pollo"]},
    )
    monkeypatch.setattr(db_inventory, "restock_inventory", restock_mock)

    out = tools.mark_shopping_list_purchased.invoke({"user_id": "u-shop-1"})
    assert "No se añadió nada a la Nevera" in out
    assert restock_mock.call_count == 0

    db_chat_src = (Path(_BACKEND_DIR) / "db_chat.py").read_text(encoding="utf-8")
    assert "ON CONFLICT (session_id, client_message_id)" in db_chat_src


# ---------------------------------------------------------------------------
# G-26: Paridad contractual entre voz live/TTS y texto
# ---------------------------------------------------------------------------

def test_g26_voice_live_and_text_contract_parity():
    """El coach live de voz comparte evidencia clínica, delegación al backend,
    verdad del diario manual y reglas multilingües con el prompt de texto."""
    import coach_live
    import coach_voz
    from health_evidence import HEALTH_EVIDENCE_RULES
    from prompts import chat_agent as ca

    # Texto base y modo stream incluyen HEALTH_EVIDENCE_RULES y modo voz recuerda concisión
    assert HEALTH_EVIDENCE_RULES in ca.CHAT_SYSTEM_PROMPT_BASE
    assert HEALTH_EVIDENCE_RULES in ca.CHAT_STREAM_SYSTEM_PROMPT_BASE
    assert "NUNCA uses delete=True" in ca._CHAT_CALL_MODE_RULES

    # Live voice (`gpt-live-1`) debe compartir reglas clínicas y delegación real al backend de texto
    for locale in ("es-DO", "en-US"):
        inst = coach_live.instrucciones(locale)
        assert HEALTH_EVIDENCE_RULES in inst
        assert "DELEGA siempre" in inst
        assert "no inventes" in inst.lower() or "never invent" in inst.lower()
        assert "una comida eliminada ya no cuenta" in inst.lower()

    # El bypass de charla corta en voz TTS no puede capturar registros reales ni consultas clínicas
    assert coach_voz.es_charla_corta("Hola") is True
    assert coach_voz.es_charla_corta("Comí dos huevos con pan hoy") is False
    assert coach_voz.es_charla_corta("Tengo alergia al maní y me duele la cabeza") is False


def test_p1_plan_lote_1011_marker_crosslink():
    """Anchor P2-HIST-AUDIT-14: slug `p1_plan_lote_1011` ↔ `P1-PLAN-LOTE-1011`."""
    assert Path(__file__).exists()
