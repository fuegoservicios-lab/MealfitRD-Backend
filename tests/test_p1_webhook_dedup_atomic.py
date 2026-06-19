"""[P1-WEBHOOK-DEDUP-ATOMIC · 2026-06-19] (audit fresco P1-2) El dedup del webhook PayPal es de DOS FASES:
el marcador `paypal_webhook:<transmission_id>` solo cuenta como 'procesado' (status=done) DESPUÉS de aplicar
la transición de estado (UPDATE de plan_tier/subscription_status).

Antes el marcador se INSERTaba+commiteaba ANTES del UPDATE: si el UPDATE fallaba transitoriamente (hiccup de
Neon/red) → except → HTTP 503 → PayPal reintenta el MISMO transmission_id, pero el reintento veía el marcador
ya committeado y la rama `deduped` SALTABA el procesamiento → el evento de billing se perdía PERMANENTEMENTE
(no-pagador en tier pagado / pagador degradado). Reabría la clase que P2-WEBHOOK-INFRA-503 cerró.

Este test ejercita el handler real con `request` y `execute_sql_write` mockeados:
  1. happy path → marca 'done' tras procesar.
  2. fallo transitorio + reintento → RE-PROCESA (NO pierde el evento) — el corazón del fix.
  3. reentrega de un evento ya completado → deduplica sin re-procesar.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi import HTTPException

import routers.billing as billing


class _FakeReq:
    """Request mínimo: body() async + headers dict-like."""
    def __init__(self, body_bytes, headers):
        self._body = body_bytes
        self.headers = headers

    async def body(self):
        return self._body


class _FakeDB:
    """Simula app_kv_store (dict) + cuenta los UPDATE de user_profiles (la transición de estado)."""
    def __init__(self):
        self.kv = {}                       # key -> {'status': ...}
        self.state_update_calls = 0
        self.fail_next_state_update = False

    def execute_sql_write(self, query, params=None, returning=False, lock_timeout_ms=None):
        q = " ".join(str(query).split())   # normaliza whitespace
        # Fase 1 — claim: INSERT ... ON CONFLICT DO NOTHING RETURNING key
        if q.startswith("INSERT INTO app_kv_store"):
            key = params[0]
            if key in self.kv:
                return []                  # conflicto = ya existía
            self.kv[key] = {"status": "processing"}
            return [{"key": key}]
        # ¿status=done?
        if q.startswith("SELECT 1 AS done FROM app_kv_store"):
            key = params[0]
            return [{"done": 1}] if self.kv.get(key, {}).get("status") == "done" else []
        # Fase 2 — marcar done
        if q.startswith("UPDATE app_kv_store SET value = jsonb_build_object"):
            key = params[0]
            self.kv[key] = {"status": "done"}
            return True
        # Transición de estado (downgrade/reactivate/cancel sobre user_profiles)
        if "user_profiles" in q:
            self.state_update_calls += 1
            if self.fail_next_state_update:
                self.fail_next_state_update = False
                raise RuntimeError("transient Neon hiccup durante el UPDATE de estado")
            return True
        return True


@pytest.fixture
def fake_db(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(billing, "execute_sql_write", fake.execute_sql_write)
    # Sin credenciales PayPal + sandbox + unsigned permitido → el handler salta la verificación de firma
    # y llega directo a la lógica de dedup (lo que queremos probar). is_production mockeada para no depender
    # del entorno de CI.
    monkeypatch.setattr(billing, "is_production", lambda: False)
    monkeypatch.setenv("MEALFIT_ALLOW_WEBHOOK_UNSIGNED", "1")
    for k in ("PAYPAL_CLIENT_ID", "PAYPAL_SECRET", "PAYPAL_WEBHOOK_ID"):
        monkeypatch.delenv(k, raising=False)
    return fake


def _suspended_req(tx_id):
    body = json.dumps({"event_type": "BILLING.SUBSCRIPTION.SUSPENDED",
                       "resource": {"id": "I-SUB-1"}}).encode("utf-8")
    return _FakeReq(body, {"paypal-transmission-id": tx_id})


def _call(req):
    return asyncio.run(billing.api_webhook_paypal(req, _rl=None))


def test_happy_path_marks_done(fake_db):
    res = _call(_suspended_req("TX-OK"))
    assert res == {"success": True}
    assert fake_db.state_update_calls == 1
    assert fake_db.kv["paypal_webhook:TX-OK"]["status"] == "done"


def test_transient_failure_leaves_processing_then_retry_reprocesses(fake_db):
    """EL FIX: una transición de estado que falla deja el marcador 'processing'; el reintento RE-PROCESA."""
    fake_db.fail_next_state_update = True
    with pytest.raises(HTTPException) as ei:
        _call(_suspended_req("TX-RETRY"))
    assert ei.value.status_code == 503                              # 503 → PayPal reintenta
    assert fake_db.kv["paypal_webhook:TX-RETRY"]["status"] == "processing"  # NO 'done'
    assert fake_db.state_update_calls == 1

    # Reintento del MISMO transmission_id: NO deduplica (status != done) → re-procesa → evento NO perdido.
    res = _call(_suspended_req("TX-RETRY"))
    assert res == {"success": True}
    assert fake_db.state_update_calls == 2                          # ← re-procesó (la prueba del fix)
    assert fake_db.kv["paypal_webhook:TX-RETRY"]["status"] == "done"


def test_completed_event_dedups_without_reprocessing(fake_db):
    first = _call(_suspended_req("TX-DEDUP"))
    assert first == {"success": True}
    assert fake_db.state_update_calls == 1

    second = _call(_suspended_req("TX-DEDUP"))                       # reentrega de un evento YA completado
    assert second == {"success": True, "deduped": True}
    assert fake_db.state_update_calls == 1                          # no re-procesó
