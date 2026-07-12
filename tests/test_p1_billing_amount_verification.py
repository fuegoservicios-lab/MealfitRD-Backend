"""[P1-BILLING-AMOUNT · 2026-07-12] `/api/subscription/verify` debe verificar el
MONTO cobrado server-side, no solo el tier.

Vector pre-fix:
    I-Billing-1 deriva el tier del plan_id (server-side) pero el cliente puede
    inyectar `plan.billing_cycles[].pricing_scheme.fixed_price` al crear la sub
    PayPal (PaymentModal lo hace para descuentos) y cobrarse el PRIMER ciclo a un
    precio arbitrario, SIN cupón válido. El tier no escala, pero el monto sí.

Fix:
    Si PayPal marca `plan_overridden`, el precio fue sobrescrito → solo legítimo si
    un cupón válido (re-validado server-side) lo justifica. Knob de 3 modos
    (off/warn/block, default warn). En warn: alerta sin bloquear. En block: 409ea
    SOLO con underpayment PROBADO (precio parseable < mínimo esperado) — nunca
    bloquea el caso ambiguo (fail-cheap). Hardening: el cupón se marca consumido.

Lo que enforza este test:
    A) Parser-based (siempre corre): knob, helpers, llamada en /verify, alert_key,
       forward de coupon_code, anchor.
    B) Funcional (skip si el módulo no importa): helpers puros + la lógica de
       decisión de `_verify_subscription_amount` (skip override, warn vs block,
       cupón legítimo, off).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_BILLING_PY = _BACKEND_ROOT / "routers" / "billing.py"


@pytest.fixture(scope="module")
def billing_src() -> str:
    return _BILLING_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Parser-based (sin deps — corre siempre)
# ---------------------------------------------------------------------------

def test_a_knob_and_helpers_present(billing_src: str):
    assert 'MEALFIT_BILLING_VERIFY_AMOUNT' in billing_src, (
        "P1-BILLING-AMOUNT: knob de modo (off/warn/block) ausente."
    )
    for helper in (
        "def _verify_subscription_amount",
        "def _validate_discount_code",
        "def _fetch_plan_list_price",
        "def _extract_override_price",
        "def _redeem_discount_code",
    ):
        assert helper in billing_src, f"P1-BILLING-AMOUNT: falta `{helper}`."


def test_b_verify_calls_amount_check_before_update(billing_src: str):
    # La llamada debe existir y aparecer ANTES del UPDATE de plan_tier.
    call_idx = billing_src.find("await _verify_subscription_amount(")
    update_idx = billing_src.find("UPDATE public.user_profiles")
    assert call_idx != -1, "P1-BILLING-AMOUNT: /verify no invoca _verify_subscription_amount."
    assert update_idx != -1
    assert call_idx < update_idx, (
        "P1-BILLING-AMOUNT: la verificación de monto debe ir ANTES del UPDATE del tier."
    )


def test_c_alert_key_and_coupon_forward(billing_src: str):
    assert "billing_price_tampering:" in billing_src, (
        "P1-BILLING-AMOUNT: alert_key de tampering ausente."
    )
    assert 'data.get("coupon_code")' in billing_src, (
        "P1-BILLING-AMOUNT: /verify no lee coupon_code del body para re-validarlo."
    )
    assert "P1-BILLING-AMOUNT" in billing_src, "anchor ausente."


# ---------------------------------------------------------------------------
# Funcional (skip si routers.billing no es importable en este entorno)
# ---------------------------------------------------------------------------

def _load():
    if str(_BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(_BACKEND_ROOT))
    try:
        import routers.billing as b
    except Exception as exc:  # pragma: no cover - entorno sin deps
        pytest.skip(f"routers.billing no importable: {exc}")
    return b


def _sub(overridden: bool, price: str):
    """sub_data PayPal con un plan embebido cuyo ciclo REGULAR cobra `price`."""
    return {
        "status": "ACTIVE",
        "plan_id": "P-ULTRA",
        "plan_overridden": overridden,
        "plan": {
            "billing_cycles": [
                {"tenure_type": "REGULAR", "pricing_scheme": {"fixed_price": {"value": price}}}
            ]
        },
    }


def test_d_pure_price_helpers():
    b = _load()
    assert b._parse_price("9.99") == 9.99
    assert b._parse_price("abc") is None
    assert b._parse_price(None) is None
    assert b._extract_override_price(_sub(True, "0.01")) == 0.01


def _run_verify(b, monkeypatch, *, mode, overridden, price, disc, list_price):
    """Ejecuta _verify_subscription_amount con deps mockeadas; devuelve
    (raised_409: bool, alerts: list)."""
    monkeypatch.setenv("MEALFIT_BILLING_VERIFY_AMOUNT", mode)

    async def _fake_validate(code, tier):
        return {"discount_percent": disc} if disc is not None else None

    async def _fake_list_price(plan_id, tier, token, base):
        return list_price

    alerts = []
    monkeypatch.setattr(b, "_validate_discount_code", _fake_validate)
    monkeypatch.setattr(b, "_fetch_plan_list_price", _fake_list_price)
    monkeypatch.setattr(b, "_persist_billing_alert", lambda **kw: alerts.append(kw))

    from fastapi import HTTPException
    raised = False
    try:
        asyncio.run(b._verify_subscription_amount(
            sub_data=_sub(overridden, price),
            verified_plan_id="P-ULTRA",
            tier="ultra",
            coupon_code=("SAVE" if disc is not None else ""),
            access_token="tok",
            paypal_api_base="https://api-m.sandbox.paypal.com",
            user_id="u1",
            subscription_id="s1",
        ))
    except HTTPException as e:
        raised = (e.status_code == 409)
    return raised, alerts


def test_e_no_override_skips(monkeypatch):
    b = _load()
    # Sin override: precio estándar → ni alerta ni bloqueo, aun en block mode.
    raised, alerts = _run_verify(b, monkeypatch, mode="block", overridden=False,
                                 price="9.99", disc=None, list_price=9.99)
    assert raised is False
    assert alerts == []


def test_f_block_on_proven_underpayment(monkeypatch):
    b = _load()
    # Override + sin cupón + precio 0.01 vs lista 9.99 → underpayment probado → 409 + alerta.
    raised, alerts = _run_verify(b, monkeypatch, mode="block", overridden=True,
                                 price="0.01", disc=None, list_price=9.99)
    assert raised is True
    assert len(alerts) == 1
    assert alerts[0]["alert_key"] == "billing_price_tampering:u1:s1"
    assert alerts[0]["severity"] == "critical"


def test_g_warn_alerts_but_does_not_block(monkeypatch):
    b = _load()
    # Mismo tampering pero en warn: alerta, NO bloquea (pago sigue).
    raised, alerts = _run_verify(b, monkeypatch, mode="warn", overridden=True,
                                 price="0.01", disc=None, list_price=9.99)
    assert raised is False
    assert len(alerts) == 1


def test_h_legit_coupon_price_ok(monkeypatch):
    b = _load()
    # Override + cupón 20% válido → esperado_min = 9.99*0.8 = 7.99; precio 7.99 → OK.
    raised, alerts = _run_verify(b, monkeypatch, mode="block", overridden=True,
                                 price="7.99", disc=20, list_price=9.99)
    assert raised is False
    assert alerts == []


def test_i_mode_off_skips(monkeypatch):
    b = _load()
    raised, alerts = _run_verify(b, monkeypatch, mode="off", overridden=True,
                                 price="0.01", disc=None, list_price=9.99)
    assert raised is False
    assert alerts == []
