"""[P2-BILLING-PAYMENT-FAILED-GRACE · 2026-05-30] `PAYMENT.FAILED` no es terminal.

Bug (audit prod-readiness 2026-05-30):
    El webhook handler de PayPal incluía `BILLING.SUBSCRIPTION.PAYMENT.FAILED`
    en `downgrade_events` → un fallo de pago TRANSITORIO (tarjeta rebota una vez;
    PayPal reintenta durante su dunning window de varios días) degradaba al
    usuario a `gratis` INSTANTÁNEAMENTE. Y NO había handler de re-activación, así
    que aunque PayPal cobrara con éxito en el reintento, el usuario perdía el
    acceso pagado de forma permanente.

Fix:
    - PAYMENT.FAILED → status no-destructivo `PAYMENT_RETRYING` (conserva plan_tier).
    - Solo SUSPENDED/EXPIRED/CANCELLED degradan.
    - ACTIVATED / PAYMENT.SALE.COMPLETED → restaura tier desde plan_id de PayPal.
    - Knob `MEALFIT_BILLING_PAYMENT_FAILED_GRACE` (default True) como kill-switch.

Tests parser-based sobre el handler del webhook.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BILLING = Path(__file__).resolve().parent.parent / "routers" / "billing.py"


def _read() -> str:
    return _BILLING.read_text(encoding="utf-8")


def _webhook_handler_body(src: str) -> str:
    """Aísla el cuerpo del handler del webhook PayPal (desde el def del handler
    que contiene `downgrade_events` hasta el siguiente `@router`/`def` toplevel)."""
    idx = src.find("downgrade_events")
    assert idx != -1, "No se encontró `downgrade_events` en billing.py."
    # Ventana generosa alrededor del bloque de manejo de eventos.
    return src[idx - 200 : idx + 3000]


def test_payment_failed_not_in_default_downgrade_list():
    """Por default (grace ON), PAYMENT.FAILED NO debe estar en la lista literal
    de `downgrade_events` (solo entra en el branch de rollback `if not grace`)."""
    src = _read()
    # La asignación literal de downgrade_events solo debe contener SUSPENDED + EXPIRED.
    m = re.search(r"downgrade_events\s*=\s*\[(.*?)\]", src, re.DOTALL)
    assert m, "No se encontró la asignación literal de `downgrade_events`."
    literal = m.group(1)
    assert "SUSPENDED" in literal and "EXPIRED" in literal, (
        "downgrade_events debe seguir conteniendo SUSPENDED + EXPIRED."
    )
    assert "PAYMENT.FAILED" not in literal, (
        "PAYMENT.FAILED NO debe estar en la lista literal de `downgrade_events` "
        "(default grace). Reabrir esto degrada al usuario por un fallo transitorio "
        "(P2-BILLING-PAYMENT-FAILED-GRACE)."
    )


def test_payment_failed_soft_branch_sets_retrying_not_gratis():
    """Debe existir un branch PAYMENT.FAILED que setee PAYMENT_RETRYING SIN
    poner plan_tier='gratis'.

    [P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado: el branch ya no termina
    en `.execute()` (PostgREST) sino en `execute_sql_write(...)` con el
    UPDATE SQL directo. Misma propiedad: solo flippea subscription_status,
    jamás degrada tier."""
    src = _read()
    assert "PAYMENT_RETRYING" in src, (
        "Falta el status no-destructivo `PAYMENT_RETRYING` para PAYMENT.FAILED."
    )
    # El branch del soft-handling no debe degradar tier.
    m = re.search(
        r'event_type\s*==\s*"BILLING\.SUBSCRIPTION\.PAYMENT\.FAILED"'
        r'[\s\S]*?execute_sql_write\([\s\S]*?\(subscription_id,\),',
        src,
    )
    assert m, "No se encontró el branch soft de PAYMENT.FAILED."
    branch = m.group(0)
    assert "subscription_status = 'PAYMENT_RETRYING'" in branch, (
        "El branch PAYMENT.FAILED debe setear subscription_status='PAYMENT_RETRYING' "
        "en el UPDATE SQL."
    )
    assert "gratis" not in branch, (
        "El branch PAYMENT.FAILED NO debe poner plan_tier='gratis' (eso es degradar)."
    )


def test_reactivation_branch_restores_tier():
    """Debe existir un branch para ACTIVATED/PAYMENT.SALE.COMPLETED que restaure
    el tier desde el plan_id de PayPal (reuse `_build_paypal_plan_tier_map`).

    [P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado: el branch ya no termina
    en `.execute()` (PostgREST) sino en el despacho del UPDATE SQL via
    `await _supabase_async(_do_reactivate)`."""
    src = _read()
    assert "BILLING.SUBSCRIPTION.ACTIVATED" in src, (
        "Falta el handler de re-activación `BILLING.SUBSCRIPTION.ACTIVATED`."
    )
    m = re.search(
        r'event_type\s+in\s+\(\s*"BILLING\.SUBSCRIPTION\.ACTIVATED"'
        r'[\s\S]*?await _supabase_async\(_do_reactivate\)',
        src,
    )
    assert m, "No se encontró el branch de re-activación."
    branch = m.group(0)
    assert "_build_paypal_plan_tier_map" in branch, (
        "El branch de re-activación debe derivar el tier desde el plan_id via "
        "`_build_paypal_plan_tier_map` (server-side, no del cliente)."
    )
    assert '"subscription_status": "ACTIVE"' in branch, (
        "La re-activación debe marcar subscription_status='ACTIVE' (limpiar retry flag)."
    )


def test_knob_present_and_default_true():
    src = _read()
    assert 'MEALFIT_BILLING_PAYMENT_FAILED_GRACE' in src, (
        "Falta el knob kill-switch `MEALFIT_BILLING_PAYMENT_FAILED_GRACE`."
    )
    m = re.search(
        r'_env_bool\(\s*"MEALFIT_BILLING_PAYMENT_FAILED_GRACE"\s*,\s*True\s*\)',
        src,
    )
    assert m, "El knob debe tener default True (grace activo por defecto)."


def test_anchor_present():
    assert "P2-BILLING-PAYMENT-FAILED-GRACE" in _read()
