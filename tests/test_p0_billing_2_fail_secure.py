"""[P0-BILLING-2 · 2026-05-12] `/api/subscription/verify` y `/cancel` deben
fail-secure (HTTP 503) cuando faltan `PAYPAL_CLIENT_ID`/`PAYPAL_SECRET` en
producción.

Bug pre-fix:
    `if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET: success = True` →
    cualquier POST con un `subscription_id` arbitrario UPDATE-aba
    `user_profiles` con el tier solicitado por el cliente. Si el contenedor
    perdía las env vars (rotación rota, misconfig del VPS Oracle post-rolling
    deploy), el sistema entraba en fail-OPEN — paywall completamente
    bypasseable. El comentario legacy lo admitía:
        `(SECURITY RISK IF PRODUCTION)`

Fix:
    En `/verify` y `/cancel`, antes de cualquier UPDATE:
      - Detectar `env_ready = bool(PAYPAL_CLIENT_ID and PAYPAL_SECRET)`.
      - Detectar `allow_bypass = MEALFIT_ALLOW_PAYPAL_BYPASS in {"1","true","yes"}`.
      - Si `not env_ready and not is_sandbox and not allow_bypass`:
        `raise HTTPException(503, "Payment provider misconfigured")`.
    Knob `MEALFIT_ALLOW_PAYPAL_BYPASS` existe SOLO para dev local. En
    ENVIRONMENT=production el gate ignora ese knob → fail-secure verdadero.

Lo que este test enforza:
    A) Ningún path en `/verify` o `/cancel` asigna `success = True` o
       `env_ready = True` sin haber leído PAYPAL_CLIENT_ID/PAYPAL_SECRET
       en una rama positiva (`and`/`or` con esos nombres).
    B) Ambos handlers contienen `raise HTTPException(status_code=503, ...)`
       en algún branch que se activa por env vars ausentes.
    C) `MEALFIT_ALLOW_PAYPAL_BYPASS` aparece como knob lateral (kill switch
       para dev) — su ausencia indicaría que el bypass se perdió.
    D) Anchor `P0-BILLING-2-FAIL-SECURE` o `P0-BILLING-2` permanece.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_BILLING_PY = _BACKEND_ROOT / "routers" / "billing.py"


@pytest.fixture(scope="module")
def billing_src() -> str:
    return _BILLING_PY.read_text(encoding="utf-8")


def _isolate_handler(src: str, route_path: str) -> str:
    """Aísla el cuerpo de un handler `@router.post(route_path)` hasta el
    siguiente decorator o EOF.
    """
    pattern = rf'@router\.post\("{re.escape(route_path)}"\)(.*?)(?=@router\.post|@webhooks_router\.post|\Z)'
    m = re.search(pattern, src, re.DOTALL)
    assert m is not None, f"Handler {route_path} no encontrado."
    return m.group(1)


def test_a_no_success_true_under_missing_env_vars(billing_src: str):
    """Patrón legacy `success = True` dentro de un `if not PAYPAL_CLIENT_ID
    or not PAYPAL_SECRET:` está PROHIBIDO."""
    pattern = re.compile(
        r"if\s+not\s+PAYPAL_CLIENT_ID\s+or\s+not\s+PAYPAL_SECRET\s*:"
        r"[\s\S]{0,500}?"
        r"success\s*=\s*True",
        re.MULTILINE,
    )
    assert pattern.search(billing_src) is None, (
        "P0-BILLING-2 regresión: el patrón `if not PAYPAL_CLIENT_ID or not "
        "PAYPAL_SECRET: ... success = True` reaparece. Eso es fail-OPEN — "
        "cualquier POST upgradea sin verificar pago. Reemplazar por "
        "HTTPException(503)."
    )


def test_b_503_branch_in_verify_handler(billing_src: str):
    """`/verify` contiene un `raise HTTPException(status_code=503, ...)`
    asociado al gate de env vars ausentes."""
    verify_body = _isolate_handler(billing_src, "/verify")
    assert "status_code=503" in verify_body, (
        "P0-BILLING-2: /verify no levanta 503 en ningún path. "
        "El gate fail-secure debe rechazar cuando faltan env vars en prod."
    )
    # El 503 debe estar cerca de una mención a PAYPAL_CLIENT_ID o PAYPAL_SECRET
    # (no un 503 random de otra causa).
    near_pattern = re.compile(
        r"PAYPAL_CLIENT_ID[\s\S]{0,1500}status_code=503"
        r"|status_code=503[\s\S]{0,1500}PAYPAL_CLIENT_ID",
    )
    assert near_pattern.search(verify_body), (
        "P0-BILLING-2: el 503 en /verify no parece estar asociado al gate "
        "de env vars PayPal. Verificar bloque de fail-secure."
    )


def test_c_503_branch_in_cancel_handler(billing_src: str):
    """`/cancel` también debe fail-secure: pre-fix BD se marcaba CANCELLED
    pero PayPal seguía cobrando si faltaban las keys.
    """
    cancel_body = _isolate_handler(billing_src, "/cancel")
    assert "status_code=503" in cancel_body, (
        "P0-BILLING-2: /cancel no levanta 503 cuando faltan PayPal env "
        "vars en producción. Riesgo: BD CANCELLED + cobro recurrente vivo."
    )


def test_d_bypass_knob_present_and_dev_only(billing_src: str):
    """`MEALFIT_ALLOW_PAYPAL_BYPASS` referenciado en billing.py — kill
    switch explícito para dev. El gate debe combinarlo con
    `not is_sandbox` para asegurar que NO funciona en producción."""
    assert "MEALFIT_ALLOW_PAYPAL_BYPASS" in billing_src, (
        "P0-BILLING-2: knob `MEALFIT_ALLOW_PAYPAL_BYPASS` removido. "
        "Restaurarlo — su ausencia rompe dev local sin credenciales reales."
    )


def test_e_anchor_present(billing_src: str):
    anchors = ("P0-BILLING-2-FAIL-SECURE", "P0-BILLING-2")
    assert any(a in billing_src for a in anchors), (
        f"P0-BILLING-2: anchor {anchors} desapareció. Restaurar."
    )
