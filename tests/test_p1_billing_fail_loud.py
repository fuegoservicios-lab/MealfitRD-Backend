"""[P1-BILLING-UPGRADE-FAIL-LOUD + P1-BILLING-CANCEL-FAIL-LOUD · 2026-05-12]
`backend/routers/billing.py` debe fail-loud cuando una cancel a PayPal falla
en `verify_subscription` (upgrade path) o `cancel_subscription`. Pre-fix
ambos paths solo emitían `logger.warning` / `logger.error` y SEGUÍAN al
UPDATE de la tabla `user_profiles`, dejando BD en estado divergente con
PayPal (doble cobro o cobro post-cancel).

Modo de fallo del bug original:

  /verify (upgrade path):
    if cancel_resp.status_code == 204:
        logger.info(...)
    else:
        logger.warning(...)        ← solo warning, sigue al UPDATE.
    # ... UPDATE plan_tier=ultra, paypal_subscription_id=new_sub_id
    # Resultado: sub vieja sigue ACTIVE en PayPal + nueva también →
    # cliente pagando 2 subs.

  /cancel:
    if cancel_resp.status_code not in [204, 200]:
        logger.error(...)          ← solo error, sigue al UPDATE.
    # ... UPDATE subscription_status='CANCELLED'
    # Resultado: BD dice CANCELLED, PayPal sigue cobrando.

Fix verificado por este test:

  A) El patrón legacy `else: logger.warning(...cancelar suscripción antigua)`
     SIN raise está PROHIBIDO en /verify.
  B) El patrón legacy `if cancel_resp.status_code not in [204, 200]:
     logger.error(...)` SIN raise está PROHIBIDO en /cancel.
  C) Ambos handlers contienen `raise HTTPException` con `status_code=409`
     (verify) o `status_code=502` (cancel) en el path de cancel-failure.
  D) Helper `_is_paypal_cancel_idempotent_success` referenciado en ambos
     callsites (404 + 422-already-cancelled tratados como éxito).
  E) Helper `_persist_billing_alert` referenciado en ambos callsites para
     emitir `billing_old_sub_cancel_failed:<>:<>` y
     `billing_cancel_failed:<>:<>` a `system_alerts`.
  F) Anchors `P1-BILLING-UPGRADE-FAIL-LOUD` y `P1-BILLING-CANCEL-FAIL-LOUD`
     presentes.

Tooltip-anchor: P1-BILLING-FAIL-LOUD-TESTS
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
    siguiente decorator o EOF. Mismo helper que `test_p0_billing_2_fail_secure`.
    """
    pattern = rf'@router\.post\("{re.escape(route_path)}"\)(.*?)(?=@router\.post|@webhooks_router\.post|@discount_router\.post|\Z)'
    m = re.search(pattern, src, re.DOTALL)
    assert m is not None, f"Handler {route_path} no encontrado."
    return m.group(1)


# ============================================================================
# Test A: el patrón legacy de upgrade (solo warning) está prohibido
# ============================================================================

def test_a_upgrade_path_no_warning_only_on_cancel_failure(billing_src: str):
    """`/verify` (upgrade) NO debe tener un bloque donde la única reacción
    a un cancel != 204 sea `logger.warning` sin `raise HTTPException`."""
    verify_body = _isolate_handler(billing_src, "/verify")
    # Patrón legacy: `else: logger.warning(... cancelar suscripción antigua ...)`
    # sin un `raise HTTPException` cerca (mismo bloque if/else).
    legacy_pattern = re.compile(
        r"if\s+cancel_resp\.status_code\s*==\s*204\s*:"
        r"[\s\S]{0,200}?"
        r"else\s*:\s*\n\s*logger\.warning\([^)]*cancelar suscripción antigua",
        re.MULTILINE,
    )
    assert legacy_pattern.search(verify_body) is None, (
        "P1-BILLING-UPGRADE-FAIL-LOUD regresión: el patrón legacy "
        "`if cancel_resp.status_code == 204: ... else: logger.warning(...)` "
        "reaparece. Eso permite doble cobro: sub vieja sigue ACTIVE en "
        "PayPal mientras la nueva también cobra. Reemplazar con "
        "fail-loud (raise HTTPException) + _persist_billing_alert."
    )


def test_b_cancel_endpoint_no_log_error_only_on_failure(billing_src: str):
    """`/cancel` NO debe tener un bloque donde la única reacción a
    `cancel_resp.status_code not in [204, 200]` sea `logger.error` sin
    `raise HTTPException`."""
    cancel_body = _isolate_handler(billing_src, "/cancel")
    # Patrón legacy: `if cancel_resp.status_code not in [204, 200]:
    # logger.error(...)` sin un `raise` en el bloque inmediato.
    legacy_pattern = re.compile(
        r"if\s+cancel_resp\.status_code\s+not\s+in\s*\[\s*204\s*,\s*200\s*\]\s*:"
        r"\s*\n\s*logger\.error\([^)]*\)\s*\n(?!\s*(raise|_persist_billing_alert))",
        re.MULTILINE,
    )
    assert legacy_pattern.search(cancel_body) is None, (
        "P1-BILLING-CANCEL-FAIL-LOUD regresión: el patrón legacy "
        "`if cancel_resp.status_code not in [204, 200]: logger.error(...)` "
        "(sin raise/alert) reaparece. Eso permite que la BD se marque "
        "CANCELLED mientras PayPal sigue cobrando. Reemplazar con "
        "fail-loud (raise HTTPException 502) + _persist_billing_alert."
    )


# ============================================================================
# Test C/D: ambos handlers contienen raise HTTPException en cancel-failure
# ============================================================================

def test_c_upgrade_raises_409_on_cancel_failure(billing_src: str):
    """`/verify` debe contener `raise HTTPException(status_code=409, ...)`
    asociado al cancel de la sub vieja."""
    verify_body = _isolate_handler(billing_src, "/verify")
    assert "status_code=409" in verify_body, (
        "P1-BILLING-UPGRADE-FAIL-LOUD: /verify no levanta 409 en ningún "
        "path. El upgrade debe abortar si la cancel de la sub vieja falla."
    )
    # El 409 debe estar cerca del bloque de cancel (referencia a cancel_resp
    # o cancel_client en una ventana razonable).
    near_pattern = re.compile(
        r"cancel_resp[\s\S]{0,1500}status_code=409"
        r"|status_code=409[\s\S]{0,1500}cancel_resp",
    )
    assert near_pattern.search(verify_body), (
        "P1-BILLING-UPGRADE-FAIL-LOUD: el 409 en /verify no parece "
        "asociado al bloque de cancel (cancel_resp). Verificar el flujo "
        "fail-loud post-cancel."
    )


def test_d_cancel_raises_502_on_paypal_failure(billing_src: str):
    """`/cancel` debe contener `raise HTTPException(status_code=502, ...)`
    asociado a fallo de PayPal (auth o cancel)."""
    cancel_body = _isolate_handler(billing_src, "/cancel")
    assert "status_code=502" in cancel_body, (
        "P1-BILLING-CANCEL-FAIL-LOUD: /cancel no levanta 502 en ningún "
        "path. La cancel debe fallar antes del UPDATE si PayPal no "
        "confirma la cancelación."
    )


# ============================================================================
# Test E: helper de idempotencia referenciado en ambos callsites
# ============================================================================

def test_e_idempotent_success_helper_used_in_both_handlers(billing_src: str):
    """`_is_paypal_cancel_idempotent_success` debe ser llamado en /verify
    (upgrade path) y en /cancel para tratar 404 / 422-already-cancelled
    como éxito."""
    verify_body = _isolate_handler(billing_src, "/verify")
    cancel_body = _isolate_handler(billing_src, "/cancel")

    assert "_is_paypal_cancel_idempotent_success" in verify_body, (
        "P1-BILLING-UPGRADE-FAIL-LOUD: /verify no referencia "
        "`_is_paypal_cancel_idempotent_success`. Sin tratamiento "
        "idempotente del caso 'sub ya cancelada' (404 / 422), "
        "los retries del cliente fallarán innecesariamente."
    )
    assert "_is_paypal_cancel_idempotent_success" in cancel_body, (
        "P1-BILLING-CANCEL-FAIL-LOUD: /cancel no referencia "
        "`_is_paypal_cancel_idempotent_success`. Mismo problema que "
        "arriba: retries de la UI fallarán cuando PayPal devuelva 404 "
        "post-cancel exitoso."
    )


def test_f_idempotent_helper_handles_known_cases(billing_src: str):
    """El helper `_is_paypal_cancel_idempotent_success` debe manejar
    204 (cancel ahora), 404 (sub ya purgada) y 422 con issues
    `SUBSCRIPTION_STATUS_INVALID`/`INVALID_SUBSCRIPTION_STATUS`/
    `SUBSCRIPTION_ALREADY_CANCELLED`."""
    # Verificar que el helper existe definido
    helper_def_re = re.compile(
        r"def\s+_is_paypal_cancel_idempotent_success\s*\(",
    )
    assert helper_def_re.search(billing_src), (
        "P1-BILLING-FAIL-LOUD: helper `_is_paypal_cancel_idempotent_success` "
        "no está definido."
    )
    # Status 204 / 404 referenciados
    assert "204" in billing_src and "404" in billing_src, (
        "P1-BILLING-FAIL-LOUD: el helper no parece manejar 204/404."
    )
    # Issues de PayPal que indican already-cancelled
    for issue in ("SUBSCRIPTION_STATUS_INVALID", "INVALID_SUBSCRIPTION_STATUS"):
        assert issue in billing_src, (
            f"P1-BILLING-FAIL-LOUD: el helper no menciona el issue PayPal "
            f"{issue!r} — sin ese filtro, retries que ya cancelaron "
            f"reportarán fallo falso."
        )


# ============================================================================
# Test G: _persist_billing_alert llamado en ambos callsites
# ============================================================================

def test_g_billing_alerts_persisted(billing_src: str):
    """Ambos handlers deben persistir alerts a `system_alerts` antes de
    levantar la HTTPException. Sin alert, SRE no se entera del incidente
    hasta que el cliente abre ticket."""
    verify_body = _isolate_handler(billing_src, "/verify")
    cancel_body = _isolate_handler(billing_src, "/cancel")

    assert "_persist_billing_alert(" in verify_body, (
        "P1-BILLING-UPGRADE-FAIL-LOUD: /verify no llama "
        "_persist_billing_alert. SRE necesita visibilidad del incidente "
        "para reconciliar BD ↔ PayPal manualmente."
    )
    assert "_persist_billing_alert(" in cancel_body, (
        "P1-BILLING-CANCEL-FAIL-LOUD: /cancel no llama "
        "_persist_billing_alert. Mismo problema."
    )

    # Verificar que los alert_keys nuevos existen (formato documentado)
    assert "billing_old_sub_cancel_failed:" in billing_src, (
        "P1-BILLING-UPGRADE-FAIL-LOUD: alert_key "
        "'billing_old_sub_cancel_failed:<user_id>:<old_sub_id>' no "
        "presente. Drift con la tabla de CLAUDE.md."
    )
    assert "billing_cancel_failed:" in billing_src, (
        "P1-BILLING-CANCEL-FAIL-LOUD: alert_key "
        "'billing_cancel_failed:<user_id>:<sub_id>' no presente. Drift "
        "con la tabla de CLAUDE.md."
    )


# ============================================================================
# Test H: anchors presentes
# ============================================================================

def test_h_anchors_present(billing_src: str):
    """Tooltip-anchors deben permanecer para drift-detection."""
    for anchor in (
        "P1-BILLING-UPGRADE-FAIL-LOUD",
        "P1-BILLING-CANCEL-FAIL-LOUD",
    ):
        assert anchor in billing_src, (
            f"P1-BILLING-FAIL-LOUD: anchor {anchor!r} desapareció. "
            f"Restaurar — es load-bearing para los test parser-based."
        )


# ============================================================================
# Test I: _persist_billing_alert usa supabase.table('system_alerts').upsert
# ============================================================================

def test_i_persist_billing_alert_uses_system_alerts_upsert(billing_src: str):
    """`_persist_billing_alert` debe escribir a `system_alerts` con
    `on_conflict='alert_key'` (idempotente, mismo patrón que el resto
    de emisores documentados en la tabla de CLAUDE.md)."""
    # Localizar el inicio del helper. Tolerante a `def ...(args) -> None:`
    # multi-línea con type hints.
    helper_start_re = re.compile(r"def\s+_persist_billing_alert\s*\(")
    m = helper_start_re.search(billing_src)
    assert m is not None, (
        "P1-BILLING-FAIL-LOUD: definición de `_persist_billing_alert` no "
        "encontrada."
    )
    # Ventana razonable post-definición: el helper debería caber en 2KB.
    # Si el body crece más allá, el test puede ampliarse — pero 2KB es
    # generoso para un UPSERT idempotente.
    body = billing_src[m.start(): m.start() + 2000]
    assert 'supabase.table("system_alerts")' in body, (
        "P1-BILLING-FAIL-LOUD: helper no escribe a `system_alerts`. "
        "Esa es la tabla canónica del policy 'system_alerts resolution' "
        "en CLAUDE.md."
    )
    assert 'on_conflict="alert_key"' in body, (
        "P1-BILLING-FAIL-LOUD: el UPSERT debe usar "
        "`on_conflict='alert_key'` para que retries del mismo incidente "
        "no dupliquen filas (idempotencia)."
    )
    assert '"alert_type": "billing"' in body, (
        "P1-BILLING-FAIL-LOUD: alert_type debe ser 'billing' para que "
        "dashboards/queries puedan filtrar por categoría."
    )
