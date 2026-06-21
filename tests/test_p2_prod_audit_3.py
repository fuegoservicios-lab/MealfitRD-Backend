"""[P2-PROD-AUDIT-3 · 2026-05-12] Bundle test parser-based para los 3 P2
del audit production-readiness post-P1-BILLING-FAIL-LOUD:

  1. P2-HEALTH-UID-STRIP — `backend/routers/system.py` debe hashear los
     UUIDs (`last_user_id`, `plan_id` en top_plans_24h) antes de devolver
     al cliente. Pre-fix: endpoints públicos `/atomic-pool-health` y
     `/tz-fallback-health` exponían UUIDs literales enumerables.

  2. P2-AUTH-ASYNC-SLEEP — `backend/auth.py::get_verified_user_id` debe
     ser `async def` y usar `await asyncio.sleep` + `asyncio.to_thread`.
     Pre-fix: `time.sleep(0.5)` sync bloqueaba worker thread durante el
     retry transient + la call `supabase.auth.get_user(token)` sync
     bloqueaba durante la HTTP roundtrip.

  3. P2-WEBHOOK-FAIL-SECURE-ALWAYS — `backend/routers/billing.py` webhook
     PayPal debe rechazar SIEMPRE cuando faltan env vars (incluyendo
     sandbox). Pre-fix: rama `if not is_sandbox: raise` permitía a sandbox
     procesar eventos sin firma → atacante podía forge eventos arbitrarios.
     Escape hatch explícito: knob `MEALFIT_ALLOW_WEBHOOK_UNSIGNED=1` solo
     se respeta cuando ENVIRONMENT != production.

Tooltip-anchor: P2-PROD-AUDIT-3-TESTS
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AUTH_PY = _BACKEND_ROOT / "auth.py"
_SYSTEM_PY = _BACKEND_ROOT / "routers" / "system.py"
_BILLING_PY = _BACKEND_ROOT / "routers" / "billing.py"


@pytest.fixture(scope="module")
def auth_src() -> str:
    return _AUTH_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def system_src() -> str:
    return _SYSTEM_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def billing_src() -> str:
    return _BILLING_PY.read_text(encoding="utf-8")


# ============================================================================
# Section 1: P2-HEALTH-UID-STRIP
# ============================================================================

def test_health_uid_strip_helper_defined(system_src: str):
    """`_hash_uuid_for_public` debe existir y usar `hashlib.sha256`."""
    assert "def _hash_uuid_for_public(" in system_src, (
        "P2-HEALTH-UID-STRIP: helper `_hash_uuid_for_public` no definido en "
        "system.py. Sin él, los endpoints públicos pueden volver a exponer "
        "UUIDs literales por copy-paste descuidado."
    )
    assert "hashlib.sha256" in system_src, (
        "P2-HEALTH-UID-STRIP: el helper no usa `hashlib.sha256`. Usar "
        "hashing criptográfico para que el digest no sea reversible."
    )


def test_atomic_pool_health_no_raw_user_id(system_src: str):
    """`/atomic-pool-health` NO debe spread el snapshot crudo (que
    contiene `last_user_id` UUID). Debe popear esa key primero."""
    # Aislar el handler get_atomic_pool_health
    handler_re = re.compile(
        r'@router\.get\("/atomic-pool-health"\)([\s\S]+?)(?=@router\.get|\Z)'
    )
    m = handler_re.search(system_src)
    assert m is not None, "Handler /atomic-pool-health no encontrado."
    body = m.group(1)
    # Debe popear el UUID antes del spread
    assert 'snapshot.pop("last_user_id"' in body, (
        "P2-HEALTH-UID-STRIP: /atomic-pool-health no extrae "
        "`last_user_id` del snapshot antes de devolverlo. El spread "
        "`**snapshot` expondría el UUID literal."
    )
    # Debe usar el hash
    assert "last_user_hash" in body, (
        "P2-HEALTH-UID-STRIP: /atomic-pool-health debe incluir "
        "`last_user_hash` (SHA-256 truncado) en lugar del UUID raw."
    )
    assert "_hash_uuid_for_public(" in body, (
        "P2-HEALTH-UID-STRIP: /atomic-pool-health no llama "
        "`_hash_uuid_for_public`."
    )


def test_tz_fallback_health_hashes_plan_id(system_src: str):
    """`/tz-fallback-health` debe devolver `plan_hash` en lugar de `plan_id`
    en `top_plans_24h`."""
    handler_re = re.compile(
        r'@router\.get\("/tz-fallback-health"\)([\s\S]+?)(?=@router\.get|\Z)'
    )
    m = handler_re.search(system_src)
    assert m is not None, "Handler /tz-fallback-health no encontrado."
    body = m.group(1)
    # Patrón legacy prohibido (plan_id literal en la response)
    legacy_pattern = re.compile(r'"plan_id"\s*:\s*pid')
    assert legacy_pattern.search(body) is None, (
        "P2-HEALTH-UID-STRIP regresión: `/tz-fallback-health` vuelve a "
        "devolver `plan_id: pid` (UUID literal). Reemplazar con "
        "`plan_hash: _hash_uuid_for_public(pid)`."
    )
    # Patrón nuevo presente
    assert "plan_hash" in body, (
        "P2-HEALTH-UID-STRIP: `/tz-fallback-health` no expone "
        "`plan_hash`. Sin él, dashboards no pueden correlacionar planes "
        "recurrentes."
    )
    assert "_hash_uuid_for_public(pid)" in body, (
        "P2-HEALTH-UID-STRIP: `/tz-fallback-health` no hashea `pid` con "
        "el helper SSOT. Drift de implementación."
    )


# ============================================================================
# Section 2: P2-AUTH-ASYNC-SLEEP
# ============================================================================

def test_get_verified_user_id_is_async(auth_src: str):
    """`get_verified_user_id` debe ser `async def`."""
    sync_pattern = re.compile(
        r"^def\s+get_verified_user_id\s*\(", re.MULTILINE
    )
    assert sync_pattern.search(auth_src) is None, (
        "P2-AUTH-ASYNC-SLEEP regresión: `get_verified_user_id` volvió a "
        "ser `def` (sync). Pre-fix `time.sleep(0.5)` bloqueaba el worker "
        "thread durante el retry. Convertir a `async def` + "
        "`await asyncio.sleep`."
    )
    async_pattern = re.compile(
        r"^async\s+def\s+get_verified_user_id\s*\(", re.MULTILINE
    )
    assert async_pattern.search(auth_src) is not None, (
        "P2-AUTH-ASYNC-SLEEP: `get_verified_user_id` no es `async def`."
    )


def test_no_time_sleep_in_auth(auth_src: str):
    """`time.sleep(...)` (BLOQUEANTE) está prohibido en auth.py — congela el event
    loop. [P1-FIRST-PARTY-SESSION · 2026-06-16] `import time` SÍ se permite ahora:
    se usa `time.time()` (no-bloqueante) para los timestamps iat/exp y el cap
    absoluto de la cookie de sesión first-party. El hazard real (time.sleep, que
    motivó P2-AUTH-ASYNC-SLEEP) sigue prohibido — usar `await asyncio.sleep`."""
    assert "time.sleep(" not in auth_src, (
        "P2-AUTH-ASYNC-SLEEP regresión: `time.sleep(...)` reintroducido. "
        "Bloquea el event loop. Usar `await asyncio.sleep(...)`."
    )


def test_uses_asyncio_sleep_and_to_thread(auth_src: str):
    """[P1-NEON-AUTH-MIGRATION · 2026-06-13] La verificación de identidad
    despacha a thread para no bloquear el event loop.

    Pre-Neon (Supabase): la call sync HTTP de validación se envolvía en
    `to_thread` + un retry con `asyncio.sleep` ante flakiness de red. Con
    Neon Auth la verificación es LOCAL contra el JWKS cacheado
    (`verify_neon_jwt`): el único I/O es el fetch ocasional del JWKS, que
    sigue yendo por `to_thread`; el retry-con-sleep ya NO aplica (no hay
    roundtrip de red por request que falle de forma transitoria)."""
    assert "import asyncio" in auth_src, (
        "P2-AUTH-ASYNC-SLEEP: falta `import asyncio` en auth.py."
    )
    assert re.search(r"to_thread\(\s*verify_neon_jwt", auth_src) is not None, (
        "P1-NEON-AUTH: la verificación `verify_neon_jwt` no está envuelta en "
        "`asyncio.to_thread`. Sin eso, el worker bloquea durante el fetch del "
        "JWKS (primer request / rotación de kid)."
    )


def test_auth_anchor_present(auth_src: str):
    assert "P2-AUTH-ASYNC-SLEEP" in auth_src, (
        "P2-AUTH-ASYNC-SLEEP: anchor desapareció — restaurar para "
        "drift-detection futura."
    )


# ============================================================================
# Section 3: P2-WEBHOOK-FAIL-SECURE-ALWAYS
# ============================================================================

def test_webhook_no_sandbox_bypass_without_knob(billing_src: str):
    """El patrón legacy `if not is_sandbox: raise HTTPException(400)` (que
    permitía a sandbox saltar la verificación) está PROHIBIDO sin un knob
    explícito que lo gobierne."""
    # Patrón legacy: missing keys + sandbox = silently process the event.
    # Lo detectamos buscando el bloque legacy específico.
    legacy_pattern = re.compile(
        r"if\s+not\s+PAYPAL_WEBHOOK_ID\s+or\s+not\s+PAYPAL_CLIENT_ID\s+or\s+not\s+PAYPAL_SECRET\s*:"
        r"\s*\n\s*logger\.warning\([^)]*\)\s*\n"
        r"\s*if\s+not\s+is_sandbox\s*:\s*\n\s*raise\s+HTTPException\(status_code=400",
        re.MULTILINE,
    )
    assert legacy_pattern.search(billing_src) is None, (
        "P2-WEBHOOK-FAIL-SECURE-ALWAYS regresión: el bloque legacy "
        "`if not is_sandbox: raise HTTPException(400)` reaparece. Eso "
        "permite a sandbox procesar eventos PayPal sin firma → atacante "
        "puede forge eventos de downgrade. Reemplazar por gate knob-aware: "
        "`if not is_sandbox or not allow_unsigned: raise 503`."
    )


def test_webhook_uses_explicit_knob(billing_src: str):
    """El gate fail-secure debe leer `MEALFIT_ALLOW_WEBHOOK_UNSIGNED` y
    levantar 503 cuando faltan env vars."""
    assert "MEALFIT_ALLOW_WEBHOOK_UNSIGNED" in billing_src, (
        "P2-WEBHOOK-FAIL-SECURE-ALWAYS: knob "
        "`MEALFIT_ALLOW_WEBHOOK_UNSIGNED` no presente. Sin él, dev local "
        "sin credenciales no tiene escape hatch — pero el knob NO debe "
        "respetarse en producción."
    )
    # El gate debe levantar 503 ligado a las env vars PayPal
    near_503_pattern = re.compile(
        r"PAYPAL_WEBHOOK_ID[\s\S]{0,2000}status_code=503"
        r"|status_code=503[\s\S]{0,2000}PAYPAL_WEBHOOK_ID",
    )
    assert near_503_pattern.search(billing_src), (
        "P2-WEBHOOK-FAIL-SECURE-ALWAYS: no encontramos `HTTPException(503)` "
        "cerca del check de `PAYPAL_WEBHOOK_ID`. El handler debe rechazar "
        "antes de procesar."
    )


def test_webhook_anchor_present(billing_src: str):
    assert "P2-WEBHOOK-FAIL-SECURE-ALWAYS" in billing_src, (
        "P2-WEBHOOK-FAIL-SECURE-ALWAYS: anchor desapareció. Restaurar."
    )
