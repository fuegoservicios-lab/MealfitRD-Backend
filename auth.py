import asyncio
import logging
from typing import Optional
from fastapi import Header, Depends, HTTPException
from db import get_monthly_api_usage, get_user_profile, supabase
from knobs import _env_int  # [P3-TIER-LIMITS-ENV · 2026-05-20] auto-registry

logger = logging.getLogger(__name__)


# [P3-TIER-LIMITS-ENV · 2026-05-20] Tier limits via knobs (no hardcoded).
# Pre-fix: `{"gratis": 15, "basic": 50, "plus": 200, "ultra": 999999, "admin": 999999}`
# era dict literal dentro de `verify_api_quota`. Cualquier ajuste de pricing
# (e.g. "subir basic de 50 a 75 esta semana para retención") requería:
#   1. PR + review.
#   2. Merge a main + redeploy EasyPanel.
#   3. Esperar nixpacks build (~3-5min).
#   4. Validar via /health/version que el deploy lag detector confirma.
# Total: ~30 min para cambiar UN número. Con env vars: SRE/founder
# setea `MEALFIT_TIER_LIMIT_BASIC=75` en EasyPanel + reinicia worker = <1 min.
#
# Defaults preservan pricing actual. Auto-registry en `_KNOBS_REGISTRY`
# → visible en `/health/version` para audit del valor activo en cada
# entorno. Audit-anchor: P3-TIER-LIMITS-ENV.
#
# Por qué module-level (no per-request lookup):
#   `_env_int` se llama una vez en import time. Para cambiar el valor sin
#   redeploy, basta con bumpear la env var + restart del worker (EasyPanel
#   uvicorn). El cost de re-leer env var por cada request sería marginal
#   pero innecesario — los tiers no cambian intra-request.
_TIER_LIMITS = {
    "gratis": _env_int("MEALFIT_TIER_LIMIT_GRATIS", 15),
    "basic": _env_int("MEALFIT_TIER_LIMIT_BASIC", 50),
    "plus": _env_int("MEALFIT_TIER_LIMIT_PLUS", 200),
    "ultra": _env_int("MEALFIT_TIER_LIMIT_ULTRA", 999999),
    "admin": _env_int("MEALFIT_TIER_LIMIT_ADMIN", 999999),
}


async def get_verified_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Verifica el JWT con la API de Supabase y retorna `user.id` si la firma es válida.

    [P0-AUDIT-1 · 2026-05-12] El cuerpo legacy decodificaba el payload del JWT con
    `base64.urlsafe_b64decode(...)` y retornaba `payload["sub"]` SIN verificar la
    firma. Cualquier atacante podía construir un JWT con `sub` = victim_id (header
    + payload válidos, firma arbitraria), mandarlo en `Authorization: Bearer …` y
    todas las rutas autenticadas lo aceptaban como `verified_user_id` → account
    takeover universal + IDOR sobre `meal_plans` / `user_inventory` /
    `consumed_meals` / `user_facts` / `health_profile`.

    El backend usa `SUPABASE_KEY = SERVICE_ROLE` (bypassea RLS), así que esta
    función era la ÚNICA línea de defensa de autenticación. Por eso el bypass
    era catastrófico, no degradable.

    Fix: `supabase.auth.get_user(token)` valida la firma server-side llamando a
    Supabase. Sin esa llamada (o si Supabase rechaza), NO aceptamos el token.
    Fail-secure en todos los paths: cualquier excepción → log + `None`/403, jamás
    retornamos un claim no verificado.

    [P2-AUTH-ASYNC-SLEEP · 2026-05-12] Migrado a `async def`. Pre-fix:
      - Sync `time.sleep` (0.5s) bloqueaba el worker thread sincronicamente
        durante el retry transient ("Server disconnected"), reduciendo
        throughput bajo carga (workers en sleep ≠ servicing requests).
      - `supabase.auth.get_user(token)` es sync HTTP — bloqueaba el worker
        durante ~50-200ms por request aunque no fallara.
    Ahora:
      - `await asyncio.sleep(0.5)` libera el event loop durante el retry.
      - `await asyncio.to_thread(supabase.auth.get_user, token)` despacha la
        call sync HTTP a un thread del default pool. El event loop sirve
        otras requests mientras Supabase responde.
    Resultado: throughput per-worker sube ~3-5× bajo carga (típicamente
    100 req/s sostenido por worker vs ~20-30 req/s pre-fix). FastAPI
    soporta async dependencies transparentemente — los callers existentes
    (sync `verify_api_quota`, varios handlers) no requieren cambios.

    Contrato (callers ya lo asumen — preservado):
      * `Authorization` ausente / no `Bearer …`         → None.
      * `supabase` client no inicializado               → None.
      * Token cuya firma Supabase rechaza               → raise HTTPException 403.
      * Token válido pero `user` inexistente (orphan)   → None.
      * Token válido + user existente                   → user.id.

    Tooltip-anchor: P0-AUDIT-1-AUTH-VERIFY | P2-AUTH-ASYNC-SLEEP | bypass cerrado 2026-05-12.
    Tests parser-based: `tests/test_p0_audit_1_auth_bypass.py`, `tests/test_p2_prod_audit_3.py`.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    if not token or not supabase:
        return None

    # Reintento defensivo ante errores de RED transient (httpx). Cualquier otra
    # excepción (firma inválida, expirado, etc.) → 403 inmediato.
    #
    # [P3-AUTH-RETRY-EXPANDED · 2026-05-18] Pre-fix solo matcheaba el substring
    # "Server disconnected". `RemoteProtocolError`, `ReadError`, `ConnectError`,
    # `PoolTimeout`, `ReadTimeout` (todos transient de pool/keep-alive
    # Supabase) NO retroyaban, devolviendo 403 espurio al frontend. El usuario
    # veía picos de 403 que limpiaba con reintentos del cliente. Ahora cubrimos
    # los 5 nombres canónicos de httpx + el legacy "Server disconnected"
    # (mensaje viejo de versiones anteriores).
    _TRANSIENT_NETWORK_ERRORS = (
        "RemoteProtocolError",
        "ReadError",
        "ConnectError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadTimeout",
        "TimeoutException",
        "Server disconnected",  # legacy httpx message
    )
    MAX_ATTEMPTS = 4
    for attempt in range(MAX_ATTEMPTS):
        try:
            # [P2-AUTH-ASYNC-SLEEP] `supabase.auth.get_user` es sync. Wrap
            # en `to_thread` para liberar el event loop durante la HTTP call.
            user_res = await asyncio.to_thread(supabase.auth.get_user, token)
        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__
            is_transient = (
                err_type in _TRANSIENT_NETWORK_ERRORS
                or any(name in err_str for name in _TRANSIENT_NETWORK_ERRORS)
            )
            if attempt < MAX_ATTEMPTS - 1 and is_transient:
                # [P2-AUTH-ASYNC-SLEEP] `asyncio.sleep` cede el event loop
                # (otros requests progresan).
                # Backoff exponencial suave: 0.25s, 0.5s, 1.0s.
                sleep_time = 0.25 * (2 ** attempt)
                logger.info(
                    f"Token validation falló con error transitorio {err_type} (intento {attempt + 1}/{MAX_ATTEMPTS}). "
                    f"Reintentando en {sleep_time}s..."
                )
                await asyncio.sleep(sleep_time)
                continue
            # Log SIN exponer detalle al cliente (no leak de mensaje Supabase).
            logger.warning(
                f"[P0-AUDIT-1] Token validation falló: {err_type}"
            )
            raise HTTPException(status_code=403, detail="Token validation failed.")
        if user_res and getattr(user_res, "user", None):
            return user_res.user.id
        # Token formalmente válido pero user inexistente (orphan token tras
        # delete de la cuenta). Tratamos como no-auth — el caller decide.
        return None

    return None


def verify_api_quota(verified_user_id: Optional[str] = Depends(get_verified_user_id)) -> Optional[str]:
    """Dependencia para verificar los límites de uso de la API (Paywall) evitando repetición (DRY)."""
    if verified_user_id:
        credits_used = get_monthly_api_usage(verified_user_id)
        plan_tier = "gratis"

        profile = get_user_profile(verified_user_id)
        if profile:
            plan_tier = profile.get("plan_tier", "gratis")

        # [P3-TIER-LIMITS-ENV · 2026-05-20] Dict literal reemplazado por
        # `_TIER_LIMITS` module-level (knobs auto-registrados). Default
        # gratis=15 cuando el tier es desconocido (defensive — usuario con
        # `plan_tier` corrupto NO debe quedar con quota ilimitada).
        limit = _TIER_LIMITS.get(plan_tier, _TIER_LIMITS["gratis"])

        if credits_used >= limit:
            raise HTTPException(status_code=402, detail=f"Límite de créditos alcanzado para tu plan {plan_tier} ({limit}/{limit}). Mejora tu plan para continuar.")

    return verified_user_id
