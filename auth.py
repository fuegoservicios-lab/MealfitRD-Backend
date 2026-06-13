import asyncio
import logging
from typing import Optional
from fastapi import Header, Depends, HTTPException
from db import (
    ensure_user_profile_exists,
    get_monthly_api_usage,
    get_user_profile,
)
from neon_auth import verify_neon_jwt  # [P1-NEON-AUTH-MIGRATION · 2026-06-13]
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
    """Verifica el JWT de Neon Auth y retorna `sub` (user_id) si la firma es válida.

    [P1-NEON-AUTH-MIGRATION · 2026-06-13] Reemplaza la verificación que antes
    hacía la API de Supabase Auth. Neon Auth (Better Auth) emite JWTs
    EdDSA (Ed25519) validados LOCALMENTE contra el JWKS público cacheado
    (`neon_auth.verify_neon_jwt`) — sin roundtrip de red por request en estado
    caliente (mejora vs. Supabase, que llamaba a su API cada vez). El `sub` del
    payload es el user_id (UUID), misma clave que `public.user_profiles.id`.

    [P0-AUDIT-1 · 2026-05-12] La invariante de seguridad se preserva intacta:
    NUNCA se decodifica el payload sin verificar la firma. `verify_neon_jwt`
    valida firma + `iss` + `aud` + `exp` con el algoritmo FIJO `["EdDSA"]`
    (sin algorithm-confusion ni `none`) antes de retornar claims. El backend
    conecta a Neon con un rol que bypassa RLS, así que esta función sigue
    siendo la ÚNICA capa de auth — el fail-secure es obligatorio: cualquier
    fallo (firma inválida, expirado, kid desconocido, JWKS inalcanzable) →
    `None`, jamás un claim no verificado.

    [P2-AUTH-ASYNC-SLEEP · 2026-05-12] `async def` + `asyncio.to_thread` para la
    verificación: en estado caliente es CPU-only (firma Ed25519) y retorna al
    instante; el único I/O es el fetch ocasional del JWKS (primer request /
    rotación de kid), que `to_thread` mantiene fuera del event loop.

    Contrato (callers ya lo asumen — preservado):
      * `Authorization` ausente / no `Bearer …`         → None.
      * Neon Auth no configurado / token vacío          → None.
      * Token cuya firma o claims son inválidos          → None.
      * Token válido sin `sub`                           → None.
      * Token válido                                     → sub (user_id).

    Tooltip-anchor: P0-AUDIT-1-AUTH-VERIFY | P1-NEON-AUTH-VERIFY | bypass cerrado 2026-05-12.
    Tests parser-based: `tests/test_p0_audit_1_auth_bypass.py`, `tests/test_p2_prod_audit_3.py`.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return None

    # [P1-NEON-AUTH-VERIFY] Verificación local contra el JWKS de Neon Auth.
    # `verify_neon_jwt` JAMÁS lanza — retorna el payload validado o None.
    # `to_thread` despacha el fetch ocasional del JWKS (red) fuera del loop.
    payload = await asyncio.to_thread(verify_neon_jwt, token)
    if not payload:
        return None
    uid = payload.get("sub")
    if not uid:
        # Token válido formalmente pero sin `sub` — no podemos identificar al
        # usuario. Fail-secure: tratamos como no-auth.
        return None

    # [P1-NEON-DB-MIGRATION · 2026-06-12] Garantiza la fila espejo en
    # public.user_profiles (reemplaza el trigger handle_new_user). Cacheado
    # in-process: tras el primer request del usuario es un no-op puro.
    # Best-effort: el JWT ya validó — no bloquear auth si el INSERT falla.
    try:
        await asyncio.to_thread(
            ensure_user_profile_exists,
            uid,
            payload.get("email"),
            payload.get("name"),
        )
    except Exception as ensure_err:
        logger.warning(
            f"[P1-NEON-DB-MIGRATION] ensure_user_profile_exists lanzó "
            f"{type(ensure_err).__name__} (auth continúa)"
        )
    return uid


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
