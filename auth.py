import logging
import time
from typing import Optional
from fastapi import Header, Depends, HTTPException
from db import get_monthly_api_usage, get_user_profile, supabase

logger = logging.getLogger(__name__)


def get_verified_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
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

    Contrato (callers ya lo asumen — preservado):
      * `Authorization` ausente / no `Bearer …`         → None.
      * `supabase` client no inicializado               → None.
      * Token cuya firma Supabase rechaza               → raise HTTPException 403.
      * Token válido pero `user` inexistente (orphan)   → None.
      * Token válido + user existente                   → user.id.

    Tooltip-anchor: P0-AUDIT-1-AUTH-VERIFY | bypass cerrado 2026-05-12.
    Test parser-based: `tests/test_p0_audit_1_auth_bypass.py`.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    if not token or not supabase:
        return None

    # Reintento defensivo SOLO ante "Server disconnected" transient (httpx).
    # Cualquier otra excepción (firma inválida, expirado, etc.) → 403 inmediato.
    for attempt in range(2):
        try:
            user_res = supabase.auth.get_user(token)
        except Exception as e:
            err_str = str(e)
            if attempt == 0 and "Server disconnected" in err_str:
                time.sleep(0.5)
                continue
            # Log SIN exponer detalle al cliente (no leak de mensaje Supabase).
            logger.warning(
                f"[P0-AUDIT-1] Token validation falló: {type(e).__name__}"
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

        tier_limits = {"gratis": 15, "basic": 50, "plus": 200, "ultra": 999999, "admin": 999999}
        limit = tier_limits.get(plan_tier, 15)

        if credits_used >= limit:
            raise HTTPException(status_code=402, detail=f"Límite de créditos alcanzado para tu plan {plan_tier} ({limit}/{limit}). Mejora tu plan para continuar.")

    return verified_user_id
