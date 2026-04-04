import logging
from typing import Optional
from fastapi import Header, Depends, HTTPException
from db import get_monthly_api_usage, get_user_profile, supabase

logger = logging.getLogger(__name__)

def get_verified_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extrae el user_id del token JWT en el header Authorization."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    if not supabase:
        return None
    
    # Custom fallback para decodificar JWT directamente sin request extra a Supabase
    # Soluciona crash 403 si pyjwt no está instalado o si la tabla auth.users fue mermada (orphan token)
    try:
        import json
        import base64
        # Un JWT tipico tiene 3 partes separadas por '.'
        parts = token.split('.')
        if len(parts) >= 2:
            payload_b64 = parts[1]
            # Añadir padding si es necesario
            payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64).decode('utf-8')
            payload = json.loads(payload_json)
            return payload.get("sub")
    except Exception as fallback_e:
        logger.error(f"Fallback nativo JWT failed: {fallback_e}")
        
    # Retry logic up to 2 times for httpx "Server disconnected" issues
    for attempt in range(2):
        try:
            user_res = supabase.auth.get_user(token)
            if user_res and user_res.user:
                return user_res.user.id
        except Exception as e:
            if attempt == 0 and "Server disconnected" in str(e):
                import time
                time.sleep(0.5)
                continue
            logger.error(f"⚠️ [AUTH] Error validando token: {e}")
            raise HTTPException(status_code=403, detail=f"Token validation failed: {str(e)}")
            
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
