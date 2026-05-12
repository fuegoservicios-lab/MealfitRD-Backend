import os
import json
import logging
import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from error_utils import safe_error_detail
from typing import Optional

# Imports relativos al backend
from auth import get_verified_user_id
from db import supabase
from rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# [P1-BILLING-3 · 2026-05-12] Rate-limiter para /api/discount/validate.
# Pre-fix: endpoint público sin auth ni throttle → atacante brute-force
# de la tabla `discount_codes` enumerando códigos válidos por chunks
# alfabéticos sin coste. 20 calls/min/user es generoso para UX legítima
# (probar 1-2 códigos por sesión) y bloqueante para enumeración. Anchor:
# P1-BILLING-3-DISCOUNT-RATELIMIT.
_DISCOUNT_VALIDATE_LIMITER = RateLimiter(max_calls=20, period_seconds=60)

router = APIRouter(
    prefix="/api/subscription",
    tags=["billing"],
)

# El webhook pasivo de paypal no puede tener el prefix /api/subscription si original era /api/webhooks/paypal
# Así que lo añadiremos directamente a la raíz de la app o podemos dejar que el router maneje '/api', pero el router ya tiene prefix.
# Vamos a crear otro router para webhooks o manejarlo aquí sin prefix.
webhooks_router = APIRouter(
    prefix="/api/webhooks",
    tags=["webhooks"]
)

discount_router = APIRouter(
    prefix="/api/discount",
    tags=["discount"],
)

@discount_router.post("/validate")
async def api_validate_discount(
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(_DISCOUNT_VALIDATE_LIMITER),
):
    """Valida un código de descuento contra la tabla discount_codes en Supabase.

    [P1-BILLING-3 · 2026-05-12] Pre-fix el endpoint era público (sin
    `Depends(get_verified_user_id)`) y sin rate-limit. Un atacante anónimo
    podía enumerar `discount_codes` por brute-force alfabético sin coste —
    cada código válido encontrado era $X gratis para usar luego con
    `/verify`. Ahora:

      - `Depends(_DISCOUNT_VALIDATE_LIMITER)`: 20 calls/min/user (o IP si
        no autenticado). El limiter inyecta `verified_user_id` y bloquea
        bursts. Backed por Redis cuando disponible; in-memory fallback.
      - Auth obligatoria (`if not verified_user_id: 401`). Sin auth NO se
        valida — los descuentos solo se aplican a usuarios registrados.

    Anchor: P1-BILLING-3-DISCOUNT-RATELIMIT
    """
    if not verified_user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required to validate discount codes.",
        )
    try:
        code = (data.get("code") or "").strip().upper()
        tier = (data.get("tier") or "").strip().lower()

        if not code:
            raise HTTPException(status_code=400, detail="Código requerido")

        # Buscar el código en la tabla
        res = supabase.table("discount_codes").select("*").eq("code", code).eq("is_active", True).execute()

        if not res.data or len(res.data) == 0:
            return {"valid": False, "message": "Código no encontrado o inactivo."}

        discount = res.data[0]

        # Verificar vigencia
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        if discount.get("valid_from"):
            valid_from = datetime.fromisoformat(discount["valid_from"].replace("Z", "+00:00"))
            if now < valid_from:
                return {"valid": False, "message": "Este código aún no está activo."}

        if discount.get("valid_until"):
            valid_until = datetime.fromisoformat(discount["valid_until"].replace("Z", "+00:00"))
            if now > valid_until:
                return {"valid": False, "message": "Este código ha expirado."}

        # Verificar usos
        if discount.get("max_uses") is not None:
            if discount.get("current_uses", 0) >= discount["max_uses"]:
                return {"valid": False, "message": "Este código ya alcanzó su límite de usos."}

        # Verificar tier aplicable
        applicable = discount.get("applicable_tiers", [])
        if applicable and tier and tier not in applicable:
            return {"valid": False, "message": f"Este código no aplica al plan seleccionado."}

        logger.info(f"✅ Código de descuento '{code}' validado: {discount['discount_percent']}% off")

        return {
            "valid": True,
            "discount_percent": discount["discount_percent"],
            "message": f"¡{discount['discount_percent']}% de descuento aplicado!"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error validando código de descuento: {str(e)}")
        raise HTTPException(status_code=500, detail="Error validando código")

# ---------------------------------------------------------------------------
# [P0-BILLING-1 · 2026-05-12] Mapping PayPal plan_id → tier interno.
# Construido server-side desde env vars. El cliente NO puede inyectar tier:
# pre-fix, el handler hacía `UPDATE user_profiles SET plan_tier = body.tier`
# confiando en el cliente → cualquier user con sub real de "basic" podía
# pasar tier="ultra" y obtener upgrade gratuito.
# Anchor: P0-BILLING-1-TIER-SERVER-SIDE.
#
# Si una env var falta, esa entrada se omite del mapping — verify_subscription
# rechazará el plan_id correspondiente con 400 ("Plan no reconocido").
# Operador debe setear las 3 env vars en producción.
# ---------------------------------------------------------------------------

def _build_paypal_plan_tier_map() -> dict[str, str]:
    mapping = {}
    for tier in ("basic", "plus", "ultra"):
        plan_id = os.environ.get(f"PAYPAL_PLAN_{tier.upper()}_ID")
        if plan_id:
            mapping[plan_id] = tier
    return mapping


@router.post("/verify")
async def api_verify_subscription(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        subscription_id = data.get("subscriptionID")
        client_hint_tier = (data.get("tier") or "").strip().lower()

        if not user_id or not subscription_id or not client_hint_tier:
            raise HTTPException(status_code=400, detail="Missing parameters")

        if not verified_user_id or verified_user_id != user_id:
            logger.warning(f"Intento no autorizado de verificar suscripcion: req_user={user_id}, auth_user={verified_user_id}")
            raise HTTPException(status_code=401, detail="No autorizado.")

        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        is_sandbox = os.environ.get("ENVIRONMENT") != "production"
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"

        # [P0-BILLING-2 · 2026-05-12] Fail-secure si faltan env vars PayPal
        # en producción. Pre-fix: cuando ambas keys eran falsy, el handler
        # forzaba `success` a True (bypass completo de verificación). Si el
        # contenedor perdía las env vars (rotación rota, misconfig Easypanel),
        # CUALQUIER POST con `subscription_id` arbitrario upgradeaba el plan.
        # Anchor: P0-BILLING-2-FAIL-SECURE.
        env_ready = bool(PAYPAL_CLIENT_ID and PAYPAL_SECRET)
        allow_bypass = (
            os.environ.get("MEALFIT_ALLOW_PAYPAL_BYPASS", "").lower()
            in ("1", "true", "yes")
        )
        if not env_ready and not is_sandbox and not allow_bypass:
            logger.error(
                "❌ [P0-BILLING-2] PAYPAL_CLIENT_ID/PAYPAL_SECRET ausentes en "
                "producción. Configurar env vars o (SOLO dev) setear "
                "MEALFIT_ALLOW_PAYPAL_BYPASS=true."
            )
            raise HTTPException(
                status_code=503,
                detail="Payment provider misconfigured. Contact support.",
            )

        access_token = None
        verified_plan_id = None

        if env_ready:
            async with httpx.AsyncClient() as client:
                auth_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/oauth2/token",
                    auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                    data={"grant_type": "client_credentials"}
                )
                if auth_resp.status_code != 200:
                    logger.error(f"Error auth Paypal: {auth_resp.text}")
                    raise HTTPException(status_code=500, detail="Error de autenticación con proveedor de pagos.")

                access_token = auth_resp.json().get("access_token")

                sub_resp = await client.get(
                    f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                if sub_resp.status_code != 200:
                    logger.error(f"Error fetching sub: {sub_resp.text}")
                    raise HTTPException(status_code=400, detail="La suscripción no fue reconocida por PayPal.")

                sub_data = sub_resp.json()
                status = sub_data.get("status")

                if status != "ACTIVE":
                    raise HTTPException(status_code=400, detail=f"Suscripción no válida. Estado actual: {status}")

                verified_plan_id = sub_data.get("plan_id")
        else:
            logger.warning(
                "⚠️ [DEV-BYPASS] PayPal env vars ausentes; bypass permitido por "
                "sandbox/MEALFIT_ALLOW_PAYPAL_BYPASS. NO en producción."
            )

        # [P0-BILLING-1] Derivar tier server-side desde verified_plan_id, NO
        # desde data.get("tier"). El client_hint_tier solo se usa para log de
        # divergencia (no afecta el UPDATE).
        if verified_plan_id:
            plan_tier_map = _build_paypal_plan_tier_map()
            server_tier = plan_tier_map.get(verified_plan_id)
            if not server_tier:
                logger.warning(
                    f"[P0-BILLING-1] PayPal plan_id={verified_plan_id!r} no "
                    f"mapea a tier interno. Configurar env vars "
                    f"PAYPAL_PLAN_BASIC_ID/PAYPAL_PLAN_PLUS_ID/"
                    f"PAYPAL_PLAN_ULTRA_ID. cliente hint: {client_hint_tier!r}."
                )
                raise HTTPException(
                    status_code=400,
                    detail="Plan no reconocido. Contacta soporte.",
                )
            if client_hint_tier and client_hint_tier != server_tier:
                logger.warning(
                    f"[P0-BILLING-1] tier mismatch: cliente={client_hint_tier!r} "
                    f"vs server={server_tier!r} (plan_id={verified_plan_id}). "
                    f"Asignando server-side."
                )
            tier_to_assign = server_tier
        elif allow_bypass or is_sandbox:
            if client_hint_tier not in ("basic", "plus", "ultra"):
                raise HTTPException(status_code=400, detail="tier inválido.")
            tier_to_assign = client_hint_tier
        else:
            # Defensa-en-profundidad: el gate de arriba ya levantó 503 si
            # env_ready=False AND not sandbox AND not bypass. Si llegamos
            # acá hay un bug — fallar antes de mutar.
            raise HTTPException(status_code=500, detail="No se pudo derivar tier.")

        logger.info(
            f"✅ Subscripcion Verificada B2B ({subscription_id}) para usuario "
            f"{user_id}. Tier asignado server-side: {tier_to_assign}"
        )

        existing_res = supabase.table("user_profiles").select("paypal_subscription_id, subscription_status").eq("id", user_id).execute()
        if existing_res.data and len(existing_res.data) > 0:
            old_sub_id = existing_res.data[0].get("paypal_subscription_id")
            old_status = existing_res.data[0].get("subscription_status")

            if old_sub_id and old_sub_id != subscription_id and old_status != "INACTIVE":
                logger.info(f"🔄 Detectado Upgrade/Cambio. Cancelando suscripción antigua {old_sub_id} en PayPal...")
                if access_token:
                    async with httpx.AsyncClient() as cancel_client:
                        cancel_resp = await cancel_client.post(
                            f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{old_sub_id}/cancel",
                            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                            json={"reason": "Upgrade o cambio a una nueva suscripción."}
                        )
                        if cancel_resp.status_code == 204:
                            logger.info(f"✅ Suscripción antigua {old_sub_id} cancelada exitosamente en PayPal.")
                        else:
                            logger.warning(f"⚠️ Error intentando cancelar suscripción antigua {old_sub_id}: {cancel_resp.text}")

        res = supabase.table("user_profiles").update({
            "plan_tier": tier_to_assign,
            "paypal_subscription_id": subscription_id,
            "subscription_status": "ACTIVE"
        }).eq("id", user_id).execute()

        return {"success": True, "message": "Suscripción verificada y plan actualizado B2B."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error interno en /api/subscription/verify: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

@router.post("/cancel")
async def api_cancel_subscription(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        
        if not verified_user_id or verified_user_id != user_id:
            logger.warning(f"Intento no autorizado de cancelar suscripcion: req_user={user_id}, auth_user={verified_user_id}")
            raise HTTPException(status_code=401, detail="No autorizado.")
            
        res = supabase.table("user_profiles").select("paypal_subscription_id").eq("id", user_id).execute()
        if not res.data or not res.data[0].get("paypal_subscription_id"):
            raise HTTPException(status_code=400, detail="No active subscription found to cancel.")
            
        subscription_id = res.data[0]["paypal_subscription_id"]
        
        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        is_sandbox = os.environ.get("ENVIRONMENT") != "production"
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"

        # [P0-BILLING-2 · 2026-05-12] Fail-secure en /cancel también: sin env
        # vars PayPal en producción el flujo legacy actualizaba BD a CANCELLED
        # pero NO notificaba PayPal → cobro recurrente sigue mientras BD dice
        # cancelado. Anchor: P0-BILLING-2-FAIL-SECURE.
        env_ready = bool(PAYPAL_CLIENT_ID and PAYPAL_SECRET)
        allow_bypass = (
            os.environ.get("MEALFIT_ALLOW_PAYPAL_BYPASS", "").lower()
            in ("1", "true", "yes")
        )
        if not env_ready and not is_sandbox and not allow_bypass:
            logger.error(
                "❌ [P0-BILLING-2] PAYPAL_CLIENT_ID/PAYPAL_SECRET ausentes en "
                "producción. /cancel rechazado para evitar estado divergente "
                "(BD CANCELLED + PayPal siguiendo cobrando)."
            )
            raise HTTPException(
                status_code=503,
                detail="Payment provider misconfigured. Contact support.",
            )

        end_date = None

        if env_ready:
            async with httpx.AsyncClient() as client:
                auth_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/oauth2/token",
                    auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                    data={"grant_type": "client_credentials"}
                )
                if auth_resp.status_code == 200:
                    access_token = auth_resp.json().get("access_token")
                    
                    sub_info_resp = await client.get(
                        f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}",
                        headers={"Authorization": f"Bearer {access_token}"}
                    )
                    if sub_info_resp.status_code == 200:
                        billing_info = sub_info_resp.json().get("billing_info", {})
                        end_date = billing_info.get("next_billing_time")
                    
                    cancel_resp = await client.post(
                        f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}/cancel",
                        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                        json={"reason": "El usuario solicitó la cancelación desde la App"}
                    )
                    
                    if cancel_resp.status_code not in [204, 200]:
                        logger.error(f"Error cancelando suscripcion con PayPal: {cancel_resp.text}")
        
        logger.info(f"✅ Suscripción {subscription_id} de usuario {user_id} cancelada. Mantendrá acceso hasta {end_date or 'fin de ciclo'}.")
        
        update_payload = {"subscription_status": "CANCELLED"}
        if end_date:
            update_payload["subscription_end_date"] = end_date
            
        supabase.table("user_profiles").update(update_payload).eq("id", user_id).execute()
        
        return {"success": True, "message": "Tu suscripción no se renovará, pero mantendrás tu plan actual hasta el final del ciclo pagado."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en /api/subscription/cancel: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

@webhooks_router.post("/paypal")
async def api_webhook_paypal(request: Request):
    try:
        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        PAYPAL_WEBHOOK_ID = os.environ.get("PAYPAL_WEBHOOK_ID")
        is_sandbox = os.environ.get("ENVIRONMENT") != "production"
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"
        
        body = await request.body()
        headers = request.headers
        
        if not PAYPAL_WEBHOOK_ID or not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
            logger.warning("⚠️ PAYPAL_WEBHOOK_ID o llaves no configuradas. Saltando verificación de firma en webhook.")
            if not is_sandbox:
                raise HTTPException(status_code=400, detail="Missing webhook config")
                
        payload_dict = json.loads(body.decode('utf-8'))
        
        if PAYPAL_CLIENT_ID and PAYPAL_SECRET and PAYPAL_WEBHOOK_ID:
            async with httpx.AsyncClient() as client:
                auth_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/oauth2/token",
                    auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                    data={"grant_type": "client_credentials"}
                )
                if auth_resp.status_code != 200:
                    logger.error("Error autenticando con PayPal en webhook")
                    return {"success": False}
                
                access_token = auth_resp.json().get("access_token")
                
                verify_payload = {
                    "auth_algo": headers.get("paypal-auth-algo"),
                    "cert_url": headers.get("paypal-cert-url"),
                    "transmission_id": headers.get("paypal-transmission-id"),
                    "transmission_sig": headers.get("paypal-transmission-sig"),
                    "transmission_time": headers.get("paypal-transmission-time"),
                    "webhook_id": PAYPAL_WEBHOOK_ID,
                    "webhook_event": payload_dict
                }
                
                verify_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/notifications/verify-webhook-signature",
                    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                    json=verify_payload
                )
                
                if verify_resp.status_code != 200 or verify_resp.json().get("verification_status") != "SUCCESS":
                    logger.warning(f"🔒 Bloqueado: Firma del Webhook PayPal inválida: {verify_resp.text}")
                    return {"success": False, "message": "Signature mismatch"}
        
        event_type = payload_dict.get("event_type")
        resource = payload_dict.get("resource", {})
        
        logger.info(f"⚡ [WEBHOOK PAYPAL] Evento recibido: {event_type}")
        
        downgrade_events = [
            "BILLING.SUBSCRIPTION.SUSPENDED", 
            "BILLING.SUBSCRIPTION.EXPIRED",
            "BILLING.SUBSCRIPTION.PAYMENT.FAILED"
        ]
        
        if event_type in downgrade_events:
            subscription_id = resource.get("id")
            if subscription_id:
                logger.info(f"⬇️ Degradando suscripción {subscription_id} en BD debido a {event_type}.")
                supabase.table("user_profiles").update({
                    "plan_tier": "gratis",
                    "subscription_status": "INACTIVE"
                }).eq("paypal_subscription_id", subscription_id).execute()
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            subscription_id = resource.get("id")
            end_date = resource.get("billing_info", {}).get("next_billing_time")
            
            if subscription_id:
                logger.info(f"ℹ️ Webhook: Suscripción {subscription_id} cancelada remota/silenciosamente.")
                
                update_payload = {"subscription_status": "CANCELLED"}
                if end_date:
                    update_payload["subscription_end_date"] = end_date
                    
                supabase.table("user_profiles").update(update_payload).eq("paypal_subscription_id", subscription_id).execute()
        
        return {"success": True}

    except Exception as e:
        logger.error(f"❌ Error procesando webhook PayPal: {e}")
        return {"success": False}
