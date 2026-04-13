import os
import json
import logging
import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from typing import Optional

# Imports relativos al backend
from auth import get_verified_user_id
from db import supabase

logger = logging.getLogger(__name__)

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
async def api_validate_discount(data: dict = Body(...)):
    """Valida un código de descuento contra la tabla discount_codes en Supabase."""
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

@router.post("/verify")
async def api_verify_subscription(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        subscription_id = data.get("subscriptionID")
        tier = data.get("tier")

        if not user_id or not subscription_id or not tier:
            raise HTTPException(status_code=400, detail="Missing parameters")

        if not verified_user_id or verified_user_id != user_id:
            logger.warning(f"Intento no autorizado de verificar suscripcion: req_user={user_id}, auth_user={verified_user_id}")
            raise HTTPException(status_code=401, detail="No autorizado.")

        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        is_sandbox = os.environ.get("ENVIRONMENT") != "production"
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"

        access_token = None
        success = False

        if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
            logger.warning("⚠️ No Paypal keys found in backend .env. Bypassing real validation! (SECURITY RISK IF PRODUCTION)")
            success = True
        else:
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
                    raise HTTPException(status_code=400, detail="La suscripción no fue recocida por PayPal.")
                
                sub_data = sub_resp.json()
                status = sub_data.get("status")

                if status != "ACTIVE":
                    raise HTTPException(status_code=400, detail=f"Suscripción no válida. Estado actual: {status}")

                success = True

        if success:
            logger.info(f"✅ Subscripcion Verificada B2B ({subscription_id}) para usuario {user_id}. Asignando tier: {tier}")
            
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
                "plan_tier": tier,
                "paypal_subscription_id": subscription_id, 
                "subscription_status": "ACTIVE"
            }).eq("id", user_id).execute()
            
            return {"success": True, "message": "Suscripción verificada y plan actualizado B2B."}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error interno en /api/subscription/verify: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        end_date = None
        
        if PAYPAL_CLIENT_ID and PAYPAL_SECRET:
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
        raise HTTPException(status_code=500, detail=str(e))

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
