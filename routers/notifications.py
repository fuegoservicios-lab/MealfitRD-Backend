from fastapi import APIRouter, HTTPException, Depends, Request
from error_utils import safe_error_detail
import logging
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from auth import get_verified_user_id
from db_core import connection_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])

class PushSubscriptionItem(BaseModel):
    endpoint: str
    expirationTime: Optional[float] = None
    keys: Dict[str, str]

@router.post("/subscribe")
async def subscribe_push(sub: PushSubscriptionItem, user_id: str = Depends(get_verified_user_id)):
    """
    Guarda o actualiza la suscripción de un dispositivo para enviar notificaciones Push.
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID en token no válido.")
    
    if not connection_pool:
        raise HTTPException(status_code=500, detail="Database connection pool unavailable")
    
    try:
        import psycopg
        import json
        
        sub_json = sub.model_dump_json()
        
        with connection_pool.connection() as conn:
            with conn.cursor() as cursor:
                # Comprobar si este endpoint exacto ya existe para el usuario
                cursor.execute(
                    "SELECT id FROM push_subscriptions WHERE user_id = %s AND subscription_data->>'endpoint' = %s",
                    (user_id, sub.endpoint)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Actualizar si hubiera algún cambio en las keys
                    cursor.execute(
                        "UPDATE push_subscriptions SET subscription_data = %s WHERE id = %s",
                        (sub_json, existing[0])
                    )
                else:
                    # Guardar nueva suscripción de dispositivo
                    cursor.execute(
                        "INSERT INTO push_subscriptions (user_id, subscription_data) VALUES (%s, %s)",
                        (user_id, sub_json)
                    )
                conn.commit()
                
        return {"status": "success", "message": "Subscription stored"}

    except Exception as e:
        logger.error(f"Error guardando push subscription: {e}")
        raise HTTPException(status_code=500, detail="Error en servidor guardando suscripción")

@router.delete("/unsubscribe")
async def unsubscribe_push(request: Request, user_id: str = Depends(get_verified_user_id)):
    """
    Elimina una suscripción de dispositivo. Recibe el endpoint de la suscripción para borrar la fila exacta.
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID en token no válido.")
    
    if not connection_pool:
        raise HTTPException(status_code=500, detail="Database connection pool unavailable")
    
    try:
        body = await request.json()
        endpoint_to_remove = body.get("endpoint")
        if not endpoint_to_remove:
            raise HTTPException(status_code=400, detail="Missing endpoint in request")
            
        with connection_pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM push_subscriptions WHERE user_id = %s AND subscription_data->>'endpoint' = %s",
                    (user_id, endpoint_to_remove)
                )
                conn.commit()
                
        return {"status": "success", "message": "Subscription removed"}

    except Exception as e:
        logger.error(f"Error borrando push subscription: {e}")
        raise HTTPException(status_code=500, detail="Error de BDD borrando suscripción")

@router.get("/test")
async def test_push_route(user_id: str, request: Request):
    """[P1-NOTIF-TEST-1 · 2026-05-11] Ruta de depuración admin-only para
    probar el envío forzado de notificaciones Push a todos los dispositivos
    suscritos de un usuario.

    ANTES (pre-fix):
      Sin `Depends(...)`. Cualquiera con la URL pública del backend +
      cualquier `user_id` válido (UUIDs leak fácilmente vía endpoints GET o
      enumeración) podía spamear push notifications a los devices del
      usuario. Sin frontend caller (verificado vía grep) — el endpoint era
      pure debug pero estaba montado en el router de producción.

    DESPUÉS:
      `_verify_admin_token(authorization)` — mismo gate que
      `/api/system/admin/plan-graph/invalidate` y `/api/plans/admin/...`.
      Requiere `Authorization: Bearer <CRON_SECRET>`. Sin CRON_SECRET en
      el ambiente, el endpoint responde 503 (fail-secure: admin endpoints
      no se exponen sin secreto).

    Tooltip-anchor: P1-NOTIF-TEST-1-AUTH
    """
    # Import lazy del helper admin (vive en routers.plans para evitar duplicar
    # CRON_SECRET checking en 5 lugares). Mismo patrón que system.py.
    from routers.plans import _verify_admin_token
    _verify_admin_token(request.headers.get("authorization"))

    if not connection_pool:
        raise HTTPException(status_code=500, detail="Database connection pool unavailable")
    try:
        import os, json
        from pywebpush import webpush, WebPushException  # type: ignore[import-untyped]
        
        vapid_private = os.environ.get("VAPID_PRIVATE_KEY")
        vapid_claim = os.environ.get("VAPID_CLAIM_EMAIL")
        
        if not vapid_private or not vapid_claim:
            return {"status": "error", "message": "Faltan las VAPID keys en el entorno."}

        with connection_pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT subscription_data FROM push_subscriptions WHERE user_id = %s",
                    (user_id,)
                )
                subs = cursor.fetchall()
        
        if not subs:
            return {"status": "error", "message": f"El usuario {user_id} no tiene dispositivos suscritos en la base de datos."}

        push_payload = json.dumps({
            "title": "Aviso de tu Nutricionista IA \U0001f9d1\u200d\u2615",
            "body": "¡Esta es una notificación de prueba manual forzada!",
            "url": f"/dashboard/agent"
        })
        
        success_count = 0
        errors = []
        for row in subs:
            sub_info = row[0]
            if isinstance(sub_info, str):
                sub_info = json.loads(sub_info)
            try:
                webpush(
                    subscription_info=sub_info,
                    data=push_payload,
                    vapid_private_key=vapid_private,
                    vapid_claims={"sub": vapid_claim}
                )
                success_count += 1
            except WebPushException as ex:
                errors.append(repr(ex))
                
        return {
            "status": "success", 
            "message": f"Notificaciones intentadas: {len(subs)}. Éxitos: {success_count}.",
            "errores": errors
        }

    except Exception as e:
        logger.error(f"Error test push subscription: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))
