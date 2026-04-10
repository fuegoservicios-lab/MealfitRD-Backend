from fastapi import APIRouter, HTTPException, Depends, Request
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
