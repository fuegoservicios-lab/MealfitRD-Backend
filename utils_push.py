import os
import json
import logging
from db_core import execute_sql_query, execute_sql_write

logger = logging.getLogger(__name__)

def send_push_notification(user_id: str, title: str, body: str, url: str = "/dashboard") -> bool:
    """
    Sends a web push notification to all subscribed devices for a given user.
    Returns True if at least one notification was attempted successfully.
    """
    try:
        from pywebpush import webpush, WebPushException  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("No se ha instalado 'pywebpush'. Las notificaciones nativas a móviles no se enviarán.")
        return False

    vapid_private = os.environ.get("VAPID_PRIVATE_KEY")
    vapid_claim = os.environ.get("VAPID_CLAIM_EMAIL")

    if not vapid_private or not vapid_claim:
        logger.warning(f"⚠️ [PUSH] Faltan llaves VAPID en el entorno. No se enviará notificación.")
        return False

    try:
        # Buscar las suscripciones de este usuario en DDBB
        subs_query = "SELECT subscription_data FROM push_subscriptions WHERE user_id = %s"
        subs = execute_sql_query(subs_query, (user_id,), fetch_all=True)

        if not subs:
            logger.debug(f"ℹ️ [PUSH] Usuario {user_id} no tiene suscripciones Push activas.")
            return False

        push_payload = json.dumps({
            "title": title,
            "body": body,
            "url": url
        })

        success_count = 0
        for sub_row in subs:
            sub_info = sub_row['subscription_data']
            if isinstance(sub_info, str):
                sub_info = json.loads(sub_info)

            try:
                webpush(
                    subscription_info=sub_info,
                    data=push_payload,
                    vapid_private_key=vapid_private,
                    vapid_claims={"sub": vapid_claim}
                )
                logger.info(f"📲 [PUSH] Notificación exitosa al dispositivo del usuario {user_id}")
                success_count += 1
            except WebPushException as ex:
                logger.error(f"❌ [PUSH] Error enviando al usuario {user_id}: {repr(ex)}")
                if ex.response is not None and ex.response.status_code in [404, 410]:
                    # La suscripción expiró o el usuario revocó permisos. Limpiarla de la base de datos.
                    endpoint = sub_info.get("endpoint")
                    if endpoint:
                        execute_sql_write(
                            "DELETE FROM push_subscriptions WHERE user_id = %s AND subscription_data->>'endpoint' = %s",
                            (user_id, endpoint)
                        )
                        logger.info(f"🗑️ [PUSH] Suscripción muerta eliminada para {user_id}")

        return success_count > 0

    except Exception as e:
        logger.error(f"❌ [PUSH] Excepción general despachando Push notification a {user_id}: {e}")
        return False
