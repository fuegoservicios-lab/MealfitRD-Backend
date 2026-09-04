import os
import json
import logging
from db_core import execute_sql_query, execute_sql_write
from knobs import _env_float

logger = logging.getLogger(__name__)

# [P1-PUSH-TIMEOUT · 2026-05-28] webpush() -> requests.post() SIN timeout bloquea
# el thread indefinidamente cuando el push-service (FCM/Mozilla autopush) está
# degradado o la ruta de red cuelga. Como send_push_notification se despacha vía
# submit_bg_task (pool BOUNDED compartido con chat title/SSE), N threads colgados
# saturan TODO el pool y degradan chat aunque el problema sea solo push; el
# watcher alerta a 120s pero no puede matar un thread bloqueado en socket. El
# timeout permite que el thread se libere solo. Knob: default 10s, clamp (0, 120].
_PUSH_HTTP_TIMEOUT_S = _env_float(
    "MEALFIT_PUSH_HTTP_TIMEOUT_S", 10.0, validator=lambda v: 0 < v <= 120
)

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
        logger.warning("⚠️ [PUSH] Faltan llaves VAPID en el entorno. No se enviará notificación.")
        return False

    try:
        # Buscar las suscripciones de este usuario en DDBB
        subs_query = "SELECT subscription_data FROM push_subscriptions WHERE user_id = %s"
        subs = execute_sql_query(subs_query, (user_id,), fetch_all=True)

        if not subs:
            logger.debug(f"ℹ️ [PUSH] Usuario {user_id} no tiene suscripciones Push activas.")
            return False

        # [P1-I18N-PUSH-CRON-ESPANOL · 2026-08-22] El idioma se resuelve AQUÍ, que es el
        # cuello de botella por el que pasa TODO push sin excepción
        # (`_dispatch_push_notification` es un envoltorio de esta función).
        #
        # Atarlo al ACTO y no a los 35 call sites es la decisión que importa: un push nuevo
        # queda cubierto sin wiring. Es la lección que este repo ya pagó dos veces —el pop
        # de `_display` colgando de siete funciones con nombre (P2-DISPLAY-POP-VECINO) y
        # «gatear call sites uno a uno es el agujero, no el cierre» (P1-COUNTRY-SYSTEM-F1).
        #
        # `P2-I18N-PUSH-SIN-LOCALE` no se ve afectado: su título ya llega resuelto, aquí no
        # encuentra clave y pasa tal cual.
        #
        # Best-effort de punta a punta: si la consulta del perfil falla, sale en español.
        # Una notificación en español es una degradación; una que no sale es un fallo.
        # [P1-I18N-PUSH-LOCALE-SIEMPRE-NULO · 2026-08-23] `fetch_one=True`, no
        # `fetch_all=False`. NO son lo mismo: `fetch_all=False` cae en la rama por defecto
        # del helper, que hace `fetchall()` y devuelve una LISTA de dicts. Una lista no tiene
        # `.get`, así que el `hasattr` de abajo salía False y `_locale` era None para TODOS
        # los usuarios, siempre — con lo que el catálogo de push (43 mensajes × 4 idiomas)
        # nunca pintó una sola traducción. Medido contra Neon: `[{'locale': 'es-DO'}]`.
        #
        # Se conserva el `hasattr` como red: si el helper vuelve a cambiar de forma, esto
        # degrada al español en vez de reventar. Pero la red ya no es el camino normal, que
        # es lo que la hacía invisible.
        _locale = None
        try:
            _perfil = execute_sql_query(
                "SELECT locale FROM user_profiles WHERE id = %s",
                (user_id,),
                fetch_one=True,
            )
            if isinstance(_perfil, (list, tuple)):
                _perfil = _perfil[0] if _perfil else None
            if _perfil:
                _locale = _perfil.get("locale") if hasattr(_perfil, "get") else None
        except Exception as _loc_err:  # noqa: BLE001
            logger.debug(f"[P1-I18N-PUSH-CRON-ESPANOL] sin locale ({_loc_err!r}); se envía en español")

        try:
            from push_i18n import translate_push_text
            title = translate_push_text(title, _locale)
            body = translate_push_text(body, _locale)
        except Exception as _tr_err:  # noqa: BLE001
            logger.debug(f"[P1-I18N-PUSH-CRON-ESPANOL] traducción no aplicada ({_tr_err!r})")

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
                    vapid_claims={"sub": vapid_claim},
                    timeout=_PUSH_HTTP_TIMEOUT_S,  # [P1-PUSH-TIMEOUT] no bloquear el bg pool
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
