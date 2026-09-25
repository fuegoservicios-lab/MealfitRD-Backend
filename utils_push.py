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

def send_push_notification(user_id: str, title: str, body: str, url: str = "/dashboard", tag: str = None,
                           solo_si_no_mira: bool = False, nativa: bool = True) -> bool:
    """
    Sends a push notification to every device of the user: Web Push (browser/PWA) and, [P1-PLAN-LOTE-280], FCM for
    the native Android app (`fcm_push.py`). Returns True if at least one was accepted. Nunca lanza.
    """
    web = _enviar_web_push(user_id, title, body, url=url, tag=tag, solo_si_no_mira=solo_si_no_mira)
    # [P1-PLAN-LOTE-280] `nativa=False`: los recordatorios de comida y agua. En la app nativa ya salen como avisos
    # LOCALES programados por el teléfono (utils/avisosDeComida.js); por FCM llegarían dos veces.
    enviados_nativa = 0
    try:
        from fcm_push import fcm_configurado, enviar_a_dispositivos
        from apns_push import apns_configurado   # [P1-PLAN-LOTE-300] iOS va directo a Apple
        if nativa and (fcm_configurado() or apns_configurado()):
            titulo, cuerpo = _traducidos(user_id, title, body)
            enviados_nativa = enviar_a_dispositivos(user_id, titulo, cuerpo, url=url, tag=tag,
                                                    solo_si_no_mira=solo_si_no_mira)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-280] push nativa no despachada a {user_id}: {e!r}")
    return bool(web) or enviados_nativa > 0


def _traducidos(user_id: str, title: str, body: str) -> tuple:
    """[P1-PLAN-LOTE-280] Título y cuerpo en el idioma del usuario para la push nativa (misma regla que la web)."""
    try:
        _perfil = execute_sql_query("SELECT locale FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
        if isinstance(_perfil, (list, tuple)):
            _perfil = _perfil[0] if _perfil else None
        _locale = _perfil.get("locale") if _perfil and hasattr(_perfil, "get") else None
        from push_i18n import translate_push_text
        return translate_push_text(title, _locale), translate_push_text(body, _locale)
    except Exception:  # noqa: BLE001
        return title, body


def _enviar_web_push(user_id: str, title: str, body: str, url: str = "/dashboard", tag: str = None,
                     solo_si_no_mira: bool = False) -> bool:
    """Web Push (VAPID) a las suscripciones del navegador/PWA. True si al menos una se envió."""
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

        _payload = {
            "title": title,
            "body": body,
            "url": url
        }
        # [P1-PLAN-LOTE-133 · 2026-09-20] Con `tag`, la notificación nueva SUSTITUYE a la anterior de la misma etiqueta
        # en vez de apilarse (cuatro recordatorios de comida = cuatro notificaciones pegajosas que cerrar a mano).
        if tag:
            _payload["tag"] = str(tag)[:64]
        # [P1-PLAN-LOTE-228 · 2026-09-25] El service worker no la muestra si hay una ventana de la app VISIBLE (el
        # usuario ya lo está viendo en pantalla). Solo para avisos que la app también da dentro (el plan listo).
        if solo_si_no_mira:
            _payload["solo_si_no_mira"] = True
        push_payload = json.dumps(_payload)

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
