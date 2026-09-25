# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-280 · 2026-09-25] Push NATIVA con Firebase Cloud Messaging (API HTTP v1).

La app de Android no tenía cómo recibir un aviso del servidor: los ~35 avisos que el backend manda (el plan listo, el
coach proactivo, la Nevera, las pausas del plan) solo llegaban a la web/PWA por Web Push. Con FCM llegan también a la
app, cerrada o no. Los recordatorios de comida y agua siguen siendo LOCALES (los programa el teléfono) y no pasan por
aquí.

Diseño:
  · Sin `firebase-admin`: `google-auth` (ya instalado) firma el token OAuth de la cuenta de servicio y `requests`
    llama a `https://fcm.googleapis.com/v1/projects/<id>/messages:send`. Menos dependencias en el VPS.
  · La clave NO vive en el repo: `FIREBASE_SERVICE_ACCOUNT_FILE` apunta al JSON en el VPS. Sin ella, todo es no-op.
  · Mensaje con `notification` (el sistema lo muestra con la app cerrada) + `data` (`url`, `tag`, `solo_si_no_mira`
    para que la app, si está delante, decida). Canal Android `bioboros-avisos` (el de los recordatorios, prioridad
    alta); `tag` sustituye el aviso anterior de la misma etiqueta en vez de apilarlo.
  · Token muerto (`UNREGISTERED`/`NOT_FOUND`, o `INVALID_ARGUMENT` sobre el token) ⇒ se borra la fila.
tooltip-anchor: P1-PLAN-LOTE-280-FCM
"""
from __future__ import annotations

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

CANAL_ANDROID = "bioboros-avisos"
_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
_lock = threading.Lock()
_cache: dict = {"cred": None, "project": None, "file": None}


def _ruta_clave() -> str | None:
    ruta = (os.environ.get("FIREBASE_SERVICE_ACCOUNT_FILE") or "").strip()
    return ruta if ruta and os.path.isfile(ruta) else None


def fcm_configurado() -> bool:
    return _ruta_clave() is not None


def _credenciales():
    """(credenciales con token vigente, project_id) o (None, None). Cachea y refresca el token de acceso."""
    ruta = _ruta_clave()
    if not ruta:
        return None, None
    with _lock:
        if _cache["cred"] is None or _cache["file"] != ruta:
            from google.oauth2 import service_account
            with open(ruta, encoding="utf-8") as fh:
                info = json.load(fh)
            _cache["cred"] = service_account.Credentials.from_service_account_info(info, scopes=[_SCOPE])
            _cache["project"] = info.get("project_id")
            _cache["file"] = ruta
        cred = _cache["cred"]
        if not cred.valid:
            from google.auth.transport.requests import Request
            cred.refresh(Request())
        return cred, _cache["project"]


def construir_mensaje(token: str, title: str, body: str, url: str = "/dashboard", tag: str | None = None,
                      solo_si_no_mira: bool = False) -> dict:
    """El cuerpo de `messages:send` para un token. Pura (la prueba la mira sin red)."""
    data = {"url": str(url or "/dashboard")}
    if tag:
        data["tag"] = str(tag)[:64]
    if solo_si_no_mira:
        data["solo_si_no_mira"] = "1"
    notif_android = {"channel_id": CANAL_ANDROID, "sound": "default", "default_vibrate_timings": True}
    if tag:
        notif_android["tag"] = str(tag)[:64]
    return {
        "message": {
            "token": token,
            "notification": {"title": str(title or ""), "body": str(body or "")},
            "data": data,
            "android": {"priority": "HIGH", "notification": notif_android},
        }
    }


def _token_muerto(status: int, cuerpo: str) -> bool:
    if status == 404:
        return True
    return status == 400 and ("UNREGISTERED" in cuerpo or "registration token" in cuerpo.lower())


def enviar_a_dispositivos(user_id: str, title: str, body: str, url: str = "/dashboard", tag: str | None = None,
                          solo_si_no_mira: bool = False) -> int:
    """Envía a todos los teléfonos del usuario. Devuelve cuántos aceptó FCM. Nunca lanza."""
    try:
        from db_core import execute_sql_query, execute_sql_write
        filas = execute_sql_query("SELECT token FROM device_push_tokens WHERE user_id = %s", (user_id,),
                                  fetch_all=True) or []
        tokens = [f["token"] for f in filas if isinstance(f, dict) and f.get("token")]
        if not tokens:
            return 0
        cred, proyecto = _credenciales()
        if not cred or not proyecto:
            logger.debug("[P1-PLAN-LOTE-280] FCM sin clave configurada: push nativa no enviada")
            return 0
        import requests
        from utils_push import _PUSH_HTTP_TIMEOUT_S
        endpoint = f"https://fcm.googleapis.com/v1/projects/{proyecto}/messages:send"
        ok = 0
        for token in tokens:
            try:
                r = requests.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {cred.token}", "Content-Type": "application/json; UTF-8"},
                    data=json.dumps(construir_mensaje(token, title, body, url, tag, solo_si_no_mira)),
                    timeout=_PUSH_HTTP_TIMEOUT_S,
                )
                if r.status_code == 200:
                    ok += 1
                elif _token_muerto(r.status_code, r.text or ""):
                    execute_sql_write("DELETE FROM device_push_tokens WHERE token = %s AND user_id = %s",
                                      (token, user_id))
                    logger.info(f"🗑️ [P1-PLAN-LOTE-280] token FCM muerto borrado para {user_id}")
                else:
                    logger.warning(f"[P1-PLAN-LOTE-280] FCM {r.status_code} para {user_id}: {(r.text or '')[:200]}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[P1-PLAN-LOTE-280] envío FCM falló para {user_id}: {e!r}")
        if ok:
            logger.info(f"📲 [P1-PLAN-LOTE-280] push nativa a {ok} dispositivo(s) de {user_id}")
        return ok
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-280] push nativa no enviada a {user_id}: {e!r}")
        return 0


def registrar_token(user_id: str, token: str, platform: str = "android") -> None:
    """UPSERT por token: si el teléfono cambió de cuenta, la fila pasa a la cuenta nueva."""
    from db_core import execute_sql_write
    plataforma = platform if platform in ("android", "ios") else "android"
    execute_sql_write(
        """
        INSERT INTO device_push_tokens (token, user_id, platform, created_at, updated_at)
        VALUES (%s, %s, %s, now(), now())
        ON CONFLICT (token) DO UPDATE SET user_id = EXCLUDED.user_id, platform = EXCLUDED.platform, updated_at = now()
        """,
        (token, user_id, plataforma),
    )


def borrar_token(user_id: str, token: str) -> None:
    from db_core import execute_sql_write
    execute_sql_write("DELETE FROM device_push_tokens WHERE token = %s AND user_id = %s", (token, user_id))

