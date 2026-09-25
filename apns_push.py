# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-300 · 2026-09-25] Push NATIVA en iOS: directo a Apple (APNs), sin Firebase.

`fcm_push.enviar_a_dispositivos` reparte por plataforma: los tokens `ios` (token APNs que entrega
`@capacitor/push-notifications`) vienen aquí. HTTP/2 (`httpx` + `h2`) contra `api.push.apple.com` con un JWT ES256
firmado con la clave .p8 del dueño (`APNS_KEY_FILE`, fuera del repo) — Apple lo acepta hasta 1 h; se renueva a los
50 min. Token muerto (410, o 400 BadDeviceToken) ⇒ se borra la fila. tooltip-anchor: P1-PLAN-LOTE-300-APNS
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_HOST = "https://api.push.apple.com"
_RENOVAR_S = 50 * 60
_lock = threading.Lock()
_cache = {"jwt": None, "desde": 0.0, "cliente": None}


def _conf():
    ruta = (os.environ.get("APNS_KEY_FILE") or "").strip()
    return {
        "ruta": ruta if ruta and os.path.isfile(ruta) else None,
        "kid": (os.environ.get("APNS_KEY_ID") or "").strip(),
        "team": (os.environ.get("APNS_TEAM_ID") or "").strip(),
        "topic": (os.environ.get("APNS_TOPIC") or "com.bioboros.app").strip(),
    }


def apns_configurado() -> bool:
    c = _conf()
    return bool(c["ruta"] and c["kid"] and c["team"])


def _jwt() -> str:
    with _lock:
        if _cache["jwt"] and time.time() - _cache["desde"] < _RENOVAR_S:
            return _cache["jwt"]
        import jwt
        c = _conf()
        with open(c["ruta"], encoding="utf-8") as fh:
            clave = fh.read()
        ahora = int(time.time())
        _cache["jwt"] = jwt.encode({"iss": c["team"], "iat": ahora}, clave, algorithm="ES256",
                                   headers={"kid": c["kid"]})
        _cache["desde"] = ahora
        return _cache["jwt"]


def construir(token, title, body, url="/dashboard", tag=None, solo_si_no_mira=False, *, jwt_token, topic):
    """(cabeceras, cuerpo JSON) de un aviso. Pura (la prueba la mira sin red)."""
    h = {"authorization": f"bearer {jwt_token}", "apns-topic": topic, "apns-push-type": "alert",
         "apns-priority": "10"}
    if tag:
        h["apns-collapse-id"] = str(tag)[:64]      # el aviso nuevo de la misma etiqueta sustituye al anterior
    cuerpo = {"aps": {"alert": {"title": str(title or ""), "body": str(body or "")}, "sound": "default"},
              "url": str(url or "/dashboard")}
    if tag:
        cuerpo["aps"]["thread-id"] = str(tag)[:64]
        cuerpo["tag"] = str(tag)[:64]
    if solo_si_no_mira:
        cuerpo["solo_si_no_mira"] = "1"
    return h, json.dumps(cuerpo, ensure_ascii=False)


def _cliente():
    with _lock:
        if _cache["cliente"] is None:
            import httpx
            from utils_push import _PUSH_HTTP_TIMEOUT_S
            _cache["cliente"] = httpx.Client(http2=True, timeout=_PUSH_HTTP_TIMEOUT_S)
        return _cache["cliente"]


def _enviar_uno(token, title, body, url, tag, solo_si_no_mira):
    c = _conf()
    h, cuerpo = construir(token, title, body, url, tag, solo_si_no_mira, jwt_token=_jwt(), topic=c["topic"])
    return _cliente().post(f"{_HOST}/3/device/{token}", headers=h, content=cuerpo.encode("utf-8"))


def token_muerto(status: int, texto: str) -> bool:
    return status == 410 or (status == 400 and ("BadDeviceToken" in (texto or "") or "Unregistered" in (texto or "")))
