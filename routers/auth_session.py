"""[P1-FIRST-PARTY-SESSION · 2026-06-16] Endpoints de la cookie de sesión
first-party `__Host-mf_session`.

Por qué (resumen — diseño/seguridad completos en `auth.py`): Neon Auth sirve la
sesión como cookie de su propio dominio (third-party relativa a mealfitrd.com);
iOS Safari/PWA la borra al cerrar la app → logout. Esta capa emite, EN ESTE
backend (mismo origen), una cookie first-party que iOS sí conserva. No reemplaza
a Neon: el usuario se loguea en Neon (email u OAuth) y luego intercambia su JWT
verificado por esta cookie.

  POST /api/auth/session  — Bearer Neon válido → emite/renueva la cookie (iat fresco).
  GET  /api/auth/me       — cookie/Bearer → {user_id} + re-issue deslizante (cap absoluto).
  POST /api/auth/logout   — borra la cookie.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Response, Cookie, Header

from auth import (
    get_verified_user_id,
    get_neon_bearer_user_id,
    set_session_cookie,
    clear_session_cookie,
    session_cookie_iat,
    session_cookies_enabled,
    derive_form_key,
    SESSION_COOKIE_NAME,
)
from rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth-session"])

# [P2-RATELIMIT-COVERAGE · 2026-06-18] Throttle de los endpoints de sesión que
# ejecutan verify_neon_jwt (firma Ed25519 en threadpool). Sin throttle, martillear
# POST /session o GET /me con Bearers basura satura el threadpool de verificación.
# 30/60s por usuario (o por IP para anónimos) es generoso para el uso legítimo
# (emisión/re-issue deslizante de la cookie al loguearse o reabrir la app).
_AUTH_SESSION_LIMITER = RateLimiter(max_calls=30, period_seconds=60)


@router.post("/session")
async def create_session(
    response: Response,
    _rl: object = Depends(_AUTH_SESSION_LIMITER),
    uid: Optional[str] = Depends(get_neon_bearer_user_id),
):
    """Tras un login Neon (email/OAuth) el frontend presenta su Bearer EdDSA y
    emitimos la cookie first-party con `iat` fresco. 401 si no hay Bearer válido.
    Si la feature está apagada (sin secreto), responde 200 `session_cookie:false`
    (degradación silenciosa a Bearer-only — el frontend sigue funcionando)."""
    if not session_cookies_enabled():
        return {"ok": True, "session_cookie": False, "reason": "disabled"}
    if not uid:
        return Response(status_code=401)
    token = set_session_cookie(response, uid)
    if not token:
        return {"ok": False, "session_cookie": False}
    # `token` también va en el body → el cliente lo guarda en localStorage como
    # fallback iOS-PWA (donde la cookie no persiste entre lanzamientos).
    # [P1-FORM-KEY · 2026-06-21] `form_key` = llave estable por usuario para cifrar
    # el formulario sensible en el navegador (sobrevive re-logins / rotación de token).
    return {"ok": True, "session_cookie": True, "user_id": uid, "token": token, "form_key": derive_form_key(uid)}


@router.get("/me")
async def session_me(
    response: Response,
    _rl: object = Depends(_AUTH_SESSION_LIMITER),
    uid: Optional[str] = Depends(get_verified_user_id),
    mf_session: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
    x_mf_session: Optional[str] = Header(None, alias="X-MF-Session"),
):
    """Estado de sesión first-party (usado al reabrir la app cuando la sesión de
    Neon ya no está). Re-emite el token deslizante (preservando el `iat`) y lo
    devuelve en el body para que el cliente actualice su localStorage. 401 si no
    hay sesión válida (ni Bearer ni cookie ni header)."""
    if not uid:
        # Limpia una cookie stale (p.ej. tras rotar el secreto) para cortar el
        # loop de 401 con una cookie no verificable que el browser retendría.
        resp = Response(status_code=401)
        clear_session_cookie(resp)
        return resp
    # Re-issue deslizante: preserva el `iat` original (del cookie O del header —
    # el PWA iOS usa el header). `session_cookie_iat` aplica el cap absoluto.
    src = mf_session or x_mf_session
    new_token = None
    if src:
        iat = session_cookie_iat(src)
        if iat:
            new_token = set_session_cookie(response, uid, iat=iat)
    # [P1-FORM-KEY · 2026-06-21] Llave estable para cifrar el form sensible (ver
    # /session). Se entrega en /me para que el cliente la tenga al reabrir la app
    # vía sesión first-party (cuando ya no hay sesión de Neon ni access_token).
    return {"user_id": uid, "token": new_token, "form_key": derive_form_key(uid)}


@router.post("/logout")
async def session_logout(response: Response):
    """Borra la cookie first-party. Sin auth a propósito: borrar la propia sesión
    siempre es seguro y debe funcionar aunque el token ya no sea válido."""
    # [RATELIMIT-EXEMPT: borrar la propia cookie es idempotente y siempre seguro,
    # sin auth ni cómputo costoso — un flood solo re-emite el header Set-Cookie de
    # borrado. Debe funcionar aunque el token ya no sea válido (P2-RATELIMIT-COVERAGE).]
    clear_session_cookie(response)
    return {"ok": True}
