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
  POST /api/auth/email-otp/verify — [P1-OTP-FIRST-PARTY · 2026-07-03] verifica el código
      OTP contra Neon Auth SERVER-SIDE y emite la sesión first-party directo.
"""
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Body, Depends, Response, Cookie, Header

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


# [P1-OTP-FIRST-PARTY · 2026-07-03] Throttle propio, más estricto que el de sesión:
# este endpoint proxya un intento de login (email+código) — 10/60s corta fuerza bruta
# de códigos sin estorbar el uso legítimo (1 verify por login).
_OTP_VERIFY_LIMITER = RateLimiter(max_calls=10, period_seconds=60)


@router.post("/email-otp/verify")
async def email_otp_verify(
    response: Response,
    data: dict = Body(...),
    _rl: object = Depends(_OTP_VERIFY_LIMITER),
):
    """[P1-OTP-FIRST-PARTY · 2026-07-03] Verifica el código OTP contra Neon Auth
    SERVER-SIDE y, si es válido, emite la sesión first-party (`__Host-mf_session`
    + token en body) SIN depender de la cookie cross-origin de Neon.

    Por qué: el flujo previo verificaba el OTP con un fetch directo del navegador a
    Neon Auth (`/sign-in/email-otp`, credentials:include). La cookie de sesión que
    Neon setea en esa respuesta es THIRD-PARTY (dominio neonauth ≠ app.mealfitrd.com)
    y viene de un XHR, no de una navegación top-level → los navegadores móviles la
    BLOQUEAN (iOS siempre; Chrome con el phase-out de third-party cookies). Tras el
    reload, getSession() no encontraba nada → rebote a /login ("pongo el código y no
    entro"). Google OAuth sí funciona porque su cookie nace en el redirect top-level.

    Este proxy verifica el código servidor-a-servidor (sin browser de por medio) y
    emite NUESTRA sesión first-party — el mismo mecanismo que ya sostiene el PWA iOS
    (P1-FIRST-PARTY-SESSION): el provider la resuelve vía _resolveViaFirstParty y
    todas las fetches autenticadas funcionan por cookie/X-MF-Session.

    Fail-secure: cualquier respuesta no-OK de Neon, o sin user.id → 401 sin cookie.
    tooltip-anchor: P1-OTP-FIRST-PARTY"""
    from neon_auth import NEON_AUTH_BASE_URL

    email = str((data or {}).get("email") or "").strip()
    otp = str((data or {}).get("otp") or "").strip()
    if not email or "@" not in email or not otp or not (4 <= len(otp) <= 12):
        return Response(status_code=401)
    if not NEON_AUTH_BASE_URL:
        logger.error("[P1-OTP-FIRST-PARTY] NEON_AUTH_BASE_URL ausente — no se puede verificar OTP.")
        return Response(status_code=503)
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.post(
                f"{NEON_AUTH_BASE_URL}/sign-in/email-otp",
                json={"email": email, "otp": otp},
            )
    except Exception as e:
        logger.warning(f"[P1-OTP-FIRST-PARTY] Neon Auth inalcanzable verificando OTP: {type(e).__name__}: {e}")
        return Response(status_code=502)
    if r.status_code != 200:
        # Código inválido/expirado (o email desconocido) — mismo mensaje neutro.
        logger.info(f"[P1-OTP-FIRST-PARTY] verify rechazado por Neon (HTTP {r.status_code}).")
        return Response(status_code=401)
    try:
        payload = r.json() or {}
    except Exception:
        payload = {}
    user = payload.get("user") or {}
    uid = str(user.get("id") or "").strip()
    if not uid:
        logger.warning("[P1-OTP-FIRST-PARTY] Neon respondió 200 sin user.id — fail-secure 401.")
        return Response(status_code=401)
    if not session_cookies_enabled():
        # Sin secreto de sesión first-party no hay forma de sostener este login
        # server-side; mejor error explícito que un 200 que no loguea.
        logger.error("[P1-OTP-FIRST-PARTY] session_cookies deshabilitadas — el login OTP requiere la feature.")
        return Response(status_code=503)
    token = set_session_cookie(response, uid)
    if not token:
        return Response(status_code=503)
    logger.info(f"🔐 [P1-OTP-FIRST-PARTY] OTP verificado server-side → sesión first-party emitida (uid={uid[:8]}…).")
    return {
        "ok": True,
        "user_id": uid,
        "email": user.get("email") or email,
        "token": token,
        "form_key": derive_form_key(uid),
        "session_cookie": True,
    }


# [P1-OAUTH-FIRST-PARTY · 2026-07-03] Mismo throttle-perfil que el OTP verify: canjear un
# verifier es un intento de login.
_OAUTH_ADOPT_LIMITER = RateLimiter(max_calls=10, period_seconds=60)


@router.post("/oauth/adopt")
async def oauth_adopt(
    response: Response,
    data: dict = Body(...),
    _rl: object = Depends(_OAUTH_ADOPT_LIMITER),
):
    """[P1-OAUTH-FIRST-PARTY · 2026-07-03] Canjea el `neon_auth_session_verifier` (query param
    con el que Neon Auth regresa del OAuth de Google) por la sesión first-party, SERVER-SIDE.

    Por qué: el SDK canjea ese verifier client-side pegándole a `<neon>/get-session` — pero el
    verifier es DE UN SOLO USO. En móvil, si esa primera petición se pierde (timeout de red /
    getSessionWithTimeout corto), el verifier queda consumido sin sesión y NINGÚN retry puede
    resolverla (y sin cookie de Neon utilizable — third-party) → el usuario debía pulsar
    "Continuar con Google" una SEGUNDA vez para obtener un verifier fresco. Canjeándolo aquí
    (server-a-server, timeout generoso, sin third-party cookies) y emitiendo `__Host-mf_session`,
    el primer click SIEMPRE termina logueado — mismo patrón de P1-OTP-FIRST-PARTY.

    Fail-secure: verifier ausente/basura, Neon non-200 o sin user.id → 401 sin cookie.
    tooltip-anchor: P1-OAUTH-FIRST-PARTY"""
    from neon_auth import NEON_AUTH_BASE_URL

    verifier = str((data or {}).get("verifier") or "").strip()
    if not verifier or len(verifier) > 512:
        return Response(status_code=401)
    if not NEON_AUTH_BASE_URL:
        logger.error("[P1-OAUTH-FIRST-PARTY] NEON_AUTH_BASE_URL ausente — no se puede canjear el verifier.")
        return Response(status_code=503)
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(
                f"{NEON_AUTH_BASE_URL}/get-session",
                params={"neon_auth_session_verifier": verifier},
            )
    except Exception as e:
        logger.warning(f"[P1-OAUTH-FIRST-PARTY] Neon Auth inalcanzable canjeando verifier: {type(e).__name__}: {e}")
        return Response(status_code=502)
    if r.status_code != 200:
        logger.info(f"[P1-OAUTH-FIRST-PARTY] canje rechazado por Neon (HTTP {r.status_code}).")
        return Response(status_code=401)
    try:
        payload = r.json() or {}
    except Exception:
        payload = {}
    user = payload.get("user") or {}
    uid = str(user.get("id") or "").strip()
    if not uid:
        logger.warning("[P1-OAUTH-FIRST-PARTY] Neon respondió 200 sin user.id — fail-secure 401.")
        return Response(status_code=401)
    if not session_cookies_enabled():
        logger.error("[P1-OAUTH-FIRST-PARTY] session_cookies deshabilitadas — el adopt requiere la feature.")
        return Response(status_code=503)
    token = set_session_cookie(response, uid)
    if not token:
        return Response(status_code=503)
    logger.info(f"🔐 [P1-OAUTH-FIRST-PARTY] verifier canjeado server-side → sesión first-party (uid={uid[:8]}…).")
    return {
        "ok": True,
        "user_id": uid,
        "email": user.get("email") or None,
        "token": token,
        "form_key": derive_form_key(uid),
        "session_cookie": True,
    }


@router.post("/logout")
async def session_logout(response: Response):
    """Borra la cookie first-party. Sin auth a propósito: borrar la propia sesión
    siempre es seguro y debe funcionar aunque el token ya no sea válido."""
    # [RATELIMIT-EXEMPT: borrar la propia cookie es idempotente y siempre seguro,
    # sin auth ni cómputo costoso — un flood solo re-emite el header Set-Cookie de
    # borrado. Debe funcionar aunque el token ya no sea válido (P2-RATELIMIT-COVERAGE).]
    clear_session_cookie(response)
    return {"ok": True}
