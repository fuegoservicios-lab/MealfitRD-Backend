import asyncio
import logging
import os
import time
from typing import Optional
import jwt  # PyJWT — ya dep (neon_auth lo usa para EdDSA); aquí HS256 para la cookie.
from fastapi import Header, Cookie, Depends, HTTPException, Response
from db import (
    ensure_user_profile_exists,
    get_monthly_api_usage,
    get_user_profile,
)
from neon_auth import verify_neon_jwt  # [P1-NEON-AUTH-MIGRATION · 2026-06-13]
from knobs import _env_int, is_production  # [P3-TIER-LIMITS-ENV · 2026-05-20] auto-registry; is_production SSOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# [P1-FIRST-PARTY-SESSION · 2026-06-16] Cookie de sesión FIRST-PARTY emitida por
# ESTE backend en mealfitrd.com.
#
# Por qué existe: Neon Auth sirve la sesión como cookie de su PROPIO dominio
# (`*.neonauth.aws.neon.tech`) → cookie de TERCERO relativa a mealfitrd.com. iOS
# Safari/PWA (ITP) la borra al cerrar la app → el usuario aparece deslogueado al
# reabrir. El SDK de Neon NO persiste nada en localStorage (verificado). Neon NO
# ofrece custom-domain para el auth (su roadmap lo lista "coming soon"). Solución
# soportada para un stack sin runtime Node: que NUESTRO backend emita la cookie.
#
# Diseño (capa ENCIMA de Neon, NO la reemplaza):
#   - El usuario se loguea en Neon (email u OAuth — da igual el método).
#   - El frontend presenta el JWT EdDSA de Neon (ya verificado por verify_neon_jwt)
#     a POST /api/auth/session → emitimos `__Host-mf_session` (JWT HS256 firmado
#     con MEALFIT_SESSION_SECRET, propio del backend).
#   - get_verified_user_id la acepta como FALLBACK del Bearer.
#
# Seguridad (preserva P0-AUDIT-1): el path de la cookie verifica SIEMPRE la firma
# HS256 con algoritmo FIJO (sin confusión) y `typ` esperado; fail-secure → None
# ante cualquier fallo. NUNCA retorna un `sub` no verificado. Gated en el secreto:
# sin MEALFIT_SESSION_SECRET (≥32 chars) la cookie ni se emite ni se acepta →
# degradación a Bearer-only (comportamiento actual, cero regresión).
# ---------------------------------------------------------------------------
SESSION_COOKIE_NAME = "__Host-mf_session"
_SESSION_SECRET = (os.environ.get("MEALFIT_SESSION_SECRET") or "").strip()
# Secreto ANTERIOR (opcional): durante una rotación, las cookies firmadas con el
# secreto viejo siguen aceptándose un período de gracia → rotar sin desloguear a
# todos. Vaciar tras el período. Ver runbook de rotación.
_SESSION_SECRET_OLD = (os.environ.get("MEALFIT_SESSION_SECRET_OLD") or "").strip()
_SESSION_TTL_S = _env_int("MEALFIT_SESSION_TTL_DAYS", 30) * 86400
_SESSION_ABS_MAX_S = _env_int("MEALFIT_SESSION_ABS_MAX_DAYS", 90) * 86400
_SESSION_SKEW_S = 300  # tolerancia de clock skew para rechazar iat "futuros".
_SESSION_ALG = "HS256"
_SESSION_TYP = "mf_session"


def session_cookies_enabled() -> bool:
    """True sólo si hay un secreto FUERTE (≥32 chars). Sin él, fail-secure: la
    cookie first-party ni se emite ni se acepta (solo Bearer)."""
    return len(_SESSION_SECRET) >= 32


# [P2-SESSION-SECRET-FAIL-LOUD · 2026-06-19] (audit fresco P2-11/P2-14) En PRODUCCIÓN, si MEALFIT_SESSION_SECRET
# falta o mide <32 chars, la cookie first-party se DESHABILITA en silencio (degrada a Bearer-only) → los usuarios
# iOS-PWA pierden la persistencia de sesión (aparecen deslogueados al reabrir) sin ninguna señal en logs. Fail-loud
# al import (asimetría con neon_auth.py, que SÍ avisa cuando su config es inválida en prod) para que el operador lo
# note. NO debilita la auth (P0-AUDIT-1 intacto: sigue Bearer-only verificando firma); solo lo hace visible.
try:
    if is_production() and not session_cookies_enabled():
        logger.error(
            "❌ [P2-SESSION-SECRET-FAIL-LOUD] MEALFIT_SESSION_SECRET ausente o <32 chars en PRODUCCIÓN → "
            "cookie first-party DESHABILITADA; los usuarios iOS-PWA perderán la sesión al reabrir la app. "
            "Setea un secreto fuerte (≥32 chars) en el .env del VPS."
        )
except Exception:  # pragma: no cover - el check de import jamás debe tumbar el arranque
    pass


def mint_session_cookie(uid: str, iat: Optional[int] = None) -> Optional[str]:
    """JWT HS256 `__Host-mf_session`. `iat` se preserva en el re-issue deslizante
    para aplicar el cap absoluto. None si está deshabilitado o sin uid."""
    if not session_cookies_enabled() or not uid:
        return None
    now = int(time.time())
    payload = {"sub": uid, "typ": _SESSION_TYP, "iat": iat or now, "exp": now + _SESSION_TTL_S}
    return jwt.encode(payload, _SESSION_SECRET, algorithm=_SESSION_ALG)


def _decode_session_cookie(value: str) -> Optional[dict]:
    """Verifica firma + exp + typ + CAP ABSOLUTO. Fail-secure → None; NUNCA lanza.

    - Algoritmo FIJO HS256 (sin confusión / sin `none`).
    - `leeway=0` y `verify_exp` EXPLÍCITOS: la cookie vive 30d, así que ni un
      segundo de gracia sobre el exp es aceptable (riesgo de robo > clock skew).
    - Rotación: prueba el secreto ACTUAL y, si la firma falla, el ANTERIOR
      (período de gracia) → rotar sin desloguear a todos.
    - Cap absoluto: rechaza si el `iat` original es ausente, FUTURO (skew / forja
      tras compromiso del secreto) o más viejo que `_SESSION_ABS_MAX_S` — así el
      cap acota la sesión EN EL PATH DE AUTH (no solo en el re-issue de /me).
    """
    if not session_cookies_enabled() or not value:
        return None
    secrets = [_SESSION_SECRET]
    if len(_SESSION_SECRET_OLD) >= 32:
        secrets.append(_SESSION_SECRET_OLD)
    payload = None
    for secret in secrets:
        try:
            payload = jwt.decode(
                value, secret, algorithms=[_SESSION_ALG],
                leeway=0, options={"verify_exp": True},
            )
            break  # firma válida con este secreto
        except jwt.InvalidSignatureError:
            continue  # rotación: probar el secreto anterior
        except Exception:
            return None  # expirado / malformado / etc → fail-secure
    if payload is None:
        return None
    if payload.get("typ") != _SESSION_TYP or not payload.get("sub"):
        return None
    iat = int(payload.get("iat") or 0)
    now = int(time.time())
    if iat <= 0 or iat > now + _SESSION_SKEW_S or (now - iat) >= _SESSION_ABS_MAX_S:
        return None
    return payload


def verify_session_cookie(value: str) -> Optional[str]:
    """user_id (sub) si la cookie es válida, o None (fail-secure)."""
    payload = _decode_session_cookie(value)
    return payload.get("sub") if payload else None


def session_cookie_within_absolute_cap(value: str) -> bool:
    """True si la sesión (por su `iat` original) aún está dentro del cap absoluto.
    Acota la vida de una cookie robada pese al refresh deslizante."""
    payload = _decode_session_cookie(value)
    if not payload:
        return False
    iat = int(payload.get("iat") or 0)
    return iat > 0 and (int(time.time()) - iat) < _SESSION_ABS_MAX_S


def session_cookie_iat(value: str) -> Optional[int]:
    """`iat` original verificado (para preservarlo en el re-issue). None si inválida."""
    payload = _decode_session_cookie(value)
    if not payload:
        return None
    iat = int(payload.get("iat") or 0)
    return iat or None


# [P3-TIER-LIMITS-ENV · 2026-05-20] Tier limits via knobs (no hardcoded).
# Pre-fix: `{"gratis": 15, "basic": 50, "plus": 200, "ultra": 999999, "admin": 999999}`
# era dict literal dentro de `verify_api_quota`. Cualquier ajuste de pricing
# (e.g. "subir basic de 50 a 75 esta semana para retención") requería:
#   1. PR + review.
#   2. Merge a main + redeploy EasyPanel.
#   3. Esperar nixpacks build (~3-5min).
#   4. Validar via /health/version que el deploy lag detector confirma.
# Total: ~30 min para cambiar UN número. Con env vars: SRE/founder
# setea `MEALFIT_TIER_LIMIT_BASIC=75` en EasyPanel + reinicia worker = <1 min.
#
# Defaults preservan pricing actual. Auto-registry en `_KNOBS_REGISTRY`
# → visible en `/health/version` para audit del valor activo en cada
# entorno. Audit-anchor: P3-TIER-LIMITS-ENV.
#
# Por qué module-level (no per-request lookup):
#   `_env_int` se llama una vez en import time. Para cambiar el valor sin
#   redeploy, basta con bumpear la env var + restart del worker (EasyPanel
#   uvicorn). El cost de re-leer env var por cada request sería marginal
#   pero innecesario — los tiers no cambian intra-request.
_TIER_LIMITS = {
    "gratis": _env_int("MEALFIT_TIER_LIMIT_GRATIS", 15),
    "basic": _env_int("MEALFIT_TIER_LIMIT_BASIC", 50),
    "plus": _env_int("MEALFIT_TIER_LIMIT_PLUS", 200),
    "ultra": _env_int("MEALFIT_TIER_LIMIT_ULTRA", 999999),
    "admin": _env_int("MEALFIT_TIER_LIMIT_ADMIN", 999999),
}


async def get_verified_user_id(
    authorization: Optional[str] = Header(None),
    mf_session: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
    x_mf_session: Optional[str] = Header(None, alias="X-MF-Session"),
) -> Optional[str]:
    """Verifica el JWT de Neon Auth y retorna `sub` (user_id) si la firma es válida.

    [P1-NEON-AUTH-MIGRATION · 2026-06-13] Reemplaza la verificación que antes
    hacía el proveedor de Auth legacy. Neon Auth (Better Auth) emite JWTs
    EdDSA (Ed25519) validados LOCALMENTE contra el JWKS público cacheado
    (`neon_auth.verify_neon_jwt`) — sin roundtrip de red por request en estado
    caliente (mejora vs. el proveedor anterior, que llamaba a su API cada vez). El `sub` del
    payload es el user_id (UUID), misma clave que `public.user_profiles.id`.

    [P0-AUDIT-1 · 2026-05-12] La invariante de seguridad se preserva intacta:
    NUNCA se decodifica el payload sin verificar la firma. `verify_neon_jwt`
    valida firma + `iss` + `aud` + `exp` con el algoritmo FIJO `["EdDSA"]`
    (sin algorithm-confusion ni `none`) antes de retornar claims. El backend
    conecta a Neon con un rol que bypassa RLS, así que esta función sigue
    siendo la ÚNICA capa de auth — el fail-secure es obligatorio: cualquier
    fallo (firma inválida, expirado, kid desconocido, JWKS inalcanzable) →
    `None`, jamás un claim no verificado.

    [P2-AUTH-ASYNC-SLEEP · 2026-05-12] `async def` + `asyncio.to_thread` para la
    verificación: en estado caliente es CPU-only (firma Ed25519) y retorna al
    instante; el único I/O es el fetch ocasional del JWKS (primer request /
    rotación de kid), que `to_thread` mantiene fuera del event loop.

    Contrato (callers ya lo asumen — preservado):
      * `Authorization` ausente / no `Bearer …`         → None.
      * Neon Auth no configurado / token vacío          → None.
      * Token cuya firma o claims son inválidos          → None.
      * Token válido sin `sub`                           → None.
      * Token válido                                     → sub (user_id).

    Tooltip-anchor: P0-AUDIT-1-AUTH-VERIFY | P1-NEON-AUTH-VERIFY | bypass cerrado 2026-05-12.
    Tests parser-based: `tests/test_p0_audit_1_auth_bypass.py`, `tests/test_p2_prod_audit_3.py`.
    """
    # 1) Bearer JWT de Neon Auth (path PRIMARIO — comportamiento sin cambios).
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        if token:
            # [P1-NEON-AUTH-VERIFY] Verificación local contra el JWKS de Neon Auth.
            # `verify_neon_jwt` JAMÁS lanza — retorna el payload validado o None.
            # `to_thread` despacha el fetch ocasional del JWKS (red) fuera del loop.
            payload = await asyncio.to_thread(verify_neon_jwt, token)
            if payload:
                uid = payload.get("sub")
                if uid:
                    # [P1-NEON-DB-MIGRATION · 2026-06-12] Garantiza la fila espejo
                    # en public.user_profiles (reemplaza el trigger handle_new_user).
                    # Cacheado in-process: tras el primer request es no-op. Best-
                    # effort: el JWT ya validó — no bloquear auth si el INSERT falla.
                    try:
                        await asyncio.to_thread(
                            ensure_user_profile_exists,
                            uid,
                            payload.get("email"),
                            payload.get("name"),
                        )
                    except Exception as ensure_err:
                        logger.warning(
                            f"[P1-NEON-DB-MIGRATION] ensure_user_profile_exists lanzó "
                            f"{type(ensure_err).__name__} (auth continúa)"
                        )
                    return uid
                # Token válido sin `sub` → no identificable. Fail-secure: sigue al
                # fallback de cookie (probablemente también ausente → None).

    # 2) [P1-FIRST-PARTY-SESSION] Fallback: cookie first-party `__Host-mf_session`.
    #    Firma HS256 verificada con algoritmo FIJO (sin confusión), `typ`
    #    esperado, fail-secure → None ante cualquier fallo. PRESERVA P0-AUDIT-1:
    #    jamás retorna un `sub` no verificado. El profile ya existe (se creó en el
    #    login vía el Bearer que originó la cookie). Sin secreto → no-op (Bearer-only).
    if mf_session:
        uid = verify_session_cookie(mf_session)
        if uid:
            return uid

    # 3) [P1-FIRST-PARTY-SESSION] Header `X-MF-Session` = el MISMO token de sesión
    #    pero guardado en localStorage (no en cookie). Necesario porque los PWA
    #    standalone de iOS NO persisten cookies de forma confiable entre
    #    lanzamientos, pero localStorage SÍ. El header solo lo añade JS (el
    #    browser no lo manda solo) → inmune a CSRF. Misma verificación HS256
    #    (firma + typ + cap + exp), fail-secure → None.
    if x_mf_session:
        uid = verify_session_cookie(x_mf_session)
        if uid:
            return uid

    return None


async def get_neon_bearer_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """[P1-FIRST-PARTY-SESSION] Verificador Bearer-only (SIN fallback de cookie).
    Para `POST /api/auth/session`: emitir/renovar la cookie first-party con `iat`
    FRESCO requiere un login Neon REAL (Bearer), no la cookie misma — así el cap
    absoluto del re-issue deslizante no se puede resetear teniendo solo la cookie.
    Misma verificación de firma EdDSA que el path Bearer de get_verified_user_id."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return None
    payload = await asyncio.to_thread(verify_neon_jwt, token)
    if not payload:
        return None
    uid = payload.get("sub")
    if not uid:
        return None
    try:
        await asyncio.to_thread(
            ensure_user_profile_exists, uid, payload.get("email"), payload.get("name")
        )
    except Exception as ensure_err:
        logger.warning(
            f"[P1-FIRST-PARTY-SESSION] ensure_user_profile_exists lanzó "
            f"{type(ensure_err).__name__} (auth continúa)"
        )
    return uid


def set_session_cookie(response: Response, uid: str, iat: Optional[int] = None) -> Optional[str]:
    """Mintea + setea `__Host-mf_session` y DEVUELVE el token (o None). El prefijo
    `__Host-` fuerza host-only (first-party), Secure y Path=/ sin Domain.
    `SameSite=Strict`: la cookie SOLO se envía en requests same-site a
    mealfitrd.com → cierra CSRF (un POST cross-site no la lleva) sin downside.
    El token devuelto también se manda en el body para que el cliente lo guarde
    en localStorage (fallback iOS-PWA donde la cookie no persiste)."""
    token = mint_session_cookie(uid, iat=iat)
    if not token:
        return None
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=_SESSION_TTL_S,
        path="/",
        secure=True,
        httponly=True,
        samesite="strict",
    )
    return token


def clear_session_cookie(response: Response) -> None:
    """Borra `__Host-mf_session` (mismos atributos para que el browser la matchee)."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME, path="/", secure=True, httponly=True, samesite="strict"
    )


def verify_api_quota(verified_user_id: Optional[str] = Depends(get_verified_user_id)) -> Optional[str]:
    """Dependencia para verificar los límites de uso de la API (Paywall) evitando repetición (DRY)."""
    if verified_user_id:
        credits_used = get_monthly_api_usage(verified_user_id)
        plan_tier = "gratis"

        profile = get_user_profile(verified_user_id)
        if profile:
            plan_tier = profile.get("plan_tier", "gratis")

        # [P3-TIER-LIMITS-ENV · 2026-05-20] Dict literal reemplazado por
        # `_TIER_LIMITS` module-level (knobs auto-registrados). Default
        # gratis=15 cuando el tier es desconocido (defensive — usuario con
        # `plan_tier` corrupto NO debe quedar con quota ilimitada).
        limit = _TIER_LIMITS.get(plan_tier, _TIER_LIMITS["gratis"])

        if credits_used >= limit:
            raise HTTPException(status_code=402, detail=f"Límite de créditos alcanzado para tu plan {plan_tier} ({limit}/{limit}). Mejora tu plan para continuar.")

    return verified_user_id
