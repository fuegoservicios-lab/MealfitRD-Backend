"""[P1-NEON-AUTH-MIGRATION · 2026-06-13] Verificación de JWTs de Neon Auth.

Reemplaza la verificación de identidad que antes hacía el proveedor de Auth legacy
(P0-AUDIT-1). Neon Auth (Better Auth) emite JWTs firmados con **EdDSA (Ed25519)**
que se validan LOCALMENTE contra el JWKS público, cacheado en proceso — sin
roundtrip de red por request (mejora vs. el proveedor anterior, que llamaba a su API cada vez).

Contrato del JWT (verificado en vivo 2026-06-13):
  - `alg=EdDSA`, `kid` apunta a una clave del JWKS.
  - `sub` = user_id (UUID) — clave de `public.user_profiles.id`.
  - `email`, `name`, `role="authenticated"`, `iss`/`aud` = origin del Auth URL.
  - `exp` ≈ 15 min (el frontend refresca el token; el backend solo valida).

Seguridad (paridad con P0-AUDIT-1):
  - NUNCA se decodifica el payload sin verificar la firma. `jwt.decode` valida
    firma + `iss` + `aud` + `exp` antes de retornar claims.
  - Fail-secure: cualquier fallo (firma inválida, expirado, kid desconocido,
    JWKS inalcanzable) → `None`. Jamás se retorna un claim no verificado.
  - El algoritmo se fija a `["EdDSA"]` — un atacante no puede degradar a `none`
    ni a HMAC (algorithm-confusion) porque la lista de algoritmos es cerrada.

Tooltip-anchor: P1-NEON-AUTH-VERIFY.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import urllib.request
from typing import Optional
from urllib.parse import urlparse

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)

NEON_AUTH_BASE_URL = (os.environ.get("NEON_AUTH_BASE_URL") or "").strip().rstrip("/")
_JWKS_URL = f"{NEON_AUTH_BASE_URL}/.well-known/jwks.json" if NEON_AUTH_BASE_URL else None
_parsed = urlparse(NEON_AUTH_BASE_URL) if NEON_AUTH_BASE_URL else None
# El issuer/audience del token es el ORIGEN del Auth URL (sin el path /neondb/auth).
_ORIGIN = f"{_parsed.scheme}://{_parsed.netloc}" if _parsed and _parsed.netloc else None

# Cache del JWKS. TTL largo (las claves rotan rara vez); un kid-miss fuerza
# refresh inmediato (cubre rotación sin esperar el TTL).
_JWKS_TTL_S = float(os.environ.get("MEALFIT_NEON_AUTH_JWKS_TTL_S") or 3600)
_jwks_cache: dict = {"keys": None, "fetched_at": 0.0, "last_fail_at": 0.0}
_jwks_lock = threading.Lock()
# [P1-NEON-AUTH-JWKS-HARDEN · 2026-06-18] Timeout bajado de 10→4s + negative-cache:
# tras un fallo del fetch, servir el cache previo (stale) por COOLDOWN s sin
# re-intentar. Las claves previas siguen válidas para tokens pre-rotación, así que
# un outage/lentitud del JWKS de Neon Auth NO tumba TODA la autenticación ni
# serializa verificaciones concurrentes detrás de fetches repetidos.
_JWKS_FETCH_TIMEOUT_S = float(os.environ.get("MEALFIT_NEON_AUTH_JWKS_TIMEOUT_S") or 4)
_JWKS_NEG_CACHE_COOLDOWN_S = float(os.environ.get("MEALFIT_NEON_AUTH_JWKS_NEG_COOLDOWN_S") or 30)

# [P1-NEON-AUTH-CONFIG-FAILLOUD · 2026-06-18] Fail-loud al import si en producción
# la config de Neon Auth es inválida: sin NEON_AUTH_BASE_URL o con _ORIGIN no
# parseable, verify_neon_jwt retorna None para TODO Bearer → "nadie entra",
# silenciosamente. Un error en logs lo hace visible al instante en un deploy mal
# configurado, en vez de degradar a "todos los logins fallan".
if os.environ.get("ENVIRONMENT", "").lower() == "production" and (
    not NEON_AUTH_BASE_URL or not _ORIGIN
):
    logger.error(
        "[P1-NEON-AUTH] CONFIG INVÁLIDA en producción: NEON_AUTH_BASE_URL/_ORIGIN "
        "ausente o no parseable → TODA verificación de JWT fallará (nadie puede "
        "autenticarse). Revisar la env var NEON_AUTH_BASE_URL (con scheme https://)."
    )


def is_neon_auth_configured() -> bool:
    """True si `NEON_AUTH_BASE_URL` está seteado Y su origin es parseable
    (auth verificable). Antes solo chequeaba la URL; un valor sin scheme dejaba
    `_ORIGIN=None` y rompía la verificación silenciosamente."""
    return bool(NEON_AUTH_BASE_URL) and bool(_ORIGIN)


def _fetch_jwks(force: bool = False) -> list:
    now = time.monotonic()
    with _jwks_lock:
        cached = _jwks_cache["keys"]
        fresh = cached is not None and (now - _jwks_cache["fetched_at"]) < _JWKS_TTL_S
        if not force and fresh:
            return cached
        # Negative-cache: si un fetch reciente falló, NO re-intentar (no martillear
        # ni serializar) — servir el cache previo mientras dure el cooldown.
        last_fail = _jwks_cache.get("last_fail_at", 0.0)
        if cached is not None and (now - last_fail) < _JWKS_NEG_CACHE_COOLDOWN_S:
            return cached
        try:
            req = urllib.request.Request(_JWKS_URL, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=_JWKS_FETCH_TIMEOUT_S) as resp:
                keys = json.loads(resp.read().decode())["keys"]
        except Exception as e:
            _jwks_cache["last_fail_at"] = time.monotonic()
            if cached is not None:
                # Stale-fallback: el JWKS está caído/lento pero las claves previas
                # siguen firmando los tokens vigentes → seguimos verificando.
                logger.warning(
                    f"[P1-NEON-AUTH] JWKS fetch falló ({type(e).__name__}); "
                    "sirviendo cache previo (stale)."
                )
                return cached
            # Sin cache previo (cold start con JWKS caído) → propagar; verify_neon_jwt
            # lo traduce a None (fail-secure).
            raise
        _jwks_cache["keys"] = keys
        _jwks_cache["fetched_at"] = time.monotonic()
        _jwks_cache["last_fail_at"] = 0.0
        return keys


def _signing_key(token: str, force_refresh: bool = False) -> Ed25519PublicKey:
    kid = jwt.get_unverified_header(token).get("kid")
    if not kid:
        raise ValueError("JWT sin kid en el header")
    keys = _fetch_jwks(force=force_refresh)
    for jwk in keys:
        if jwk.get("kid") == kid and jwk.get("kty") == "OKP":
            return Ed25519PublicKey.from_public_bytes(
                base64.urlsafe_b64decode(jwk["x"] + "==")
            )
    # kid no encontrado → posible rotación de claves; refrescar UNA vez.
    if not force_refresh:
        return _signing_key(token, force_refresh=True)
    raise ValueError(f"No hay JWK que matchee kid={kid!r} en el JWKS")


# [P2-AUTH-FAILURE-OBS · 2026-06-18] Contadores in-process de fallos de verificación
# por razón. Cheap (sin DB en el hot path); el cron `_auth_failure_alert_job`
# (cron_tasks.py) los drena periódicamente, emite el agregado a pipeline_metrics y
# alerta (`auth_failure_rate_high`) si las fallas SOSPECHOSAS (firma mala / JWKS caído /
# kid-miss / claims, NO 'expired' que es normal cada ~15min) superan el umbral. Cierra el
# blind-spot: un credential-spray o un outage del JWKS de Neon eran invisibles hasta que
# los usuarios reportaban. Worker único (uvicorn --workers 1) → un proceso comparte el contador.
_AUTH_FAIL_LOCK = threading.Lock()
_AUTH_FAIL_COUNTS: dict = {
    "expired": 0, "bad_sig": 0, "bad_claims": 0, "jwks_unreachable": 0, "kid_miss": 0, "other": 0,
}


def _classify_auth_fail(exc: Exception) -> str:
    name = type(exc).__name__
    if isinstance(exc, jwt.InvalidSignatureError) or "Signature" in name or name == "DecodeError":
        return "bad_sig"
    if isinstance(exc, (jwt.InvalidAudienceError, jwt.InvalidIssuerError)):
        return "bad_claims"
    if isinstance(exc, (OSError, TimeoutError)) or name in ("URLError", "HTTPError", "timeout"):
        return "jwks_unreachable"
    msg = str(exc).lower()
    if "kid" in msg or "jwk" in msg:
        return "kid_miss"
    return "other"


def _record_auth_fail(reason: str) -> None:
    try:
        with _AUTH_FAIL_LOCK:
            _AUTH_FAIL_COUNTS[reason] = _AUTH_FAIL_COUNTS.get(reason, 0) + 1
    except Exception:
        pass


def drain_auth_fail_counts() -> dict:
    """Snapshot de los contadores + reset a 0 (lo llama el cron de observabilidad).
    Best-effort; nunca lanza."""
    try:
        with _AUTH_FAIL_LOCK:
            snap = dict(_AUTH_FAIL_COUNTS)
            for k in _AUTH_FAIL_COUNTS:
                _AUTH_FAIL_COUNTS[k] = 0
        return snap
    except Exception:
        return {}


def verify_neon_jwt(token: str) -> Optional[dict]:
    """Verifica firma + claims (iss/aud/exp) de un JWT de Neon Auth.

    Retorna el payload (dict) si es válido, o `None` si no (fail-secure).
    NUNCA lanza — todo error se traduce a `None` para que el caller trate
    el token como no-autenticado sin filtrar detalle al cliente.
    """
    if not NEON_AUTH_BASE_URL or not _ORIGIN or not token:
        return None
    try:
        key = _signing_key(token)
        payload = jwt.decode(
            token,
            key=key,
            algorithms=["EdDSA"],
            issuer=_ORIGIN,
            audience=_ORIGIN,
        )
        return payload
    except jwt.ExpiredSignatureError:
        # Normal cada ~15 min; el frontend refresca y reintenta. No es ataque.
        _record_auth_fail("expired")
        logger.info("[P1-NEON-AUTH] JWT expirado (el cliente debe refrescar).")
        return None
    except Exception as e:
        # Firma inválida, kid desconocido, JWKS inalcanzable, claims malos, etc.
        _record_auth_fail(_classify_auth_fail(e))
        logger.warning(f"[P1-NEON-AUTH] Verificación de JWT falló: {type(e).__name__}")
        return None
