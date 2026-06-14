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
_jwks_cache: dict = {"keys": None, "fetched_at": 0.0}
_jwks_lock = threading.Lock()
_JWKS_FETCH_TIMEOUT_S = 10


def is_neon_auth_configured() -> bool:
    """True si `NEON_AUTH_BASE_URL` está seteado (auth verificable)."""
    return bool(NEON_AUTH_BASE_URL)


def _fetch_jwks(force: bool = False) -> list:
    now = time.monotonic()
    with _jwks_lock:
        cached = _jwks_cache["keys"]
        if (not force and cached is not None
                and (now - _jwks_cache["fetched_at"]) < _JWKS_TTL_S):
            return cached
        req = urllib.request.Request(_JWKS_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_JWKS_FETCH_TIMEOUT_S) as resp:
            keys = json.loads(resp.read().decode())["keys"]
        _jwks_cache["keys"] = keys
        _jwks_cache["fetched_at"] = now
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


def verify_neon_jwt(token: str) -> Optional[dict]:
    """Verifica firma + claims (iss/aud/exp) de un JWT de Neon Auth.

    Retorna el payload (dict) si es válido, o `None` si no (fail-secure).
    NUNCA lanza — todo error se traduce a `None` para que el caller trate
    el token como no-autenticado sin filtrar detalle al cliente.
    """
    if not NEON_AUTH_BASE_URL or not token:
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
        logger.info("[P1-NEON-AUTH] JWT expirado (el cliente debe refrescar).")
        return None
    except Exception as e:
        # Firma inválida, kid desconocido, JWKS inalcanzable, claims malos, etc.
        logger.warning(f"[P1-NEON-AUTH] Verificación de JWT falló: {type(e).__name__}")
        return None
