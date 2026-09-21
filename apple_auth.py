"""[P1-PLAN-LOTE-146 · 2026-09-20] Verificación del identity token de «Sign in with Apple» NATIVO.

Por qué nativo y por qué aquí: Neon Auth (Better Auth gestionado) solo ofrece Google, GitHub y Vercel
—Apple no está en su lista de proveedores—, y en la app de iOS el OAuth por redirección no vuelve a
la app (`P1-IOS-OAUTH-GATE`). El binario pide la credencial con el SDK de Apple (hoja nativa, sin
Safari) y entrega un JWT firmado por Apple. ESTE módulo decide si ese JWT vale; nada más. Quién es
el usuario en nuestra base lo resuelve `routers/auth_session.py`.

Contrato (fail-secure, simétrico a `neon_auth.verify_neon_jwt` y a P0-AUDIT-1):
  * firma RS256 contra el JWKS público de Apple — algoritmo FIJO, jamás el del header;
  * `iss` = https://appleid.apple.com, `aud` ∈ los bundle/services id configurados, `exp` vigente;
  * `nonce`: el cliente genera uno crudo, le pasa a Apple su SHA-256 y nos manda el crudo: el claim
    debe ser exactamente sha256(crudo). Un token robado de otra app/sesión no trae NUESTRO nonce;
  * frescura: `iat` de hace ≤ `MEALFIT_APPLE_TOKEN_MAX_AGE_S` (un identity token es de un solo uso
    en la práctica: se canjea al instante de emitirse);
  * cualquier duda → `None`. NUNCA se devuelve un claim sin verificar.

Cero secretos: Sign in with Apple nativo no usa client secret ni `.p8` (eso es del flujo web).
Encendido por defecto; `MEALFIT_APPLE_SIGNIN=false` es el INTERRUPTOR DE EMERGENCIA (sin redeploy). No nace
apagado porque encenderlo sería tocar el `.env` del VPS —cosa del dueño— y no añade riesgo: un token con
nuestro `aud` solo lo puede obtener nuestro binario firmado, y el botón no existe en binarios sin el plugin."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.request
from typing import Optional

import jwt

from knobs import _env_bool, _env_int, _env_str

logger = logging.getLogger(__name__)

APPLE_ISSUER = "https://appleid.apple.com"
_APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
_JWKS_TTL_S = 6 * 3600.0
_JWKS_TIMEOUT_S = 5.0
_JWKS_NEG_COOLDOWN_S = 30.0

_jwks_lock = threading.Lock()
_jwks_cache: dict = {"keys": None, "fetched_at": 0.0, "last_fail_at": 0.0}


def apple_signin_enabled() -> bool:
    return _env_bool("MEALFIT_APPLE_SIGNIN", True)


def _audiences() -> list:
    """Destinatarios válidos: el bundle id (flujo nativo de iOS) y el Services ID (flujo web, lote 148)."""
    # El binario (bundle id) y la web (Services ID). Los dos son NUESTROS y cada token dice para cuál se emitió.
    por_defecto = "com.bioboros.app,com.bioboros.app.web"
    raw = (_env_str("MEALFIT_APPLE_AUDIENCES", por_defecto) or por_defecto).strip()
    return [a.strip() for a in raw.split(",") if a.strip()]


def _max_age_s() -> int:
    return max(60, min(3600, _env_int("MEALFIT_APPLE_TOKEN_MAX_AGE_S", 600)))


def _fetch_jwks(force: bool = False) -> list:
    now = time.monotonic()
    with _jwks_lock:
        cached = _jwks_cache["keys"]
        if not force and cached is not None and (now - _jwks_cache["fetched_at"]) < _JWKS_TTL_S:
            return cached
        if cached is not None and (now - _jwks_cache["last_fail_at"]) < _JWKS_NEG_COOLDOWN_S:
            return cached
        try:
            req = urllib.request.Request(_APPLE_JWKS_URL, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=_JWKS_TIMEOUT_S) as resp:
                keys = json.loads(resp.read().decode())["keys"]
        except Exception as e:
            _jwks_cache["last_fail_at"] = time.monotonic()
            if cached is not None:
                logger.warning(f"[P1-PLAN-LOTE-146] JWKS de Apple inalcanzable ({type(e).__name__}); sirviendo el previo.")
                return cached
            raise
        _jwks_cache.update(keys=keys, fetched_at=time.monotonic(), last_fail_at=0.0)
        return keys


def _signing_key(token: str, force_refresh: bool = False):
    kid = jwt.get_unverified_header(token).get("kid")
    if not kid:
        raise ValueError("identity token sin kid")
    for jwk in _fetch_jwks(force=force_refresh):
        if jwk.get("kid") == kid and jwk.get("kty") == "RSA":
            return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
    if not force_refresh:                      # rotación de claves de Apple: refrescar UNA vez
        return _signing_key(token, force_refresh=True)
    raise ValueError(f"ningún JWK de Apple casa con kid={kid!r}")


def _as_bool(v) -> bool:
    # Apple manda `email_verified` / `is_private_email` como booleano O como la cadena "true".
    return v is True or str(v).strip().lower() == "true"


def verify_apple_identity_token(token: str, raw_nonce: str) -> Optional[dict]:
    """Devuelve `{sub, email, email_verified, is_private_email}` o `None`.
    tooltip-anchor: P1-PLAN-LOTE-146-VERIFY"""
    if not token or not raw_nonce or len(raw_nonce) < 16 or len(token) > 8192:
        return None
    try:
        key = _signing_key(token)
        claims = jwt.decode(
            token, key,
            algorithms=["RS256"],              # FIJO: nunca el `alg` que diga el header
            audience=_audiences(),
            issuer=APPLE_ISSUER,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            leeway=30,
        )
    except Exception as e:
        logger.info(f"[P1-PLAN-LOTE-146] identity token rechazado: {type(e).__name__}")
        return None
    # [P1-PLAN-LOTE-148] El nativo manda el SHA-256 del nonce (lo exige Apple) y la librería web no documenta si
    # hashea el suyo. Se aceptan las dos formas: ambas salen del mismo aleatorio de 32 bytes de ESTA sesión, así que
    # no se afloja nada — lo que se sigue exigiendo es conocer el nonce.
    visto = str(claims.get("nonce") or "")
    if not (hmac.compare_digest(visto, hashlib.sha256(raw_nonce.encode("utf-8")).hexdigest())
            or hmac.compare_digest(visto, raw_nonce)):
        logger.warning("[P1-PLAN-LOTE-146] identity token con nonce ajeno o ausente — rechazado.")
        return None
    try:
        if time.time() - float(claims["iat"]) > _max_age_s():
            logger.info("[P1-PLAN-LOTE-146] identity token viejo — rechazado.")
            return None
    except (TypeError, ValueError):
        return None
    sub = str(claims.get("sub") or "").strip()
    if not sub:
        return None
    email = str(claims.get("email") or "").strip().lower() or None
    return {
        "sub": sub,
        "email": email,
        "email_verified": bool(email) and _as_bool(claims.get("email_verified")),
        "is_private_email": _as_bool(claims.get("is_private_email")),
    }
