"""[P1-PLAN-LOTE-147 · 2026-09-21] «Continuar con Google» en la app de iOS, sin SDK de Google.

Por qué existe: en la app nativa el OAuth por REDIRECCIÓN no vuelve (`P1-IOS-OAUTH-GATE`) — Neon Auth manda al
navegador y `capacitor://localhost/...` no es una dirección que Safari acepte. El binario abre ahora la pantalla de
Google en una `ASWebAuthenticationSession` (marco del propio iOS, cero dependencias nuevas) que SÍ sabe volver por un
esquema propio, y trae un CÓDIGO de autorización. Este módulo lo canjea y verifica lo que Google devuelve.

Dos piezas, y el orden importa:

  1. `exchange_code_for_id_token` — canje PKCE contra `oauth2.googleapis.com/token`. Un cliente de tipo iOS **no
     tiene secreto**: lo que prueba que el canje lo pide quien inició el flujo es el `code_verifier` (PKCE, RFC 7636).
     Por eso el verifier lo genera y lo guarda el cliente, y jamás viaja hasta este momento.
  2. `verify_google_id_token` — el `id_token` que viene en esa respuesta se verifica igual que el de Apple: firma
     RS256 contra el JWKS público de Google, `iss`, `aud` = NUESTRO cliente, `exp`, `nonce` y frescura.

**Se verifica aunque venga del canje.** Podría parecer redundante —el token llega por TLS desde Google— pero la firma
es lo que hace que el resto del sistema no dependa de que este proceso haya hablado con el Google correcto, y el `aud`
es lo que impide que un token emitido para OTRA app sirva aquí. Es la misma regla que P0-AUDIT-1: nunca un claim sin
verificar.

Cero secretos: el client_id de un cliente iOS es público y viaja dentro del binario."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from typing import Optional

import jwt

from knobs import _env_bool, _env_int, _env_str

logger = logging.getLogger(__name__)

# Google emite con los dos `iss` desde hace años; ambos son válidos y hay tokens vivos de los dos.
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
_GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_JWKS_TTL_S = 6 * 3600.0
_JWKS_TIMEOUT_S = 5.0
_JWKS_NEG_COOLDOWN_S = 30.0
_TOKEN_TIMEOUT_S = 12.0

_jwks_lock = threading.Lock()
_jwks_cache: dict = {"keys": None, "fetched_at": 0.0, "last_fail_at": 0.0}


def google_signin_enabled() -> bool:
    """Interruptor de emergencia, como el de Apple: encendido salvo que el operador lo apague sin redeploy."""
    return _env_bool("MEALFIT_GOOGLE_SIGNIN", True)


def google_ios_client_id() -> str:
    """El cliente de tipo iOS del proyecto `mealfitt`. Público (va en el binario); el knob permite rotarlo."""
    return (_env_str("MEALFIT_GOOGLE_IOS_CLIENT_ID",
                     "323329713741-7o63vat4382kpdg00vun714ag6i2em23.apps.googleusercontent.com") or "").strip()


def _max_age_s() -> int:
    return max(60, min(3600, _env_int("MEALFIT_GOOGLE_TOKEN_MAX_AGE_S", 600)))


def _fetch_jwks(force: bool = False) -> list:
    now = time.monotonic()
    with _jwks_lock:
        cached = _jwks_cache["keys"]
        if not force and cached is not None and (now - _jwks_cache["fetched_at"]) < _JWKS_TTL_S:
            return cached
        if cached is not None and (now - _jwks_cache["last_fail_at"]) < _JWKS_NEG_COOLDOWN_S:
            return cached
        try:
            req = urllib.request.Request(_GOOGLE_JWKS_URL, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=_JWKS_TIMEOUT_S) as resp:
                keys = json.loads(resp.read().decode())["keys"]
        except Exception as e:
            _jwks_cache["last_fail_at"] = time.monotonic()
            if cached is not None:
                logger.warning(f"[P1-PLAN-LOTE-147] JWKS de Google inalcanzable ({type(e).__name__}); sirviendo el previo.")
                return cached
            raise
        _jwks_cache.update(keys=keys, fetched_at=time.monotonic(), last_fail_at=0.0)
        return keys


def _signing_key(token: str, force_refresh: bool = False):
    kid = jwt.get_unverified_header(token).get("kid")
    if not kid:
        raise ValueError("id_token sin kid")
    for jwk in _fetch_jwks(force=force_refresh):
        if jwk.get("kid") == kid and jwk.get("kty") == "RSA":
            return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
    if not force_refresh:                      # Google rota sus claves a menudo: refrescar UNA vez
        return _signing_key(token, force_refresh=True)
    raise ValueError(f"ningún JWK de Google casa con kid={kid!r}")


def _as_bool(v) -> bool:
    return v is True or str(v).strip().lower() == "true"


def exchange_code_for_id_token(code: str, code_verifier: str, redirect_uri: str) -> Optional[str]:
    """Canje PKCE. Devuelve el `id_token` en crudo, o `None`. NO lo verifica: de eso va la función de abajo."""
    if not code or not code_verifier or len(code_verifier) < 43 or not redirect_uri:
        return None
    # El redirect de un cliente iOS es el client_id AL REVÉS. Fijarlo aquí, y no aceptar el que diga el cliente,
    # cierra que alguien use nuestro client_id para canjear un código contra una dirección suya.
    esperado = _redirect_uri_esperado()
    if redirect_uri != esperado:
        logger.warning("[P1-PLAN-LOTE-147] redirect_uri ajeno en el canje — rechazado.")
        return None
    datos = urllib.parse.urlencode({
        "client_id": google_ios_client_id(),
        "code": code,
        "code_verifier": code_verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }).encode()
    try:
        req = urllib.request.Request(_GOOGLE_TOKEN_URL, data=datos,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=_TOKEN_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as e:
        logger.info(f"[P1-PLAN-LOTE-147] el canje con Google falló: {type(e).__name__}")
        return None
    return str(payload.get("id_token") or "").strip() or None


def _redirect_uri_esperado() -> str:
    cid = google_ios_client_id()
    base = cid[:-len(".apps.googleusercontent.com")] if cid.endswith(".apps.googleusercontent.com") else cid
    return f"com.googleusercontent.apps.{base}:/oauth2redirect"


def verify_google_id_token(token: str, raw_nonce: str) -> Optional[dict]:
    """Devuelve `{sub, email, email_verified, is_private_email}` o `None`.
    tooltip-anchor: P1-PLAN-LOTE-147-VERIFY"""
    if not token or not raw_nonce or len(raw_nonce) < 16 or len(token) > 8192:
        return None
    try:
        key = _signing_key(token)
        claims = jwt.decode(
            token, key,
            algorithms=["RS256"],              # FIJO: nunca el `alg` del header
            audience=google_ios_client_id(),
            issuer=list(GOOGLE_ISSUERS),
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            leeway=30,
        )
    except Exception as e:
        logger.info(f"[P1-PLAN-LOTE-147] id_token de Google rechazado: {type(e).__name__}")
        return None
    # Google copia el `nonce` TAL CUAL en el token (a diferencia del nativo de Apple, donde se manda su hash). Se
    # aceptan los dos para no depender de esa asimetría: ambos salen del mismo aleatorio de 32 bytes de ESTA sesión,
    # así que aceptar las dos formas no abre nada — lo que se exige sigue siendo conocer el nonce.
    visto = str(claims.get("nonce") or "")
    if not (hmac.compare_digest(visto, raw_nonce)
            or hmac.compare_digest(visto, hashlib.sha256(raw_nonce.encode("utf-8")).hexdigest())):
        logger.warning("[P1-PLAN-LOTE-147] id_token con nonce ajeno o ausente — rechazado.")
        return None
    try:
        if time.time() - float(claims["iat"]) > _max_age_s():
            logger.info("[P1-PLAN-LOTE-147] id_token viejo — rechazado.")
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
        "is_private_email": False,             # Google no tiene relay: el correo que da es el real
        "name": str(claims.get("name") or "").strip() or None,
    }
