"""[P2-AUTH-TEST-COVERAGE · 2026-06-18] Test FUNCIONAL del path real de
verify_neon_jwt (neon_auth.py) — firma un JWT EdDSA real contra un JWKS fake.

Por qué: antes TODOS los tests que tocaban verify_neon_jwt eran parser-based o lo
stubeaban; ninguno ejercitaba la verificación real. Un test funcional con un token
con `aud` habría cazado el `_ORIGIN=None` (outage silencioso) y un JWKS-down. Cubre
además el stale-fallback nuevo (P1-NEON-AUTH-JWKS-HARDEN): un fallo del fetch sirve
el cache previo en vez de tumbar toda la auth.

Anchor: P2-AUTH-TEST-COVERAGE.
"""
from __future__ import annotations

import base64
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import neon_auth

_ORIGIN = "https://test.neonauth.example"
_KID = "test-kid-1"


def _make_keypair():
    priv = Ed25519PrivateKey.generate()
    pub_raw = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    # JWK x: base64url SIN padding (igual que emite Neon Auth; el código de
    # neon_auth añade "==" al decodificar).
    x = base64.urlsafe_b64encode(pub_raw).decode().rstrip("=")
    jwk = {"kid": _KID, "kty": "OKP", "crv": "Ed25519", "x": x}
    priv_pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return priv_pem, jwk


def _claims(**over):
    base = {
        "sub": "user-uuid-123",
        "iss": _ORIGIN,
        "aud": _ORIGIN,
        "exp": int(time.time()) + 900,
        "iat": int(time.time()),
        "role": "authenticated",
    }
    base.update(over)
    return base


@pytest.fixture
def configured(monkeypatch):
    """Configura neon_auth con un JWKS fake y un origin de prueba."""
    priv_pem, jwk = _make_keypair()
    monkeypatch.setattr(neon_auth, "NEON_AUTH_BASE_URL", _ORIGIN, raising=False)
    monkeypatch.setattr(neon_auth, "_ORIGIN", _ORIGIN, raising=False)
    monkeypatch.setattr(neon_auth, "_JWKS_URL", f"{_ORIGIN}/.well-known/jwks.json", raising=False)
    monkeypatch.setattr(neon_auth, "_fetch_jwks", lambda force=False: [jwk], raising=True)
    return priv_pem, jwk


def _sign(priv_pem, claims, kid=_KID, alg="EdDSA"):
    return jwt.encode(claims, priv_pem, algorithm=alg, headers={"kid": kid})


# ---------------------------------------------------------------------------
# 1. Token válido → payload con sub
# ---------------------------------------------------------------------------
def test_valid_token_returns_payload(configured):
    priv_pem, _ = configured
    token = _sign(priv_pem, _claims())
    payload = neon_auth.verify_neon_jwt(token)
    assert payload is not None
    assert payload["sub"] == "user-uuid-123"
    assert payload["aud"] == _ORIGIN


# ---------------------------------------------------------------------------
# 2. Fail-secure: expirado / aud malo / iss malo / firma mala / alg-confusion
# ---------------------------------------------------------------------------
def test_expired_token_returns_none(configured):
    priv_pem, _ = configured
    token = _sign(priv_pem, _claims(exp=int(time.time()) - 10))
    assert neon_auth.verify_neon_jwt(token) is None


def test_wrong_audience_returns_none(configured):
    priv_pem, _ = configured
    token = _sign(priv_pem, _claims(aud="https://attacker.example"))
    assert neon_auth.verify_neon_jwt(token) is None


def test_wrong_issuer_returns_none(configured):
    priv_pem, _ = configured
    token = _sign(priv_pem, _claims(iss="https://attacker.example"))
    assert neon_auth.verify_neon_jwt(token) is None


def test_tampered_signature_returns_none(configured):
    """Firmado con OTRA clave (no la del JWKS) → firma inválida → None."""
    _, _ = configured
    other_pem, _ = _make_keypair()
    token = _sign(other_pem, _claims())
    assert neon_auth.verify_neon_jwt(token) is None


def test_algorithm_confusion_hs256_returns_none(configured):
    """Un atacante intenta degradar a HMAC (HS256) usando la x pública como
    secreto. El algoritmo fijo ['EdDSA'] lo rechaza → None (no account takeover)."""
    _, jwk = configured
    forged = jwt.encode(_claims(), jwk["x"], algorithm="HS256", headers={"kid": _KID})
    assert neon_auth.verify_neon_jwt(forged) is None


def test_no_token_returns_none(configured):
    assert neon_auth.verify_neon_jwt("") is None
    assert neon_auth.verify_neon_jwt(None) is None


# ---------------------------------------------------------------------------
# 3. Outage de config: _ORIGIN None → None (no decodifica con aud=None)
# ---------------------------------------------------------------------------
def test_missing_origin_returns_none(monkeypatch):
    priv_pem, jwk = _make_keypair()
    monkeypatch.setattr(neon_auth, "NEON_AUTH_BASE_URL", _ORIGIN, raising=False)
    monkeypatch.setattr(neon_auth, "_ORIGIN", None, raising=False)  # mal configurado
    monkeypatch.setattr(neon_auth, "_fetch_jwks", lambda force=False: [jwk], raising=True)
    token = _sign(priv_pem, _claims())
    assert neon_auth.verify_neon_jwt(token) is None


# ---------------------------------------------------------------------------
# 4. JWKS stale-fallback (P1-NEON-AUTH-JWKS-HARDEN): si el fetch falla pero hay
#    cache previo, _fetch_jwks lo sirve en vez de tumbar la auth.
# ---------------------------------------------------------------------------
def test_jwks_stale_fallback_serves_previous_cache(monkeypatch):
    _, jwk = _make_keypair()
    # Pre-poblar el cache con la clave buena, marcado como stale (fetched_at viejo).
    monkeypatch.setattr(
        neon_auth, "_jwks_cache",
        {"keys": [jwk], "fetched_at": 0.0, "last_fail_at": 0.0}, raising=False,
    )
    monkeypatch.setattr(neon_auth, "_JWKS_URL", f"{_ORIGIN}/.well-known/jwks.json", raising=False)

    def _boom(*a, **k):
        raise OSError("JWKS unreachable")

    monkeypatch.setattr(neon_auth.urllib.request, "urlopen", _boom, raising=True)
    # force=True salta el check de frescura; el fetch falla → debe servir el stale.
    keys = neon_auth._fetch_jwks(force=True)
    assert keys == [jwk]
    assert neon_auth._jwks_cache["last_fail_at"] > 0  # registró el fallo
