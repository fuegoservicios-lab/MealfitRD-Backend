# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-146 · 2026-09-20] El identity token de «Sign in with Apple» nativo: qué se acepta y qué no.

Funcional, sin red: se firma con una RSA local y se le sirve su JWK al verificador. Cada caso es un ataque
o un descuido concreto; el verificador es la ÚNICA puerta de ese login (simétrico a P0-AUDIT-1)."""
from __future__ import annotations

import hashlib
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

import apple_auth

_NONCE = "n" * 32
_KID = "kid-de-prueba"


@pytest.fixture()
def firma(monkeypatch):
    clave = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(clave.public_key()))
    jwk.update(kid=_KID, alg="RS256", use="sig")
    monkeypatch.setattr(apple_auth, "_fetch_jwks", lambda force=False: [jwk])

    def _firmar(clave_firma=None, **cambios):
        ahora = int(time.time())
        claims = {
            "iss": apple_auth.APPLE_ISSUER, "aud": "com.bioboros.app", "sub": "001234.abcdef.5678",
            "iat": ahora, "exp": ahora + 600, "email": "Alguien@Example.com", "email_verified": "true",
            "nonce": hashlib.sha256(_NONCE.encode()).hexdigest(),
        }
        claims.update(cambios)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(claims, clave_firma or clave, algorithm="RS256", headers={"kid": _KID})

    return _firmar


def test_un_token_bueno_entra_y_el_correo_sale_normalizado(firma):
    r = apple_auth.verify_apple_identity_token(firma(), _NONCE)
    assert r == {"sub": "001234.abcdef.5678", "email": "alguien@example.com", "email_verified": True, "is_private_email": False}


def test_firmado_por_otro_no_entra(firma):
    intruso = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert apple_auth.verify_apple_identity_token(firma(clave_firma=intruso), _NONCE) is None


@pytest.mark.parametrize("cambio", [
    {"aud": "com.otra.app"},                       # token de OTRA app: el ataque clásico a este flujo
    {"iss": "https://appleid.apple.com.evil"},
    {"exp": int(time.time()) - 3600},
    {"iat": int(time.time()) - 3600, "exp": int(time.time()) + 600},   # vigente pero viejo: un token se canjea al instante
    {"nonce": hashlib.sha256(b"otro-nonce-distinto").hexdigest()},
    {"nonce": None},
    {"sub": None},
])
def test_cada_claim_torcido_cierra_la_puerta(firma, cambio):
    assert apple_auth.verify_apple_identity_token(firma(**cambio), _NONCE) is None


def test_alg_none_y_hs256_no_cuelan(firma):
    """El algoritmo es FIJO (RS256): ni `none` ni un HS256 firmado con la clave pública como secreto."""
    ahora = int(time.time())
    base = {"iss": apple_auth.APPLE_ISSUER, "aud": "com.bioboros.app", "sub": "x", "iat": ahora, "exp": ahora + 600,
            "nonce": hashlib.sha256(_NONCE.encode()).hexdigest()}
    sin_firma = jwt.encode(base, key="", algorithm="none", headers={"kid": _KID})
    assert apple_auth.verify_apple_identity_token(sin_firma, _NONCE) is None
    hs = jwt.encode(base, key="s" * 40, algorithm="HS256", headers={"kid": _KID})
    assert apple_auth.verify_apple_identity_token(hs, _NONCE) is None


def test_sin_nonce_crudo_ni_se_intenta(firma):
    assert apple_auth.verify_apple_identity_token(firma(), "") is None
    assert apple_auth.verify_apple_identity_token(firma(), "corto") is None
    assert apple_auth.verify_apple_identity_token("", _NONCE) is None


def test_correo_oculto_y_correo_sin_verificar(firma):
    oculto = apple_auth.verify_apple_identity_token(
        firma(email="abc123@privaterelay.appleid.com", is_private_email="true"), _NONCE)
    assert oculto["is_private_email"] is True and oculto["email_verified"] is True
    sin_verificar = apple_auth.verify_apple_identity_token(firma(email_verified="false"), _NONCE)
    assert sin_verificar["email_verified"] is False, "un correo sin verificar JAMÁS puede enlazar una cuenta existente"
    sin_correo = apple_auth.verify_apple_identity_token(firma(email=None, email_verified=None), _NONCE)
    assert sin_correo["email"] is None and sin_correo["email_verified"] is False


def test_encendido_por_defecto_con_interruptor_de_emergencia(monkeypatch):
    monkeypatch.delenv("MEALFIT_APPLE_SIGNIN", raising=False)
    assert apple_auth.apple_signin_enabled() is True
    monkeypatch.setenv("MEALFIT_APPLE_SIGNIN", "false")
    assert apple_auth.apple_signin_enabled() is False
