# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-147 · 2026-09-21] «Continuar con Google» nativo en iOS: canje PKCE + verificación del id_token.

Funcional y sin red: se firma con una RSA local, se le sirve su JWK al verificador y el canje con Google va
simulado. Cada caso es un ataque o un descuido concreto — este par de funciones es la única puerta de ese login."""
from __future__ import annotations

import hashlib
import inspect
import json
import time
import urllib.parse

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

import google_auth
import social_identity
import routers.auth_session as auth_session

_NONCE = "n" * 32
_KID = "kid-google-prueba"
_CID = "323329713741-7o63vat4382kpdg00vun714ag6i2em23.apps.googleusercontent.com"
_REDIRECT = "com.googleusercontent.apps.323329713741-7o63vat4382kpdg00vun714ag6i2em23:/oauth2redirect"


@pytest.fixture()
def firma(monkeypatch):
    clave = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(clave.public_key()))
    jwk.update(kid=_KID, alg="RS256", use="sig")
    monkeypatch.setattr(google_auth, "_fetch_jwks", lambda force=False: [jwk])

    def _firmar(clave_firma=None, **cambios):
        ahora = int(time.time())
        claims = {
            "iss": "https://accounts.google.com", "aud": _CID, "sub": "104729384756102938475",
            "iat": ahora, "exp": ahora + 600, "email": "Angelo@Example.com", "email_verified": True,
            "nonce": _NONCE, "name": "Angelo Brito",
        }
        claims.update(cambios)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(claims, clave_firma or clave, algorithm="RS256", headers={"kid": _KID})

    return _firmar


def test_un_token_bueno_entra_y_el_correo_sale_normalizado(firma):
    r = google_auth.verify_google_id_token(firma(), _NONCE)
    assert r["sub"] == "104729384756102938475" and r["email"] == "angelo@example.com"
    assert r["email_verified"] is True and r["is_private_email"] is False and r["name"] == "Angelo Brito"


def test_el_nonce_vale_crudo_o_hasheado_pero_solo_el_NUESTRO(firma):
    """Google lo copia tal cual; Apple nativo manda su hash. Se aceptan ambos — los dos salen del mismo aleatorio."""
    assert google_auth.verify_google_id_token(firma(), _NONCE) is not None
    hasheado = firma(nonce=hashlib.sha256(_NONCE.encode()).hexdigest())
    assert google_auth.verify_google_id_token(hasheado, _NONCE) is not None
    ajeno = firma(nonce=hashlib.sha256(b"de-otra-sesion").hexdigest())
    assert google_auth.verify_google_id_token(ajeno, _NONCE) is None


def test_firmado_por_otro_no_entra(firma):
    intruso = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert google_auth.verify_google_id_token(firma(clave_firma=intruso), _NONCE) is None


@pytest.mark.parametrize("cambio", [
    {"aud": "otro-cliente.apps.googleusercontent.com"},          # token de OTRA app: el ataque clásico
    {"iss": "https://accounts.google.com.evil"},
    {"exp": int(time.time()) - 3600},
    {"iat": int(time.time()) - 3600, "exp": int(time.time()) + 600},   # vigente pero viejo
    {"nonce": None},
    {"sub": None},
])
def test_cada_claim_torcido_cierra_la_puerta(firma, cambio):
    assert google_auth.verify_google_id_token(firma(**cambio), _NONCE) is None


def test_el_otro_iss_de_google_tambien_vale(firma):
    assert google_auth.verify_google_id_token(firma(iss="accounts.google.com"), _NONCE) is not None


def test_alg_none_y_hs256_no_cuelan(firma):
    ahora = int(time.time())
    base = {"iss": "https://accounts.google.com", "aud": _CID, "sub": "x", "iat": ahora, "exp": ahora + 600,
            "nonce": _NONCE}
    assert google_auth.verify_google_id_token(jwt.encode(base, key="", algorithm="none", headers={"kid": _KID}), _NONCE) is None
    assert google_auth.verify_google_id_token(jwt.encode(base, key="s" * 40, algorithm="HS256", headers={"kid": _KID}), _NONCE) is None


def test_correo_sin_verificar_no_enlaza(firma):
    r = google_auth.verify_google_id_token(firma(email_verified=False), _NONCE)
    assert r["email_verified"] is False, "un correo sin verificar JAMÁS puede enlazar una cuenta existente"


# ─────────────────────────── el canje PKCE ───────────────────────────

def test_el_redirect_uri_lo_fija_el_SERVIDOR(monkeypatch):
    """Aceptar el que diga el cliente dejaría usar nuestro client_id para canjear contra una dirección ajena."""
    llamadas = []
    monkeypatch.setattr(google_auth.urllib.request, "urlopen", lambda *a, **k: llamadas.append(a) or None)
    assert google_auth.exchange_code_for_id_token("cod", "v" * 50, "https://evil.example/cb") is None
    assert llamadas == [], "ni se intentó el canje"
    assert google_auth._redirect_uri_esperado() == _REDIRECT


def test_el_canje_manda_el_verifier_y_NINGUN_secreto(monkeypatch):
    visto = {}

    class _Resp:
        def read(self): return json.dumps({"id_token": "el-token"}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _urlopen(req, timeout=None):
        visto["url"] = req.full_url
        visto["body"] = dict(urllib.parse.parse_qsl(req.data.decode()))
        return _Resp()

    monkeypatch.setattr(google_auth.urllib.request, "urlopen", _urlopen)
    assert google_auth.exchange_code_for_id_token("cod", "v" * 50, _REDIRECT) == "el-token"
    assert visto["url"] == "https://oauth2.googleapis.com/token"
    assert visto["body"]["code_verifier"] == "v" * 50
    assert visto["body"]["grant_type"] == "authorization_code"
    assert "client_secret" not in visto["body"], "un cliente iOS no tiene secreto; PKCE es la prueba"


def test_un_verifier_corto_no_se_canjea(monkeypatch):
    monkeypatch.setattr(google_auth.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no debió llamar"))
    assert google_auth.exchange_code_for_id_token("cod", "corto", _REDIRECT) is None
    assert google_auth.exchange_code_for_id_token("", "v" * 50, _REDIRECT) is None


def test_encendido_por_defecto_con_interruptor_de_emergencia(monkeypatch):
    monkeypatch.delenv("MEALFIT_GOOGLE_SIGNIN", raising=False)
    assert google_auth.google_signin_enabled() is True
    monkeypatch.setenv("MEALFIT_GOOGLE_SIGNIN", "false")
    assert google_auth.google_signin_enabled() is False


# ─────────────────────────── el endpoint ───────────────────────────

_CUERPO = {"code": "cod", "code_verifier": "v" * 50, "nonce": _NONCE, "redirect_uri": _REDIRECT}


def _cliente(monkeypatch, *, encendido=True, id_token="jwt", identidad=None, usuario=None, error=None):
    monkeypatch.setattr(google_auth, "google_signin_enabled", lambda: encendido)
    monkeypatch.setattr(google_auth, "exchange_code_for_id_token", lambda c, v, r: id_token)
    monkeypatch.setattr(google_auth, "verify_google_id_token", lambda t, n: identidad)

    def _resolver(proveedor, ident, name=None):
        assert proveedor == "google"
        if error:
            raise social_identity.SocialIdentityError(error)
        return usuario

    monkeypatch.setattr(social_identity, "resolve_social_user", _resolver)
    monkeypatch.setattr(auth_session, "session_cookies_enabled", lambda: True)
    monkeypatch.setattr(auth_session, "set_session_cookie", lambda resp, uid, iat=None: f"mf-token-{uid}")
    monkeypatch.setattr(auth_session, "derive_form_key", lambda uid: f"fk-{uid}")
    monkeypatch.setattr(auth_session, "ensure_user_profile_exists", lambda *a, **k: None)
    app = FastAPI()
    app.include_router(auth_session.router)
    app.dependency_overrides[auth_session._GOOGLE_NATIVE_LIMITER] = lambda: None
    return TestClient(app)


def test_apagado_por_knob_el_endpoint_no_existe(monkeypatch):
    c = _cliente(monkeypatch, encendido=False)
    assert c.post("/api/auth/google/native", json=_CUERPO).status_code == 404


def test_si_el_canje_falla_401_sin_sesion(monkeypatch):
    c = _cliente(monkeypatch, id_token=None)
    r = c.post("/api/auth/google/native", json=_CUERPO)
    assert r.status_code == 401 and "set-cookie" not in {k.lower() for k in r.headers}


def test_si_el_token_no_verifica_401(monkeypatch):
    c = _cliente(monkeypatch, identidad=None)
    assert c.post("/api/auth/google/native", json=_CUERPO).status_code == 401


def test_identidad_verificada_emite_la_misma_sesion_que_el_otp(monkeypatch):
    c = _cliente(monkeypatch, identidad={"sub": "1", "email": "a@b.co", "email_verified": True, "name": "Ana"},
                 usuario={"user_id": "u-1", "email": "a@b.co", "name": "Ana", "created": False, "linked": False})
    r = c.post("/api/auth/google/native", json=_CUERPO)
    assert r.status_code == 200
    assert r.json() == {"ok": True, "user_id": "u-1", "email": "a@b.co", "token": "mf-token-u-1",
                        "form_key": "fk-u-1", "session_cookie": True, "created": False}


def test_cuerpo_incompleto_ni_llega_a_google(monkeypatch):
    c = _cliente(monkeypatch, id_token="jwt")
    for falta in ("code", "code_verifier", "nonce", "redirect_uri"):
        cuerpo = dict(_CUERPO)
        cuerpo.pop(falta)
        assert c.post("/api/auth/google/native", json=cuerpo).status_code == 401, falta


def test_canjea_ANTES_de_verificar_y_verifica_ANTES_de_tocar_la_base():
    src = inspect.getsource(auth_session.google_native_sign_in)
    assert "Depends(_GOOGLE_NATIVE_LIMITER)" in src
    assert src.index("exchange_code_for_id_token, code") < src.index("verify_google_id_token, id_token")
    assert src.index("if not identidad:") < src.index("resolve_social_user, \"google\"")
