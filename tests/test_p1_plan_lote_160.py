# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-160 · 2026-09-22] «Continuar con Google» en Android: el token llega YA emitido.

EL DUEÑO: «dale con el continuar con google en android».

POR QUÉ NO VALE EL CAMINO DE iOS. El lote 147 abre la pantalla de Google en una vista web del sistema y
vuelve por un esquema propio con un CÓDIGO que luego canjeamos (PKCE). Google retiró ese destino en Android
— «Custom URI schemes are no longer supported on Android» —, así que no hay vuelta y no hay código. El
camino vigente es Credential Manager, que entrega el `id_token` firmado y deja el canje sin nada que canjear.

LO QUE ESTE TEST VIGILA, y es lo único nuevo que de verdad decide quién eres: **la AUDIENCIA**.

En Android el token no se emite para el cliente de tipo Android sino para el de tipo **Web** (Google exige
pasarlo como `serverClientId`). Son dos clientes distintos del mismo proyecto, y el `aud` es la única
diferencia entre «este token es para nuestra app de Android» y «este token es para nuestra app de iOS».
Por eso la audiencia se pasa por PARÁMETRO y no se acumulan las dos en una lista: una lista dejaría pasar
un token de iOS por la puerta de Android y al revés, borrando esa distinción sin que nada se pusiera rojo.

Tooltip-anchor: P1-PLAN-LOTE-160
"""
from __future__ import annotations

import inspect
import json
import re
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

import google_auth
import social_identity
import routers.auth_session as auth_session

_NONCE = "n" * 32
_KID = "kid-google-android"
_AUD_WEB = "323329713741-qqcajd7sslluuegc1hq0pcdcrik0mvkb.apps.googleusercontent.com"
_AUD_IOS = "323329713741-7o63vat4382kpdg00vun714ag6i2em23.apps.googleusercontent.com"


@pytest.fixture()
def firma(monkeypatch):
    clave = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(clave.public_key()))
    jwk.update(kid=_KID, alg="RS256", use="sig")
    monkeypatch.setattr(google_auth, "_fetch_jwks", lambda force=False: [jwk])

    def _firmar(**cambios):
        ahora = int(time.time())
        claims = {
            "iss": "https://accounts.google.com", "aud": _AUD_WEB, "sub": "104729384756102938475",
            "iat": ahora, "exp": ahora + 600, "email": "Angelo@Example.com", "email_verified": True,
            "nonce": _NONCE, "name": "Angelo Brito",
        }
        claims.update(cambios)
        return jwt.encode(claims, clave, algorithm="RS256", headers={"kid": _KID})

    return _firmar


# ─────────────────────────── la audiencia ───────────────────────────

def test_la_audiencia_de_android_es_el_cliente_WEB_y_no_el_de_ios(monkeypatch):
    """El error natural sería poner aquí el cliente de tipo Android. Google no emite para ése."""
    monkeypatch.delenv("MEALFIT_GOOGLE_ANDROID_CLIENT_ID", raising=False)
    assert google_auth.google_android_audience() == _AUD_WEB
    assert google_auth.google_android_audience() != google_auth.google_ios_client_id()


def test_un_token_de_android_entra_y_el_correo_sale_normalizado(firma):
    ident = google_auth.verify_google_id_token(firma(), _NONCE, _AUD_WEB)
    assert ident and ident["sub"] == "104729384756102938475"
    assert ident["email"] == "angelo@example.com" and ident["email_verified"] is True


def test_el_token_de_UNA_de_nuestras_apps_no_entra_por_la_puerta_de_la_OTRA(firma):
    """La prueba de que la separación existe: mismo proyecto, misma firma, `aud` distinto.

    Si alguien «simplifica» pasando las dos audiencias en una lista, este test es el que lo acusa.
    """
    token_de_ios = firma(aud=_AUD_IOS)
    assert google_auth.verify_google_id_token(token_de_ios, _NONCE, _AUD_WEB) is None
    # Y la simétrica: el de Android tampoco vale por la puerta de iOS (que es la que no pasa audiencia).
    assert google_auth.verify_google_id_token(firma(), _NONCE) is None


def test_sin_audiencia_configurada_no_se_verifica_NADA(firma, monkeypatch):
    """Fail-secure. Sin saber para quién se emitió un token, la respuesta correcta no es «confía»."""
    assert google_auth.verify_google_id_token(firma(), _NONCE, "") is None


def test_el_interruptor_de_android_no_tumba_al_de_ios(monkeypatch):
    """Rollback sin redeploy, y SELECTIVO: si falla el camino nuevo, el viejo sigue en pie.

    Vaciar el client_id NO sirve para apagarlo —`_env_str` cae al default con la env vacía— y un apagado
    que no apaga es peor que no tenerlo: el operador cree que cerró la puerta.
    """
    monkeypatch.delenv("MEALFIT_GOOGLE_ANDROID_SIGNIN", raising=False)
    assert google_auth.google_android_audience() == _AUD_WEB
    monkeypatch.setenv("MEALFIT_GOOGLE_ANDROID_CLIENT_ID", "")
    assert google_auth.google_android_audience() == _AUD_WEB, "vaciar el id no es el interruptor"
    monkeypatch.setenv("MEALFIT_GOOGLE_ANDROID_SIGNIN", "false")
    assert google_auth.google_android_audience() == ""
    assert google_auth.google_signin_enabled() is True, "iOS y la web siguen encendidos"


def test_el_nonce_sigue_siendo_obligatorio_en_android(firma):
    """Es lo que ata el token a ESTA pulsación; Credential Manager lo copia tal cual."""
    assert google_auth.verify_google_id_token(firma(nonce="otro" * 8), _NONCE, _AUD_WEB) is None
    assert google_auth.verify_google_id_token(firma(nonce=None), _NONCE, _AUD_WEB) is None


# ─────────────────────────── el endpoint ───────────────────────────

_CUERPO_ANDROID = {"id_token": "jwt-de-android", "nonce": _NONCE}
_CUERPO_IOS = {"code": "cod", "code_verifier": "v" * 50, "nonce": _NONCE,
               "redirect_uri": f"com.googleusercontent.apps.{_AUD_IOS.split('.')[0]}:/oauth2redirect"}


def _cliente(monkeypatch, *, identidad=None, usuario=None, canjes=None, audiencias=None):
    """`canjes` y `audiencias` son listas donde se anota lo que de verdad se llamó."""
    monkeypatch.setattr(google_auth, "google_signin_enabled", lambda: True)

    def _canjear(c, v, r):
        if canjes is not None:
            canjes.append((c, v, r))
        return "jwt-canjeado"

    def _verificar(t, n, aud=None):
        if audiencias is not None:
            audiencias.append(aud)
        return identidad

    monkeypatch.setattr(google_auth, "exchange_code_for_id_token", _canjear)
    monkeypatch.setattr(google_auth, "verify_google_id_token", _verificar)
    monkeypatch.setattr(social_identity, "resolve_social_user", lambda p, i, n=None: usuario)
    monkeypatch.setattr(auth_session, "session_cookies_enabled", lambda: True)
    monkeypatch.setattr(auth_session, "set_session_cookie", lambda resp, uid, iat=None: f"mf-token-{uid}")
    monkeypatch.setattr(auth_session, "derive_form_key", lambda uid: f"fk-{uid}")
    monkeypatch.setattr(auth_session, "ensure_user_profile_exists", lambda *a, **k: None)
    app = FastAPI()
    app.include_router(auth_session.router)
    app.dependency_overrides[auth_session._GOOGLE_NATIVE_LIMITER] = lambda: None
    return TestClient(app)


_IDENT = {"sub": "1", "email": "a@b.co", "email_verified": True, "name": "Ana"}
_USUARIO = {"user_id": "u-1", "email": "a@b.co", "name": "Ana", "created": False, "linked": False}


def test_android_entra_y_recibe_la_MISMA_sesion_que_ios(monkeypatch):
    """Un endpoint, dos formas: lo que decide quién eres no se duplica."""
    c = _cliente(monkeypatch, identidad=_IDENT, usuario=_USUARIO)
    r = c.post("/api/auth/google/native", json=_CUERPO_ANDROID)
    assert r.status_code == 200
    assert r.json() == {"ok": True, "user_id": "u-1", "email": "a@b.co", "token": "mf-token-u-1",
                        "form_key": "fk-u-1", "session_cookie": True, "created": False}


def test_con_id_token_NO_se_canjea_nada(monkeypatch):
    """Canjear un código que no existe sería una llamada a Google garantizada a fallar."""
    canjes, audiencias = [], []
    c = _cliente(monkeypatch, identidad=_IDENT, usuario=_USUARIO, canjes=canjes, audiencias=audiencias)
    assert c.post("/api/auth/google/native", json=_CUERPO_ANDROID).status_code == 200
    assert canjes == [], "la rama de Android no debe tocar el canje"
    assert audiencias == [_AUD_WEB], "y debe verificar contra la audiencia de Android"


def test_el_camino_de_ios_sigue_intacto(monkeypatch):
    """Regresión del 147: sin `id_token`, se canjea y se verifica SIN audiencia explícita (iOS)."""
    canjes, audiencias = [], []
    c = _cliente(monkeypatch, identidad=_IDENT, usuario=_USUARIO, canjes=canjes, audiencias=audiencias)
    assert c.post("/api/auth/google/native", json=_CUERPO_IOS).status_code == 200
    assert len(canjes) == 1 and canjes[0][0] == "cod"
    assert audiencias == [None]


def test_la_rama_la_elige_la_FORMA_del_cuerpo_no_un_campo_del_cliente(monkeypatch):
    """Un cliente que mienta sobre su plataforma no gana nada: no hay campo de plataforma que mentir.

    El ancla mira las LECTURAS del cuerpo, no la prosa: ninguna debe sacar del cliente el nombre de su
    sistema. Si un día alguien añade `data.get("platform")`, la rama pasaría a elegirla el cliente.
    """
    src = inspect.getsource(auth_session.google_native_sign_in)
    lecturas = re.findall(r'\(data or \{\}\)\.get\("([a-z_]+)"\)', src)
    assert set(lecturas) == {"nonce", "id_token", "code", "code_verifier", "redirect_uri"}, lecturas
    canjes = []
    c = _cliente(monkeypatch, identidad=_IDENT, usuario=_USUARIO, canjes=canjes)
    # Cuerpo con AMBAS formas: manda el `id_token`, que es el que ya viene firmado.
    c.post("/api/auth/google/native", json={**_CUERPO_IOS, **_CUERPO_ANDROID})
    assert canjes == []


def test_sin_audiencia_configurada_el_endpoint_responde_404(monkeypatch):
    """Como el interruptor de emergencia: si el camino está apagado, no existe."""
    monkeypatch.setenv("MEALFIT_GOOGLE_ANDROID_SIGNIN", "false")
    c = _cliente(monkeypatch, identidad=_IDENT, usuario=_USUARIO)
    r = c.post("/api/auth/google/native", json=_CUERPO_ANDROID)
    assert r.status_code == 404
    assert "set-cookie" not in {k.lower() for k in r.headers}


def test_sin_nonce_ninguna_de_las_dos_formas_llega_a_google(monkeypatch):
    canjes, audiencias = [], []
    c = _cliente(monkeypatch, identidad=_IDENT, usuario=_USUARIO, canjes=canjes, audiencias=audiencias)
    for cuerpo in (_CUERPO_ANDROID, _CUERPO_IOS):
        sin_nonce = {k: v for k, v in cuerpo.items() if k != "nonce"}
        assert c.post("/api/auth/google/native", json=sin_nonce).status_code == 401
    assert canjes == [] and audiencias == []


def test_si_el_token_de_android_no_verifica_401_sin_sesion(monkeypatch):
    c = _cliente(monkeypatch, identidad=None)
    r = c.post("/api/auth/google/native", json=_CUERPO_ANDROID)
    assert r.status_code == 401 and "set-cookie" not in {k.lower() for k in r.headers}
