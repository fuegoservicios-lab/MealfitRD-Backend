# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-146 · 2026-09-20 · genérico en el 147] De una identidad social verificada a un usuario, y el endpoint que la canjea.

La base va SIMULADA en memoria (dos dicts con la forma de `neon_auth."user"` y `neon_auth.account`): ningún test
escribe en la base real. Cada caso es una decisión de seguridad del diseño."""
from __future__ import annotations

import inspect
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import social_identity
import routers.auth_session as auth_session


class _Base:
    def __init__(self):
        self.users = {}      # id -> {id, email, name, banned}
        self.accounts = []   # {accountId, providerId, userId}

    def query(self, sql, params=None, fetch_all=False, **_):
        if "FROM neon_auth.account a" in sql:
            prov, sub = params
            for a in self.accounts:
                if a["providerId"] == prov and a["accountId"] == sub:
                    return [dict(self.users[a["userId"]])]
            return []
        if 'FROM neon_auth."user" WHERE lower(email)' in sql:
            return [dict(u) for u in self.users.values() if u["email"].lower() == params[0]][:1]
        raise AssertionError(sql)

    def write(self, sql, params=None, returning=False, **_):
        if 'INSERT INTO neon_auth."user"' in sql:
            nombre, email = params
            if any(u["email"].lower() == email for u in self.users.values()):
                raise RuntimeError("unique_violation")
            uid = str(uuid.uuid4())
            self.users[uid] = {"id": uid, "email": email, "name": nombre, "banned": None}
            return [{"id": uid}]
        if "INSERT INTO neon_auth.account" in sql:
            sub, prov, uid = params
            self.accounts.append({"accountId": sub, "providerId": prov, "userId": uid})
            return True
        raise AssertionError(sql)


@pytest.fixture()
def base(monkeypatch):
    b = _Base()
    monkeypatch.setattr(social_identity, "execute_sql_query", b.query)
    monkeypatch.setattr(social_identity, "execute_sql_write", b.write)
    return b


def _ident(**k):
    d = {"sub": "001.abc", "email": "ana@example.com", "email_verified": True, "is_private_email": False}
    d.update(k)
    return d


def test_correo_verificado_con_cuenta_entra_a_ESA_cuenta_y_queda_enlazado(base):
    base.users["u-1"] = {"id": "u-1", "email": "Ana@Example.com", "name": "Ana", "banned": None}
    r = social_identity.resolve_social_user("apple", _ident())
    assert r["user_id"] == "u-1" and r["created"] is False and r["linked"] is True
    assert base.accounts == [{"accountId": "001.abc", "providerId": "apple", "userId": "u-1"}]
    # segunda vez: por el enlace, sin volver a mirar el correo ni duplicar la fila
    r2 = social_identity.resolve_social_user("apple", _ident(email=None, email_verified=False))
    assert r2["user_id"] == "u-1" and len(base.accounts) == 1


def test_un_correo_SIN_verificar_jamas_enlaza_una_cuenta_ajena(base):
    base.users["u-victima"] = {"id": "u-victima", "email": "ana@example.com", "name": "Ana", "banned": None}
    with pytest.raises(social_identity.SocialIdentityError) as e:
        social_identity.resolve_social_user("apple", _ident(email_verified=False))
    assert e.value.code == "social_email_unverified" and base.accounts == []


def test_nadie_con_ese_correo_nace_la_identidad_en_neon_auth(base):
    r = social_identity.resolve_social_user("apple", _ident(), name="Ana  <b>Pérez</b>")
    assert r["created"] is True and r["user_id"] in base.users
    assert base.users[r["user_id"]]["name"] == "Ana  bPérez/b", "el nombre del cliente se limpia de < >"
    assert base.accounts[0]["userId"] == r["user_id"]


def test_correo_oculto_cuenta_nueva_con_el_relay_y_nombre_neutro(base):
    r = social_identity.resolve_social_user("apple", _ident(email="x9z@privaterelay.appleid.com", is_private_email=True))
    assert r["created"] is True and base.users[r["user_id"]]["name"] == "Usuario"


def test_vetado_no_entra_ni_por_enlace_ni_por_correo(base):
    base.users["u-b"] = {"id": "u-b", "email": "ana@example.com", "name": "Ana", "banned": True}
    with pytest.raises(social_identity.SocialIdentityError) as e:
        social_identity.resolve_social_user("apple", _ident())
    assert e.value.code == "account_banned"
    base.accounts.append({"accountId": "001.abc", "providerId": "apple", "userId": "u-b"})
    with pytest.raises(social_identity.SocialIdentityError):
        social_identity.resolve_social_user("apple", _ident())


def test_sin_correo_y_sin_enlace_no_se_inventa_una_identidad(base):
    with pytest.raises(social_identity.SocialIdentityError) as e:
        social_identity.resolve_social_user("apple", _ident(email=None, email_verified=False))
    assert e.value.code == "social_no_email" and not base.users


def test_doble_toque_el_alta_que_choca_se_resuelve_por_correo(base, monkeypatch):
    original = base.write

    def _choque(sql, params=None, returning=False, **k):
        if 'INSERT INTO neon_auth."user"' in sql:
            base.users["u-otra"] = {"id": "u-otra", "email": params[1], "name": params[0], "banned": None}
            raise RuntimeError("unique_violation")      # la OTRA petición ganó la carrera
        return original(sql, params, returning, **k)

    monkeypatch.setattr(social_identity, "execute_sql_write", _choque)
    r = social_identity.resolve_social_user("apple", _ident())
    assert r["user_id"] == "u-otra" and len(base.accounts) == 1


# ─────────────────────────── el endpoint ───────────────────────────

def _cliente(monkeypatch, *, encendido=True, identidad=None, usuario=None, error=None):
    import apple_auth
    monkeypatch.setattr(apple_auth, "apple_signin_enabled", lambda: encendido)
    monkeypatch.setattr(apple_auth, "verify_apple_identity_token", lambda t, n: identidad)

    def _resolver(proveedor, ident, name=None):
        if error:
            raise social_identity.SocialIdentityError(error)
        return usuario

    monkeypatch.setattr(social_identity, "resolve_social_user", _resolver)
    monkeypatch.setattr(auth_session, "session_cookies_enabled", lambda: True)
    monkeypatch.setattr(auth_session, "set_session_cookie", lambda resp, uid, iat=None: f"mf-token-{uid}")
    monkeypatch.setattr(auth_session, "derive_form_key", lambda uid: f"fk-{uid}")
    monkeypatch.setattr(auth_session, "ensure_user_profile_exists", lambda *a, **k: None)
    monkeypatch.setattr(auth_session._APPLE_NATIVE_LIMITER, "__call__", lambda *a, **k: None, raising=False)
    app = FastAPI()
    app.include_router(auth_session.router)
    app.dependency_overrides[auth_session._APPLE_NATIVE_LIMITER] = lambda: None
    return TestClient(app)


_CUERPO = {"identity_token": "jwt", "nonce": "n" * 32}


def test_apagado_por_knob_el_endpoint_no_existe(monkeypatch):
    c = _cliente(monkeypatch, encendido=False, identidad=_ident())
    assert c.post("/api/auth/apple/native", json=_CUERPO).status_code == 404


def test_token_que_no_verifica_401_sin_sesion(monkeypatch):
    c = _cliente(monkeypatch, identidad=None)
    r = c.post("/api/auth/apple/native", json=_CUERPO)
    assert r.status_code == 401 and "set-cookie" not in {k.lower() for k in r.headers}


def test_identidad_verificada_emite_la_misma_sesion_que_el_otp(monkeypatch):
    c = _cliente(monkeypatch, identidad=_ident(),
                 usuario={"user_id": "u-1", "email": "ana@example.com", "name": "Ana", "created": False, "linked": True})
    r = c.post("/api/auth/apple/native", json=_CUERPO)
    assert r.status_code == 200
    assert r.json() == {"ok": True, "user_id": "u-1", "email": "ana@example.com", "token": "mf-token-u-1",
                        "form_key": "fk-u-1", "session_cookie": True, "created": False}


def test_los_errores_de_identidad_salen_con_su_codigo(monkeypatch):
    c = _cliente(monkeypatch, identidad=_ident(), error="social_email_unverified")
    r = c.post("/api/auth/apple/native", json=_CUERPO)
    assert r.status_code == 409 and r.json() == {"ok": False, "error_code": "social_email_unverified"}
    c = _cliente(monkeypatch, identidad=_ident(), error="account_banned")
    assert c.post("/api/auth/apple/native", json=_CUERPO).status_code == 403


def test_tiene_su_propio_limitador_y_verifica_ANTES_de_tocar_la_base():
    src = inspect.getsource(auth_session.apple_native_sign_in)
    assert "Depends(_APPLE_NATIVE_LIMITER)" in src
    assert src.index("verify_apple_identity_token, token, nonce") < src.index("resolve_social_user, \"apple\", identidad")
    assert src.index("if not identidad:") < src.index("resolve_social_user, \"apple\", identidad")
