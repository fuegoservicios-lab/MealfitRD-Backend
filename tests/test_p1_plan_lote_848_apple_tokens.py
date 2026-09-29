# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-848 · 2026-09-29] Sign in with Apple: canje del código, refresh token cifrado y revocación al borrar.

Auditoría App Store 2026-09-29, fila 3.2 (§A.6): la guía de borrado de cuentas de Apple pide revocar los tokens de
Sign in with Apple con `/auth/revoke` al borrar la cuenta. Sin red: `apple_tokens._post_form` es el único punto de red
y aquí se sustituye; la base se simula donde hace falta.

Qué se fija:
  * el `client_secret` es un JWT ES256 con los claims que Apple exige (iss/sub/aud/exp, header kid);
  * el refresh token se guarda CIFRADO (Fernet) y la migración rechaza uno en claro;
  * el borrado de cuenta revoca ANTES de purgar, después de PayPal, y nunca se bloquea por Apple;
  * sin claves configuradas no hay canje ni revocación, y el login jamás falla por el canje.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import jwt
import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

import apple_tokens

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_MIG = "p1_plan_lote_848_apple_tokens_2026_09_29.sql"
_UID = "11111111-2222-3333-4444-555555555555"
_ENV = ("APPLE_SIWA_KEY_ID", "APPLE_SIWA_TEAM_ID", "APPLE_SIWA_PRIVATE_KEY", "APPLE_SIWA_KEY_FILE",
        "MEALFIT_TOKEN_ENC_KEY", "APNS_TEAM_ID", "MEALFIT_APPLE_SIWA_TOKENS", "MEALFIT_APPLE_SIWA_CLIENT_ID")


@pytest.fixture()
def sin_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(apple_tokens, "_avisado_sin_configurar", False)
    return monkeypatch


@pytest.fixture()
def clave_ec():
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture()
def configurado(sin_env, clave_ec):
    pem = clave_ec.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    sin_env.setenv("APPLE_SIWA_KEY_ID", "ABC123DEFG")
    sin_env.setenv("APPLE_SIWA_TEAM_ID", "TEAM123456")
    sin_env.setenv("APPLE_SIWA_PRIVATE_KEY", pem)
    sin_env.setenv("MEALFIT_TOKEN_ENC_KEY", Fernet.generate_key().decode())
    return sin_env


@pytest.fixture()
def red(monkeypatch):
    llamadas: list = []
    respuestas: dict = {}

    def _post(url, datos):
        llamadas.append((url, dict(datos)))
        r = respuestas.get(url, (200, {}))
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(apple_tokens, "_post_form", _post)
    return llamadas, respuestas


# ─────────────────────────── client_secret ───────────────────────────

def test_client_secret_es_un_jwt_es256_con_los_claims_de_apple(configurado, clave_ec):
    ahora = int(time.time()) - 5
    secreto = apple_tokens.construir_client_secret(ahora=ahora)
    cabecera = jwt.get_unverified_header(secreto)
    assert cabecera["alg"] == "ES256" and cabecera["kid"] == "ABC123DEFG"
    claims = jwt.decode(secreto, clave_ec.public_key(), algorithms=["ES256"],
                        audience="https://appleid.apple.com", options={"verify_exp": False})
    assert claims["iss"] == "TEAM123456"
    assert claims["sub"] == "com.bioboros.app"
    assert claims["aud"] == "https://appleid.apple.com"
    assert claims["iat"] == ahora
    assert 0 < claims["exp"] - claims["iat"] <= 15_777_000, "Apple no acepta un client_secret de más de 6 meses"


def test_el_team_id_de_apns_sirve_si_falta_el_propio_y_el_pem_admite_barras_n(configurado, clave_ec):
    configurado.delenv("APPLE_SIWA_TEAM_ID")
    configurado.setenv("APNS_TEAM_ID", "APNSTEAM99")
    pem = clave_ec.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    configurado.setenv("APPLE_SIWA_PRIVATE_KEY", pem.strip().replace("\n", "\\n"))
    claims = jwt.decode(apple_tokens.construir_client_secret(), clave_ec.public_key(), algorithms=["ES256"],
                        audience="https://appleid.apple.com")
    assert claims["iss"] == "APNSTEAM99"


def test_la_clave_tambien_puede_venir_de_un_fichero(configurado, clave_ec, tmp_path):
    ruta = tmp_path / "AuthKey_ABC123DEFG.p8"
    ruta.write_text(clave_ec.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode(), encoding="utf-8")
    configurado.delenv("APPLE_SIWA_PRIVATE_KEY")
    configurado.setenv("APPLE_SIWA_KEY_FILE", str(ruta))
    assert apple_tokens.configurado()
    jwt.decode(apple_tokens.construir_client_secret(), clave_ec.public_key(), algorithms=["ES256"],
               audience="https://appleid.apple.com")


# ─────────────────────────── cifrado en reposo ───────────────────────────

def test_el_refresh_token_se_guarda_cifrado_y_nunca_en_claro(configurado, monkeypatch):
    import db

    escrituras: list = []
    monkeypatch.setattr(db, "execute_sql_write", lambda q, p=None, **k: escrituras.append((q, p)) or True)
    assert apple_tokens.guardar_refresh_token(_UID, "r.EN-CLARO-123") is True
    (sql, params), = escrituras
    assert "INSERT INTO public.apple_signin_tokens" in sql and "ON CONFLICT (user_id) DO UPDATE" in sql
    assert params[0] == _UID
    assert all("EN-CLARO" not in str(p) for p in params), "el refresh token no viaja en claro a la base"
    assert params[1].startswith("gAAAAA")
    assert apple_tokens.descifrar(params[1]) == "r.EN-CLARO-123"
    # Otra clave no lo abre: el dato de la base sin MEALFIT_TOKEN_ENC_KEY no sirve.
    configurado.setenv("MEALFIT_TOKEN_ENC_KEY", Fernet.generate_key().decode())
    with pytest.raises(Exception):
        apple_tokens.descifrar(params[1])


def test_la_migracion_solo_admite_tokens_fernet(configurado):
    src = (_BACKEND / "migrations" / _MIG).read_text(encoding="utf-8")
    patron = re.search(r"refresh_token_enc ~ '([^']+)'", src).group(1)
    assert re.fullmatch(patron, apple_tokens.cifrar("r.algo")), "un token cifrado de verdad pasa el CHECK"
    assert not re.fullmatch(patron, "r.abc123.def456"), "un refresh token en claro NO pasa el CHECK"


def test_migracion_identica_en_los_dos_dirs_idempotente_y_con_cascada():
    backend = (_BACKEND / "migrations" / _MIG).read_bytes()
    raiz = _ROOT / "migrations" / _MIG
    if raiz.exists():
        assert raiz.read_bytes() == backend, "las dos copias de la migración divergieron"
    src = backend.decode("utf-8")
    assert "CREATE TABLE IF NOT EXISTS public.apple_signin_tokens" in src
    assert "REFERENCES public.user_profiles(id) ON DELETE CASCADE" in src
    assert "REVOKE ALL ON public.apple_signin_tokens FROM PUBLIC;" in src
    for add in re.findall(r"ADD CONSTRAINT (\w+)", src):
        assert f"DROP CONSTRAINT IF EXISTS {add}" in src, f"{add} sin DROP previo (no idempotente)"
    assert "DO $$" in src and "RAISE EXCEPTION" in src


# ─────────────────────────── canje en el login ───────────────────────────

def test_canje_manda_el_codigo_con_client_secret_y_guarda_el_refresh(configurado, red, monkeypatch):
    llamadas, respuestas = red
    respuestas[apple_tokens.APPLE_TOKEN_URL] = (200, {"refresh_token": "r.nuevo", "access_token": "a", "id_token": "i"})
    guardados: list = []
    monkeypatch.setattr(apple_tokens, "guardar_refresh_token", lambda uid, tok: guardados.append((uid, tok)) or True)
    assert apple_tokens.canjear_y_guardar(_UID, "c.codigo") is True
    (url, datos), = llamadas
    assert url == "https://appleid.apple.com/auth/token"
    assert datos["grant_type"] == "authorization_code" and datos["code"] == "c.codigo"
    assert datos["client_id"] == "com.bioboros.app"
    assert jwt.get_unverified_header(datos["client_secret"])["alg"] == "ES256"
    assert guardados == [(_UID, "r.nuevo")]


@pytest.mark.parametrize("respuesta", [
    (400, {"error": "invalid_grant"}), (500, None), (200, {"access_token": "a"}), ConnectionError("sin red"),
])
def test_un_canje_fallido_no_lanza_ni_guarda(configurado, red, monkeypatch, respuesta):
    _, respuestas = red
    respuestas[apple_tokens.APPLE_TOKEN_URL] = respuesta
    monkeypatch.setattr(apple_tokens, "guardar_refresh_token", lambda *a: pytest.fail("no debía guardar"))
    assert apple_tokens.canjear_y_guardar(_UID, "c.codigo") is False


def test_sin_configurar_no_hay_canje_y_se_avisa_una_vez(sin_env, red, caplog):
    llamadas, _ = red
    with caplog.at_level("WARNING"):
        assert apple_tokens.canjear_y_guardar(_UID, "c.codigo") is False
        assert apple_tokens.canjear_y_guardar(_UID, "c.codigo") is False
    assert llamadas == []
    avisos = [r for r in caplog.records if "P1-PLAN-LOTE-848" in r.getMessage()]
    assert len(avisos) == 1 and "MEALFIT_TOKEN_ENC_KEY" in avisos[0].getMessage()


def test_el_interruptor_apaga_canje_y_revocacion(configurado, red):
    llamadas, _ = red
    configurado.setenv("MEALFIT_APPLE_SIWA_TOKENS", "false")
    assert apple_tokens.canjear_y_guardar(_UID, "c.codigo") is False
    assert apple_tokens.revocar_de_usuario(_UID) == {"revocado": False, "motivo": "sin_configurar"}
    assert llamadas == []


def test_nunca_se_loguean_codigos_ni_tokens(configurado, red, monkeypatch, caplog):
    _, respuestas = red
    respuestas[apple_tokens.APPLE_TOKEN_URL] = (400, {"error": "invalid_grant", "error_description": "c.SECRETO"})
    respuestas[apple_tokens.APPLE_REVOKE_URL] = (500, {"error": "r.SECRETO"})
    import db

    monkeypatch.setattr(db, "execute_sql_query",
                        lambda *a, **k: {"refresh_token_enc": apple_tokens.cifrar("r.SECRETO")})
    with caplog.at_level("DEBUG"):
        apple_tokens.canjear_y_guardar(_UID, "c.SECRETO")
        apple_tokens.revocar_de_usuario(_UID)
    texto = "\n".join(r.getMessage() for r in caplog.records)
    assert "SECRETO" not in texto
    assert "client_secret" not in texto and "BEGIN PRIVATE KEY" not in texto


# ─────────────────────────── el endpoint de login ───────────────────────────

def _cliente_login(monkeypatch, canje):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import apple_auth
    import social_identity
    from routers import auth_session

    monkeypatch.setattr(apple_auth, "apple_signin_enabled", lambda: True)
    monkeypatch.setattr(apple_auth, "verify_apple_identity_token",
                        lambda t, n: {"sub": "001.a.2", "email": "ana@example.com", "email_verified": True,
                                      "is_private_email": False})
    monkeypatch.setattr(social_identity, "resolve_social_user",
                        lambda p, i, name=None: {"user_id": _UID, "email": "ana@example.com", "name": "Ana",
                                                 "created": False, "linked": True})
    monkeypatch.setattr(auth_session, "session_cookies_enabled", lambda: True)
    monkeypatch.setattr(auth_session, "set_session_cookie", lambda resp, uid, iat=None: f"mf-token-{uid}")
    monkeypatch.setattr(auth_session, "derive_form_key", lambda uid: f"fk-{uid}")
    monkeypatch.setattr(auth_session, "ensure_user_profile_exists", lambda *a, **k: None)
    monkeypatch.setattr(apple_tokens, "canjear_y_guardar", canje)
    app = FastAPI()
    app.include_router(auth_session.router)
    app.dependency_overrides[auth_session._APPLE_NATIVE_LIMITER] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


def test_el_login_manda_el_codigo_al_canje_despues_de_emitir_la_sesion(monkeypatch):
    canjes: list = []
    c = _cliente_login(monkeypatch, lambda uid, codigo: canjes.append((uid, codigo)) or True)
    r = c.post("/api/auth/apple/native", json={"identity_token": "jwt", "nonce": "n" * 32,
                                               "authorization_code": "c.codigo"})
    assert r.status_code == 200 and r.json()["token"] == f"mf-token-{_UID}"
    assert canjes == [(_UID, "c.codigo")]


def test_el_login_nunca_falla_por_el_canje(monkeypatch):
    def _revienta(uid, codigo):
        raise RuntimeError("Apple caído")

    c = _cliente_login(monkeypatch, _revienta)
    r = c.post("/api/auth/apple/native", json={"identity_token": "jwt", "nonce": "n" * 32,
                                               "authorization_code": "c.codigo"})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_el_canje_real_sin_configurar_tampoco_rompe_el_login(monkeypatch, sin_env, red):
    llamadas, _ = red
    c = _cliente_login(monkeypatch, apple_tokens.canjear_y_guardar)
    # `_cliente_login` lo sustituyó por sí mismo: el real, sin claves, no llama a Apple.
    r = c.post("/api/auth/apple/native", json={"identity_token": "jwt", "nonce": "n" * 32,
                                               "authorization_code": "c.codigo"})
    assert r.status_code == 200 and llamadas == []


def test_un_binario_viejo_sin_codigo_entra_igual_y_no_hay_canje(monkeypatch):
    canjes: list = []
    c = _cliente_login(monkeypatch, lambda uid, codigo: canjes.append(codigo))
    r = c.post("/api/auth/apple/native", json={"identity_token": "jwt", "nonce": "n" * 32})
    assert r.status_code == 200 and canjes == []


def test_el_canje_va_despues_de_la_sesion_y_en_segundo_plano():
    import inspect

    from routers import auth_session

    src = inspect.getsource(auth_session.apple_native_sign_in)
    assert "background_tasks.add_task(canjear_y_guardar, uid, codigo)" in src
    assert src.index("sesion = set_session_cookie(response, uid)") < src.index("background_tasks.add_task(")
    assert "tooltip-anchor: P1-PLAN-LOTE-848-CANJE" in src


# ─────────────────────────── revocación al borrar la cuenta ───────────────────────────

def test_revocar_manda_el_refresh_descifrado_a_auth_revoke(configurado, red, monkeypatch):
    llamadas, respuestas = red
    respuestas[apple_tokens.APPLE_REVOKE_URL] = (200, None)
    import db

    consultas: list = []

    def _query(q, p=None, **k):
        consultas.append((q, p))
        return {"refresh_token_enc": apple_tokens.cifrar("r.guardado")}

    monkeypatch.setattr(db, "execute_sql_query", _query)
    assert apple_tokens.revocar_de_usuario(_UID) == {"revocado": True, "motivo": "ok"}
    assert consultas[0][1] == (_UID,) and "WHERE user_id = %s" in consultas[0][0]
    (url, datos), = llamadas
    assert url == "https://appleid.apple.com/auth/revoke"
    assert datos["token"] == "r.guardado" and datos["token_type_hint"] == "refresh_token"
    assert datos["client_id"] == "com.bioboros.app"
    assert jwt.get_unverified_header(datos["client_secret"])["kid"] == "ABC123DEFG"


def test_revocar_sin_token_guardado_no_llama_a_apple(configurado, red, monkeypatch):
    llamadas, _ = red
    import db

    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: None)
    assert apple_tokens.revocar_de_usuario(_UID) == {"revocado": False, "motivo": "sin_token"}
    assert llamadas == []


@pytest.mark.parametrize("respuesta,motivo", [((400, {"error": "invalid_client"}), "http_400"),
                                              (TimeoutError("lento"), "error")])
def test_revocar_fallido_no_lanza(configurado, red, monkeypatch, respuesta, motivo):
    _, respuestas = red
    respuestas[apple_tokens.APPLE_REVOKE_URL] = respuesta
    import db

    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: {"refresh_token_enc": apple_tokens.cifrar("r.x")})
    assert apple_tokens.revocar_de_usuario(_UID) == {"revocado": False, "motivo": motivo}


def test_revocar_sin_configurar_es_inerte(sin_env, red, monkeypatch):
    llamadas, _ = red
    import db

    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: pytest.fail("no debía leer la base"))
    assert apple_tokens.revocar_de_usuario(_UID) == {"revocado": False, "motivo": "sin_configurar"}
    assert llamadas == []


def _cuerpo_borrado() -> str:
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    i = app.index("async def api_delete_my_account")
    return app[i:app.index("\n@app.", i)]


def test_el_borrado_revoca_despues_de_paypal_y_antes_de_la_purga():
    cuerpo = _cuerpo_borrado()
    revoca = cuerpo.index("await asyncio.to_thread(revocar_de_usuario, verified_user_id)")
    assert cuerpo.index("await cancel_paypal_subscription_for_user(") < revoca
    assert revoca < cuerpo.index("asyncio.to_thread(delete_account_data")
    assert "tooltip-anchor: P1-PLAN-LOTE-848-REVOCAR" in cuerpo


@pytest.fixture(scope="module")
def app_module():
    import app as _app

    return _app


@pytest.mark.parametrize("revocacion", [
    {"revocado": True, "motivo": "ok"}, {"revocado": False, "motivo": "http_500"},
])
def test_el_borrado_llama_a_revocar_y_sigue_aunque_apple_falle(app_module, monkeypatch, revocacion):
    import db_profiles
    from starlette.responses import Response

    orden: list = []
    monkeypatch.setattr(app_module, "execute_sql_query", lambda *a, **k: None)   # sin suscripción PayPal
    monkeypatch.setattr(app_module, "execute_sql_write", lambda *a, **k: True)
    monkeypatch.setattr(apple_tokens, "revocar_de_usuario", lambda uid: orden.append(("revocar", uid)) or revocacion)

    def _purga(uid, include_profile=True):
        orden.append(("purga", uid))
        return {"user_id": uid, "deleted": {"user_profiles": 1}, "anonymized": {}, "errors": [], "failed_steps": [],
                "profile_deleted": True, "identity_deleted": True, "storage_objects_removed": 0}

    monkeypatch.setattr(db_profiles, "delete_account_data", _purga)
    out = asyncio.run(app_module.api_delete_my_account(
        response=Response(), data={"confirm": "ELIMINAR"}, verified_user_id=_UID))
    assert orden == [("revocar", _UID), ("purga", _UID)]
    assert out["success"] is True and out["identity_deleted"] is True
