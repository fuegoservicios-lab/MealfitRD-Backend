# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-90 · 2026-09-17] «Agrego el correo, verifico el código y no me inicia sesión» (dueño, PWA de iOS).

Medido en producción (journal + nginx, 20:33-20:37 UTC): dos `POST /api/auth/email-otp/verify` → 200 con la sesión emitida
(uid=61a13831…) y, tras la recarga, NINGUNA llamada a `/api/auth/me`: la app volvió a /login en menos de un segundo. El
arranque del cliente solo preguntaba si encontraba el token en localStorage (NO-401-NOISE); ese día no estaba, y la cookie
—válida, recién puesta— no sirvió de nada. El último OTP que funcionó (11-sep) sí llamó a `/me` tras verificar.

Dos arreglos, los dos del lado que SÍ se puede comprobar:
  1. un marcador `__Host-mf_has_session=1` (legible por JS, sin secreto) que viaja CON la cookie de sesión: el cliente
     pregunta cuando lo ve, sin depender de localStorage, y el visitante anónimo sigue sin generar un 401;
  2. una cookie válida de una identidad BORRADA ya no decide sola: si además llega `X-MF-Session`, se prueba. El mismo
     incidente empezó con la cookie de la cuenta purgada del dueño (sub=f47126cb…) tumbando el arranque."""
from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import jwt
import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"
_FIRMA_DE_PRUEBA = "k" * 48  # ≥32 caracteres; valor de relleno, solo existe dentro de este test


def _run(coro):
    return asyncio.run(coro)


def _auth_con_firma(monkeypatch, muertos=()):
    """`auth` con db/neon de mentira, la firma de relleno y un guard que da por borrados los uid de `muertos`."""
    db_stub = MagicMock()
    db_stub.ensure_user_profile_exists = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "db", db_stub)
    neon_stub = MagicMock()
    neon_stub.verify_neon_jwt = lambda token: None
    monkeypatch.setitem(sys.modules, "neon_auth", neon_stub)
    monkeypatch.delitem(sys.modules, "auth", raising=False)
    import auth as _auth  # type: ignore
    monkeypatch.setattr(_auth, "_SESSION_SECRET", _FIRMA_DE_PRUEBA, raising=False)
    monkeypatch.setattr(_auth, "_uid_si_la_identidad_vive", lambda uid, via: None if uid in muertos else uid)
    return _auth


class _Resp:
    def __init__(self):
        self.puestas, self.borradas = [], []

    def set_cookie(self, **kw):
        self.puestas.append(kw)

    def delete_cookie(self, **kw):
        self.borradas.append(kw)


# ---------------------------------------------------------------- 1) el marcador viaja con la cookie
def test_emitir_la_sesion_emite_tambien_el_marcador(monkeypatch):
    _auth = _auth_con_firma(monkeypatch)
    r = _Resp()
    tok = _auth.set_session_cookie(r, "uid-1")
    assert tok and _auth.verify_session_cookie(tok) == "uid-1"
    marcador, sesion = r.puestas
    assert marcador["key"] == _auth.SESSION_MARKER_COOKIE_NAME == "__Host-mf_has_session"
    # sin secreto y legible por JS: es una SEÑAL, no una credencial
    assert marcador["value"] == "1" and marcador["httponly"] is False
    # misma vida y mismo alcance que la sesión; `__Host-` exige Secure + Path=/ sin Domain
    assert marcador["max_age"] == sesion["max_age"] and marcador["path"] == "/" and marcador["secure"] is True
    assert marcador["samesite"] == "strict" and "domain" not in marcador
    # la de sesión va la ÚLTIMA y sigue siendo HttpOnly
    assert sesion["key"] == _auth.SESSION_COOKIE_NAME and sesion["httponly"] is True and sesion["value"] == tok


def test_borrar_la_sesion_borra_tambien_el_marcador(monkeypatch):
    _auth = _auth_con_firma(monkeypatch)
    r = _Resp()
    _auth.clear_session_cookie(r)
    assert {b["key"] for b in r.borradas} == {_auth.SESSION_COOKIE_NAME, _auth.SESSION_MARKER_COOKIE_NAME}
    assert all(b["path"] == "/" and b["secure"] is True for b in r.borradas)


# ---------------------------------------------------------------- 2) la cookie de una cuenta borrada no decide sola
def test_la_cookie_de_una_cuenta_borrada_ya_no_tapa_un_header_valido(monkeypatch):
    _auth = _auth_con_firma(monkeypatch, muertos={"uid-purgado"})
    cookie_vieja = _auth.mint_session_cookie("uid-purgado")
    header_nuevo = _auth.mint_session_cookie("uid-vivo")
    assert _run(_auth.get_verified_user_id(None, cookie_vieja, header_nuevo)) == "uid-vivo"


def test_pero_sigue_sin_conceder_nada(monkeypatch):
    _auth = _auth_con_firma(monkeypatch, muertos={"uid-purgado"})
    cookie_vieja = _auth.mint_session_cookie("uid-purgado")
    # sola → nadie
    assert _run(_auth.get_verified_user_id(None, cookie_vieja, None)) is None
    # con un header de la MISMA cuenta borrada → nadie (el guard también cierra esa puerta)
    assert _run(_auth.get_verified_user_id(None, cookie_vieja, cookie_vieja)) is None
    # con un header falsificado (otra firma) → nadie
    now = int(time.time())
    falso = jwt.encode({"sub": "victima", "typ": "mf_session", "iat": now, "exp": now + 900}, "q" * 48, algorithm="HS256")
    assert _run(_auth.get_verified_user_id(None, cookie_vieja, falso)) is None
    # y una cookie VIVA sigue mandando sobre el header (orden intacto)
    viva = _auth.mint_session_cookie("uid-a")
    otro = _auth.mint_session_cookie("uid-b")
    assert _run(_auth.get_verified_user_id(None, viva, otro)) == "uid-a"


# ---------------------------------------------------------------- 3) /me: el iat sale de la credencial que ES del uid
def _cliente_me(monkeypatch, uid):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    _auth = _auth_con_firma(monkeypatch)
    monkeypatch.delitem(sys.modules, "routers.auth_session", raising=False)
    import routers.auth_session as rs  # type: ignore
    app = FastAPI()
    app.include_router(rs.router)
    app.dependency_overrides[_auth.get_verified_user_id] = lambda: uid
    return _auth, TestClient(app, base_url="https://testserver")


def test_me_no_hereda_la_antiguedad_de_una_cookie_ajena(monkeypatch):
    _auth, c = _cliente_me(monkeypatch, "uid-vivo")
    ahora = int(time.time())
    cookie_ajena = _auth.mint_session_cookie("uid-otro", iat=ahora - 5000)
    header_propio = _auth.mint_session_cookie("uid-vivo", iat=ahora - 1000)
    r = c.get("/api/auth/me", headers={"X-MF-Session": header_propio, "Cookie": f"{_auth.SESSION_COOKIE_NAME}={cookie_ajena}"})
    assert r.status_code == 200
    nuevo = r.json()["token"]
    assert _auth.verify_session_cookie(nuevo) == "uid-vivo"
    assert _auth.session_cookie_iat(nuevo) == ahora - 1000, "el iat debe salir del header propio, no de la cookie ajena"
    puestas = r.headers.get_list("set-cookie")
    assert any(h.startswith("__Host-mf_has_session=1") for h in puestas)
    assert any(h.startswith("__Host-mf_session=") and "HttpOnly" in h for h in puestas)


def test_me_con_credencial_ajena_y_uid_por_bearer_reemite_fresca(monkeypatch):
    """El uid llegó por Bearer y la cookie es de otra cuenta: se re-emite (como haría POST /session) para que deje de viajar."""
    _auth, c = _cliente_me(monkeypatch, "uid-bearer")
    cookie_ajena = _auth.mint_session_cookie("uid-otro", iat=int(time.time()) - 5000)
    r = c.get("/api/auth/me", headers={"Cookie": f"{_auth.SESSION_COOKIE_NAME}={cookie_ajena}"})
    assert r.status_code == 200 and _auth.verify_session_cookie(r.json()["token"]) == "uid-bearer"


def test_me_401_borra_las_dos_cookies(monkeypatch):
    _auth, c = _cliente_me(monkeypatch, None)
    r = c.get("/api/auth/me")
    assert r.status_code == 401
    borradas = r.headers.get_list("set-cookie")
    assert any(h.startswith("__Host-mf_has_session=") and "Max-Age=0" in h for h in borradas)
    assert any(h.startswith("__Host-mf_session=") and "Max-Age=0" in h for h in borradas)


# ---------------------------------------------------------------- 4) el cliente (ancla cross-repo)
def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def test_el_cliente_pregunta_con_el_marcador_y_lo_apaga_al_salir():
    js = _front("src/utils/firstPartySession.js")
    assert "const MF_SESSION_MARKER = '__Host-mf_has_session=1';" in js
    assert "if (!tok && !hasSessionMarker()) return null;" in js
    # el atajo viejo (solo token) no puede volver: era la única puerta y falló
    assert "if (!tok) return null;" not in js
    plano = re.sub(r"\s+", " ", js)
    assert "if (res && res.status === 401) { clearStoredMfSession(); clearSessionMarker(); }" in plano
    salir = js[js.index("export async function logoutFirstPartySession()"):]
    assert salir.index("clearSessionMarker();") < salir.index("/api/auth/logout"), "el marcador cae ANTES del POST: sin red también cierra"
    assert "document.cookie = '__Host-mf_has_session=; Max-Age=0; Path=/; Secure; SameSite=Strict';" in js


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 90
    assert "P1-PLAN-LOTE-90" in (_BACKEND / "docs" / "sesion_first_party_marcador.md").read_text(encoding="utf-8")
