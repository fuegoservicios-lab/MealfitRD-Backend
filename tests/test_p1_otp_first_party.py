"""[P1-OTP-FIRST-PARTY · 2026-07-03] Login por código OTP vía sesión first-party.

El flujo previo verificaba el OTP con fetch directo del navegador a Neon Auth: la cookie
de sesión resultante es THIRD-PARTY seteada por XHR → bloqueada en navegadores móviles
(iOS siempre; Chrome phase-out) → tras el reload no había sesión y el usuario rebotaba a
/login ("pongo el código y no entro"). Google OAuth sí funcionaba (cookie de redirect
top-level) — la asimetría que delató la causa.

Cierra: POST /api/auth/email-otp/verify — proxy server-side a Neon (`/sign-in/email-otp`)
que, con OTP válido, emite DIRECTO la sesión first-party (`__Host-mf_session` + token,
mismo mecanismo del PWA iOS). El frontend (verifyEmailOtpFirstParty) adopta el token y el
provider resuelve vía _resolveViaFirstParty sin depender de la cookie de Neon.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import get_verified_user_id, get_neon_bearer_user_id
import routers.auth_session as auth_session

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend" / "src"


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-OTP-FIRST-PARTY" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-03"


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Reemplazo de httpx.AsyncClient — captura el POST a Neon sin red."""
    resp = _FakeResp(401, {})
    last_url = None
    last_json = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **k):
        _FakeAsyncClient.last_url = url
        _FakeAsyncClient.last_json = json
        return _FakeAsyncClient.resp


_ENSURED = []


def _client(monkeypatch, *, neon_resp, cookies_enabled=True, base_url="https://fake.neonauth.test/neondb/auth"):
    import neon_auth
    monkeypatch.setattr(neon_auth, "NEON_AUTH_BASE_URL", base_url)
    _FakeAsyncClient.resp = neon_resp
    monkeypatch.setattr(auth_session.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(auth_session, "session_cookies_enabled", lambda: cookies_enabled)
    monkeypatch.setattr(auth_session, "set_session_cookie", lambda resp, uid, iat=None: f"mf-token-{uid}")
    monkeypatch.setattr(auth_session, "derive_form_key", lambda uid: f"fk-{uid}")
    _ENSURED.clear()
    monkeypatch.setattr(auth_session, "ensure_user_profile_exists",
                        lambda uid, email=None, name=None: _ENSURED.append((uid, email)))
    app = FastAPI()
    app.include_router(auth_session.router)
    app.dependency_overrides[get_neon_bearer_user_id] = lambda: None
    app.dependency_overrides[get_verified_user_id] = lambda: None
    return TestClient(app)


def test_valid_otp_mints_first_party_session(monkeypatch):
    c = _client(monkeypatch, neon_resp=_FakeResp(200, {"token": "neon-tok", "user": {"id": "user-123", "email": "a@b.co"}}))
    r = c.post("/api/auth/email-otp/verify", json={"email": "a@b.co", "otp": "123456"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["user_id"] == "user-123"
    assert body["token"] == "mf-token-user-123", "el token first-party debe viajar en el body (localStorage iOS)"
    assert body["form_key"] == "fk-user-123"
    assert _FakeAsyncClient.last_url.endswith("/sign-in/email-otp")
    assert _FakeAsyncClient.last_json == {"email": "a@b.co", "otp": "123456"}
    # un usuario NUEVO por OTP jamás presenta Bearer → la fila espejo de user_profiles
    # debe garantizarse AQUÍ (sin ella, el INSERT del assessment fallaría por FK).
    assert _ENSURED == [("user-123", "a@b.co")]


def test_invalid_otp_returns_401_without_session(monkeypatch):
    c = _client(monkeypatch, neon_resp=_FakeResp(401, {"message": "invalid otp"}))
    r = c.post("/api/auth/email-otp/verify", json={"email": "a@b.co", "otp": "000000"})
    assert r.status_code == 401


def test_neon_200_without_user_id_fails_secure(monkeypatch):
    c = _client(monkeypatch, neon_resp=_FakeResp(200, {"token": "x", "user": {}}))
    r = c.post("/api/auth/email-otp/verify", json={"email": "a@b.co", "otp": "123456"})
    assert r.status_code == 401, "200 de Neon sin user.id JAMÁS emite sesión (fail-secure)"


def test_garbage_body_rejected_before_neon(monkeypatch):
    c = _client(monkeypatch, neon_resp=_FakeResp(200, {"user": {"id": "u"}}))
    _FakeAsyncClient.last_url = None
    for body in ({}, {"email": "", "otp": "123456"}, {"email": "no-arroba", "otp": "123456"},
                 {"email": "a@b.co", "otp": ""}, {"email": "a@b.co", "otp": "x" * 40}):
        r = c.post("/api/auth/email-otp/verify", json=body)
        assert r.status_code == 401, f"body inválido debe rechazarse: {body}"
    assert _FakeAsyncClient.last_url is None, "input inválido no debe llegar a Neon"


def test_cookies_disabled_returns_503(monkeypatch):
    # sin la feature first-party este login no puede sostenerse → error explícito,
    # no un 200 que no loguea.
    c = _client(monkeypatch, neon_resp=_FakeResp(200, {"user": {"id": "u1"}}), cookies_enabled=False)
    r = c.post("/api/auth/email-otp/verify", json={"email": "a@b.co", "otp": "123456"})
    assert r.status_code == 503


def test_endpoint_has_own_rate_limiter():
    import inspect
    sig = inspect.getsource(auth_session.email_otp_verify)
    assert "_OTP_VERIFY_LIMITER" in sig, "el verify proxya intentos de login — throttle propio obligatorio"


# ════════════════════════════════════════════════════════════════════════════
# Frontend: el flujo OTP usa el endpoint first-party (no la cookie de Neon)
# ════════════════════════════════════════════════════════════════════════════
def test_frontend_wired_to_first_party_verify():
    login = (_FRONT / "pages" / "Login.jsx").read_text(encoding="utf-8")
    assert "verifyEmailOtpFirstParty" in login, "Login debe verificar el OTP vía el backend first-party"
    assert "signInWithEmailOtp(" not in login, "el fetch directo a Neon (cookie third-party) quedó fuera del flujo"
    # el flag de retry OAuth NO aplica al OTP (esperaría ~8s un getSession de Neon inexistente)
    _otp_block = login[login.index("handleCodeSubmit"):login.index("handleGoogle")]
    assert "sessionStorage.setItem('mf_oauth_pending'" not in _otp_block
    fps = (_FRONT / "utils" / "firstPartySession.js").read_text(encoding="utf-8")
    assert "export async function verifyEmailOtpFirstParty" in fps
    assert "/api/auth/email-otp/verify" in fps
