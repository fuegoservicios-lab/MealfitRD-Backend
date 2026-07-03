"""[P1-OAUTH-FIRST-PARTY · 2026-07-03] OAuth de Google sin doble click.

Neon regresa del OAuth con `neon_auth_session_verifier` (query param de UN SOLO USO) que el
SDK canjea client-side en /get-session. En móvil, si esa primera petición se pierde por
timeout, el verifier queda consumido sin sesión y ningún retry resuelve → el usuario debía
pulsar "Continuar con Google" una SEGUNDA vez (verifier fresco). Cierra:

  POST /api/auth/oauth/adopt {verifier} — canje SERVER-SIDE del verifier contra Neon
  (`/get-session?neon_auth_session_verifier=…`) que emite la sesión first-party directo
  (mismo patrón de P1-OTP-FIRST-PARTY). El provider lo invoca ANTES del getSession del SDK;
  si mintó, quita el param y desarma el flag de retry; si falla, deja el param intacto
  (el SDK conserva su canje nativo — fail-open).
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
    if "P1-OAUTH-FIRST-PARTY" in m.group(1):
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
    resp = _FakeResp(401, {})
    last_url = None
    last_params = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None, **k):
        _FakeAsyncClient.last_url = url
        _FakeAsyncClient.last_params = params
        return _FakeAsyncClient.resp


def _client(monkeypatch, *, neon_resp, cookies_enabled=True):
    import neon_auth
    monkeypatch.setattr(neon_auth, "NEON_AUTH_BASE_URL", "https://fake.neonauth.test/neondb/auth")
    _FakeAsyncClient.resp = neon_resp
    monkeypatch.setattr(auth_session.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(auth_session, "session_cookies_enabled", lambda: cookies_enabled)
    monkeypatch.setattr(auth_session, "set_session_cookie", lambda resp, uid, iat=None: f"mf-token-{uid}")
    monkeypatch.setattr(auth_session, "derive_form_key", lambda uid: f"fk-{uid}")
    app = FastAPI()
    app.include_router(auth_session.router)
    app.dependency_overrides[get_neon_bearer_user_id] = lambda: None
    app.dependency_overrides[get_verified_user_id] = lambda: None
    return TestClient(app)


def test_valid_verifier_mints_first_party_session(monkeypatch):
    c = _client(monkeypatch, neon_resp=_FakeResp(200, {"session": {"token": "s"}, "user": {"id": "user-9", "email": "g@x.co"}}))
    r = c.post("/api/auth/oauth/adopt", json={"verifier": "abc123"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["user_id"] == "user-9"
    assert body["token"] == "mf-token-user-9"
    assert _FakeAsyncClient.last_url.endswith("/get-session")
    assert _FakeAsyncClient.last_params == {"neon_auth_session_verifier": "abc123"}


def test_consumed_or_invalid_verifier_returns_401(monkeypatch):
    c = _client(monkeypatch, neon_resp=_FakeResp(401, {}))
    r = c.post("/api/auth/oauth/adopt", json={"verifier": "muerto"})
    assert r.status_code == 401


def test_neon_200_without_user_fails_secure(monkeypatch):
    c = _client(monkeypatch, neon_resp=_FakeResp(200, {"session": None, "user": None}))
    r = c.post("/api/auth/oauth/adopt", json={"verifier": "abc"})
    assert r.status_code == 401


def test_garbage_verifier_rejected_before_neon(monkeypatch):
    c = _client(monkeypatch, neon_resp=_FakeResp(200, {"user": {"id": "u"}}))
    _FakeAsyncClient.last_url = None
    for body in ({}, {"verifier": ""}, {"verifier": "x" * 600}):
        r = c.post("/api/auth/oauth/adopt", json=body)
        assert r.status_code == 401, f"verifier inválido debe rechazarse: {body}"
    assert _FakeAsyncClient.last_url is None, "input inválido no debe llegar a Neon"


def test_endpoint_has_own_rate_limiter():
    import inspect
    sig = inspect.getsource(auth_session.oauth_adopt)
    assert "_OAUTH_ADOPT_LIMITER" in sig, "canjear un verifier es un intento de login — throttle obligatorio"


# ════════════════════════════════════════════════════════════════════════════
# Frontend: el provider canjea el verifier ANTES del getSession del SDK
# ════════════════════════════════════════════════════════════════════════════
def test_frontend_wired_adopt_before_getsession():
    ctx = (_FRONT / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
    assert "adoptOAuthVerifierFirstParty" in ctx
    idx_adopt = ctx.index("_adoptOAuthVerifier()")
    assert "getSessionWithTimeout()" in ctx[idx_adopt:idx_adopt + 120], \
        "el adopt debe encadenarse ANTES de getSessionWithTimeout (el verifier es de un solo uso)"
    # si mintó: quita el param (el SDK no re-pega un verifier consumido) y desarma el retry
    blk = ctx[ctx.index("const _adoptOAuthVerifier"):idx_adopt]
    assert "searchParams.delete('neon_auth_session_verifier')" in blk
    assert "removeItem('mf_oauth_pending')" in blk
    fps = (_FRONT / "utils" / "firstPartySession.js").read_text(encoding="utf-8")
    assert "export async function adoptOAuthVerifierFirstParty" in fps
    assert "/api/auth/oauth/adopt" in fps


def test_frontend_verifier_stashed_before_react_mounts():
    """El <Navigate to="/dashboard" replace/> de '/' descarta el query en el PRIMER ciclo de
    render (efectos hijo corren antes que el del provider) → el verifier debe capturarse en
    main.jsx ANTES de montar React, y el provider debe leer el stash como fallback."""
    main = (_FRONT / "main.jsx").read_text(encoding="utf-8")
    assert "neon_auth_session_verifier" in main, "main.jsx debe capturar el verifier pre-mount"
    assert "sessionStorage.setItem('mf_oauth_verifier'" in main
    ctx = (_FRONT / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
    blk = ctx[ctx.index("const _adoptOAuthVerifier"):ctx.index("_adoptOAuthVerifier()")]
    assert "sessionStorage.getItem('mf_oauth_verifier')" in blk, "el provider debe leer el stash"
    assert "sessionStorage.removeItem('mf_oauth_verifier')" in blk, \
        "el stash es single-use: consumirlo SIEMPRE (aunque el adopt falle)"
