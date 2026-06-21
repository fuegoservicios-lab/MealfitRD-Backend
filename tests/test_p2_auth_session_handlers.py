"""[P2-AUTH-TEST-COVERAGE · 2026-06-18] Tests de los handlers HTTP de
routers/auth_session.py vía TestClient. Antes ningún test montaba este router
→ los contratos (401 sin auth, 200 degradado sin secreto, /logout sin auth,
rate-limit presente) no estaban verificados.

Anchor: P2-AUTH-TEST-COVERAGE.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import get_verified_user_id, get_neon_bearer_user_id
import routers.auth_session as auth_session


def _client(monkeypatch, *, uid_bearer=None, uid_verified=None, cookies_enabled=True):
    app = FastAPI()
    app.include_router(auth_session.router)
    # El limiter _AUTH_SESSION_LIMITER depende internamente de get_verified_user_id,
    # así que este override también cubre el bucketing del rate-limit.
    app.dependency_overrides[get_neon_bearer_user_id] = lambda: uid_bearer
    app.dependency_overrides[get_verified_user_id] = lambda: uid_verified
    monkeypatch.setattr(auth_session, "session_cookies_enabled", lambda: cookies_enabled)
    return TestClient(app)


def test_session_disabled_returns_200_no_cookie(monkeypatch):
    """Sin secreto (feature apagada) → 200 con session_cookie:false (degradación
    silenciosa a Bearer-only, el frontend sigue funcionando)."""
    c = _client(monkeypatch, uid_bearer="u1", cookies_enabled=False)
    r = c.post("/api/auth/session")
    assert r.status_code == 200
    assert r.json()["session_cookie"] is False


def test_session_no_bearer_returns_401(monkeypatch):
    """Feature activa pero sin Bearer Neon válido → 401."""
    c = _client(monkeypatch, uid_bearer=None, cookies_enabled=True)
    r = c.post("/api/auth/session")
    assert r.status_code == 401


def test_me_no_auth_returns_401(monkeypatch):
    """GET /me sin sesión válida (ni Bearer ni cookie) → 401."""
    c = _client(monkeypatch, uid_verified=None)
    r = c.get("/api/auth/me")
    assert r.status_code == 401


def test_logout_returns_200_without_auth(monkeypatch):
    """POST /logout es intencionalmente sin auth (borrar la propia cookie siempre
    es seguro) y debe responder 200 aunque no haya sesión."""
    c = _client(monkeypatch)
    r = c.post("/api/auth/logout")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_session_endpoint_has_rate_limiter_dep():
    """El handler POST /session debe declarar el rate-limiter (defensa DoS sobre
    verify_neon_jwt). Verifica la firma del handler real (no el override)."""
    import inspect
    sig = inspect.getsource(auth_session.create_session)
    assert "_AUTH_SESSION_LIMITER" in sig, (
        "POST /api/auth/session perdió su Depends(_AUTH_SESSION_LIMITER) — sin "
        "throttle, martillear el endpoint satura el threadpool de verify_neon_jwt."
    )
