"""[P1-OAUTH-FIRST-PARTY · 2026-07-03 · REESCRITO P1-OAUTH-CHALLENGE-COOKIE 2026-08-10]
El retorno del OAuth de Google termina en sesión al PRIMER intento.

QUÉ CAMBIÓ, Y POR QUÉ ESTE FICHERO ES UN AVISO ADEMÁS DE UN TEST
────────────────────────────────────────────────────────────────
La versión anterior probaba que `POST /oauth/adopt {verifier}` canjeaba el
verifier contra Neon SERVER-SIDE. El test pasaba en verde. Producción, en
paralelo, acumulaba **24 canjes y 0 éxitos** durante un mes, todos HTTP 400.

El test pasaba porque `_FakeAsyncClient` devolvía 200 a la llamada a Neon: es
decir, SIMULABA PRECISAMENTE LA PARTE QUE ERA IMPOSIBLE. El verifier no es
autosuficiente — va emparejado a una cookie `__Secure-neon-auth.session_challange`
que Neon deja en el NAVEGADOR y en SU dominio, y que un proceso de servidor
nunca tendrá. Neon respondía siempre `SESSION_CHALLENGE_COOKIE_NOT_FOUND`.

    LECCIÓN: un mock que responde lo que el sistema real nunca responde no
    prueba el sistema, prueba el mock. Cuando el doble del mundo exterior
    decide el resultado del test, la pregunta obligada es «¿el de verdad
    contesta así?» — aquí bastaba una petición para descubrir que no.

Contrato NUEVO: el canje ocurre en el navegador (que tiene la cookie) y este
endpoint recibe el Bearer EdDSA resultante, lo valida contra el JWKS y emite
`__Host-mf_session`. Sigue decidiendo el servidor; lo que cambia es qué se le
manda.
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

_ENSURED = []


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-OAUTH" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-03"


def _client(monkeypatch, *, uid, cookies_enabled=True):
    monkeypatch.setattr(auth_session, "session_cookies_enabled", lambda: cookies_enabled)
    monkeypatch.setattr(auth_session, "set_session_cookie", lambda resp, uid_, iat=None: f"mf-token-{uid_}")
    monkeypatch.setattr(auth_session, "derive_form_key", lambda uid_: f"fk-{uid_}")
    _ENSURED.clear()
    monkeypatch.setattr(auth_session, "ensure_user_profile_exists",
                        lambda uid_, email=None, name=None: _ENSURED.append((uid_, email)))
    app = FastAPI()
    app.include_router(auth_session.router)
    app.dependency_overrides[get_neon_bearer_user_id] = lambda: uid
    app.dependency_overrides[get_verified_user_id] = lambda: uid
    return TestClient(app)


def test_bearer_valido_emite_sesion_first_party(monkeypatch):
    """El camino feliz nuevo: llega el Bearer que salió del canje en el navegador."""
    c = _client(monkeypatch, uid="user-9")
    r = c.post("/api/auth/oauth/adopt", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["user_id"] == "user-9"
    assert body["token"] == "mf-token-user-9"
    # El primer login por Google puede resolverse SOLO por aquí → la fila espejo
    # de user_profiles se garantiza en este punto.
    assert _ENSURED == [("user-9", None)]


def test_sin_bearer_es_401(monkeypatch):
    c = _client(monkeypatch, uid=None)
    assert c.post("/api/auth/oauth/adopt", json={}).status_code == 401


def test_un_verifier_ya_no_abre_sesion(monkeypatch):
    """EL GUARD DE LA REGRESIÓN. Si alguien vuelve a mandar el verifier esperando
    que el servidor lo canjee, no puede recibir sesión: ese canje es imposible
    server-side y volvería a fallar en silencio 24 veces seguidas."""
    c = _client(monkeypatch, uid=None)
    r = c.post("/api/auth/oauth/adopt", json={"verifier": "abc123"})
    assert r.status_code == 401, "un verifier sin Bearer NO puede emitir sesión"


def test_sin_cookies_de_sesion_falla_ruidoso(monkeypatch):
    c = _client(monkeypatch, uid="user-9", cookies_enabled=False)
    assert c.post("/api/auth/oauth/adopt", json={}).status_code == 503


def test_endpoint_conserva_su_rate_limiter():
    import inspect
    sig = inspect.getsource(auth_session.oauth_adopt)
    assert "_OAUTH_ADOPT_LIMITER" in sig, "adoptar un retorno de OAuth es un intento de login — throttle obligatorio"


def test_el_backend_ya_no_pide_get_session_con_verifier():
    """Contrapartida server-side del guard de arriba: la llamada que Neon rechaza
    siempre (`/get-session?neon_auth_session_verifier=`) no debe volver a este
    endpoint. Se comprueba sobre el CUERPO de la función, no sobre el fichero:
    el flujo OTP de al lado sí usa httpx legítimamente."""
    import inspect
    cuerpo = inspect.getsource(auth_session.oauth_adopt)
    assert "neon_auth_session_verifier" not in cuerpo.split('"""')[-1], \
        "el canje del verifier NO puede hacerse server-side: falta la cookie de challenge"


# ════════════════════════════════════════════════════════════════════════════
# Frontend: el canje vive donde vive la cookie
# ════════════════════════════════════════════════════════════════════════════
def test_frontend_canjea_en_el_navegador_con_credenciales():
    fps = (_FRONT / "utils" / "firstPartySession.js").read_text(encoding="utf-8")
    assert "export async function adoptOAuthVerifierFirstParty" in fps
    assert "get-session?neon_auth_session_verifier=" in fps, \
        "el canje debe hacerse desde el navegador contra Neon"
    assert "credentials: 'include'" in fps, \
        "sin credentials la cookie de challenge no viaja y Neon devuelve 400"
    assert "/api/auth/oauth/adopt" in fps, \
        "el token resultante lo valida NUESTRO backend (no se confía en el cliente)"


def test_frontend_wired_adopt_before_getsession():
    ctx = (_FRONT / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
    assert "adoptOAuthVerifierFirstParty" in ctx
    idx_adopt = ctx.index("_adoptOAuthVerifier()")
    assert "getSessionWithTimeout()" in ctx[idx_adopt:idx_adopt + 120], \
        "el adopt debe encadenarse ANTES de getSessionWithTimeout (el verifier es de un solo uso)"
    blk = ctx[ctx.index("const _adoptOAuthVerifier"):idx_adopt]
    assert "searchParams.delete('neon_auth_session_verifier')" in blk
    assert "removeItem('mf_oauth_pending')" in blk


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
