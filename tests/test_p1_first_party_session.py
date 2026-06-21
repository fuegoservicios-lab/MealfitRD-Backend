"""[P1-FIRST-PARTY-SESSION · 2026-06-16] Cookie de sesión first-party
`__Host-mf_session` emitida por el backend.

Contrato de seguridad (extiende P0-AUDIT-1 al nuevo path de la cookie):
  - La cookie es un JWT HS256 firmado con MEALFIT_SESSION_SECRET. El path SIEMPRE
    verifica la firma con algoritmo FIJO (sin confusión) + `typ` esperado.
  - Fail-secure: cualquier fallo → None; jamás un `sub` no verificado.
  - Gated en el secreto: sin MEALFIT_SESSION_SECRET (≥32 chars) la cookie ni se
    emite ni se acepta → Bearer-only (cero regresión).
  - get_verified_user_id acepta la cookie SÓLO como fallback del Bearer, con la
    misma garantía de firma verificada.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

import jwt
import pytest


def _run(coro):
    return asyncio.run(coro)


_STRONG_SECRET = "x" * 48  # ≥32 chars


def _import_auth(monkeypatch, *, secret: str = _STRONG_SECRET, verify_return=None):
    """Importa auth con db/neon stubeados; setea el secreto de sesión."""
    db_stub = MagicMock()
    db_stub.get_monthly_api_usage = lambda *a, **k: 0
    db_stub.get_user_profile = lambda *a, **k: None
    db_stub.ensure_user_profile_exists = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "db", db_stub)

    neon_stub = MagicMock()
    neon_stub.verify_neon_jwt = lambda token: verify_return
    monkeypatch.setitem(sys.modules, "neon_auth", neon_stub)

    if "auth" in sys.modules:
        monkeypatch.delitem(sys.modules, "auth", raising=False)
    import auth as _auth  # type: ignore

    monkeypatch.setattr(_auth, "verify_neon_jwt", lambda token: verify_return, raising=False)
    monkeypatch.setattr(_auth, "_SESSION_SECRET", secret, raising=False)
    return _auth


def test_disabled_without_secret(monkeypatch):
    _auth = _import_auth(monkeypatch, secret="")
    assert _auth.session_cookies_enabled() is False
    assert _auth.mint_session_cookie("uid-1") is None
    # Aun una cookie "bien formada" se rechaza si la feature está apagada.
    tok = jwt.encode({"sub": "uid-1", "typ": "mf_session"}, "whatever-secret-32-chars-long!!", algorithm="HS256")
    assert _auth.verify_session_cookie(tok) is None


def test_weak_secret_is_disabled(monkeypatch):
    _auth = _import_auth(monkeypatch, secret="short")  # <32 → deshabilitado
    assert _auth.session_cookies_enabled() is False
    assert _auth.mint_session_cookie("uid-1") is None


def test_mint_and_verify_roundtrip(monkeypatch):
    _auth = _import_auth(monkeypatch)
    assert _auth.session_cookies_enabled() is True
    tok = _auth.mint_session_cookie("uid-42")
    assert isinstance(tok, str)
    assert _auth.verify_session_cookie(tok) == "uid-42"


def test_forged_cookie_wrong_secret_rejected(monkeypatch):
    _auth = _import_auth(monkeypatch)
    # Cookie firmada con OTRO secreto → firma inválida → None.
    forged = jwt.encode({"sub": "victim", "typ": "mf_session"}, "a" * 48, algorithm="HS256")
    assert _auth.verify_session_cookie(forged) is None


def test_wrong_typ_rejected(monkeypatch):
    _auth = _import_auth(monkeypatch)
    # Firmado con el secreto correcto PERO typ distinto → None (no es session cookie).
    wrong = jwt.encode({"sub": "uid-1", "typ": "not_session"}, _STRONG_SECRET, algorithm="HS256")
    assert _auth.verify_session_cookie(wrong) is None


def test_unsigned_alg_none_rejected(monkeypatch):
    _auth = _import_auth(monkeypatch)
    # alg=none (sin firma), construido a mano — decode con algorithms=["HS256"]
    # lo rechaza (algorithm-confusion / downgrade cerrado).
    import base64 as _b64
    import json as _json

    def _b64u(d):
        return _b64.urlsafe_b64encode(_json.dumps(d).encode()).decode().rstrip("=")

    unsigned = f"{_b64u({'alg': 'none', 'typ': 'JWT'})}.{_b64u({'sub': 'uid-1', 'typ': 'mf_session'})}."
    assert _auth.verify_session_cookie(unsigned) is None


def test_get_verified_user_id_cookie_fallback(monkeypatch):
    """Bearer forjado (verify_neon_jwt=None) + cookie VÁLIDA → retorna uid (fallback)."""
    _auth = _import_auth(monkeypatch, verify_return=None)
    good_cookie = _auth.mint_session_cookie("uid-99")
    result = _run(_auth.get_verified_user_id("Bearer forged.token.x", good_cookie))
    assert result == "uid-99"


def test_get_verified_user_id_forged_both_returns_none(monkeypatch):
    """Bearer forjado + cookie forjada (otro secreto) → None (P0-AUDIT-1 preservado)."""
    _auth = _import_auth(monkeypatch, verify_return=None)
    forged_cookie = jwt.encode({"sub": "victim", "typ": "mf_session"}, "b" * 48, algorithm="HS256")
    result = _run(_auth.get_verified_user_id("Bearer forged.token.x", forged_cookie))
    assert result is None


def test_get_verified_user_id_bearer_wins(monkeypatch):
    """Bearer válido tiene prioridad sobre la cookie."""
    _auth = _import_auth(monkeypatch, verify_return={"sub": "bearer-uid", "email": "u@t.local"})
    other_cookie = _auth.mint_session_cookie("cookie-uid")
    result = _run(_auth.get_verified_user_id("Bearer valid.token.x", other_cookie))
    assert result == "bearer-uid"


def test_get_verified_user_id_no_auth_none(monkeypatch):
    _auth = _import_auth(monkeypatch, verify_return=None)
    assert _run(_auth.get_verified_user_id(None, None)) is None


def test_cookie_ignored_when_feature_disabled(monkeypatch):
    """Sin secreto, get_verified_user_id NO acepta la cookie (Bearer-only)."""
    _auth = _import_auth(monkeypatch, secret="", verify_return=None)
    # Cookie firmada con un secreto cualquiera — debe ignorarse (feature off).
    tok = jwt.encode({"sub": "uid-1", "typ": "mf_session"}, "z" * 48, algorithm="HS256")
    assert _run(_auth.get_verified_user_id(None, tok)) is None


def test_absolute_cap(monkeypatch):
    _auth = _import_auth(monkeypatch)
    import time as _t
    recent = _auth.mint_session_cookie("uid-1", iat=int(_t.time()))
    assert _auth.session_cookie_within_absolute_cap(recent) is True
    # iat muy viejo (más que el cap absoluto) → fuera del cap.
    old_iat = int(_t.time()) - (_auth._SESSION_ABS_MAX_S + 10)
    old = _auth.mint_session_cookie("uid-1", iat=old_iat)
    assert _auth.session_cookie_within_absolute_cap(old) is False


def test_iat_preserved_on_reissue(monkeypatch):
    """El re-issue (sliding) preserva el iat original → el cap absoluto no se resetea."""
    _auth = _import_auth(monkeypatch)
    import time as _t
    orig_iat = int(_t.time()) - 1000
    tok1 = _auth.mint_session_cookie("uid-1", iat=orig_iat)
    read_iat = _auth.session_cookie_iat(tok1)
    assert read_iat == orig_iat
    tok2 = _auth.mint_session_cookie("uid-1", iat=read_iat)
    assert _auth.session_cookie_iat(tok2) == orig_iat


# --- Hardening de la revisión de seguridad adversaria (2026-06-16) ---

def test_verify_rejects_expired(monkeypatch):
    """exp en el pasado → rechazado (exp/leeway=0 explícitos)."""
    _auth = _import_auth(monkeypatch)
    import time as _t
    now = int(_t.time())
    expired = jwt.encode(
        {"sub": "uid-1", "typ": "mf_session", "iat": now - 100, "exp": now - 10},
        _STRONG_SECRET, algorithm="HS256",
    )
    assert _auth.verify_session_cookie(expired) is None


def test_verify_rejects_past_absolute_cap(monkeypatch):
    """Cookie con exp aún válido PERO iat fuera del cap absoluto → rechazada EN EL
    PATH DE AUTH (no solo en el re-issue). Cierra el hallazgo HIGH del review."""
    _auth = _import_auth(monkeypatch)
    import time as _t
    now = int(_t.time())
    old_iat = now - (_auth._SESSION_ABS_MAX_S + 100)
    tok = jwt.encode(
        {"sub": "victim", "typ": "mf_session", "iat": old_iat, "exp": now + 1000},
        _STRONG_SECRET, algorithm="HS256",
    )
    assert _auth.verify_session_cookie(tok) is None
    assert _run(_auth.get_verified_user_id(None, tok)) is None


def test_verify_rejects_future_iat(monkeypatch):
    """iat futuro (skew / forja) → rechazado (no resetea el cap con delta negativo)."""
    _auth = _import_auth(monkeypatch)
    import time as _t
    now = int(_t.time())
    tok = jwt.encode(
        {"sub": "uid-1", "typ": "mf_session", "iat": now + 99999, "exp": now + 100000},
        _STRONG_SECRET, algorithm="HS256",
    )
    assert _auth.verify_session_cookie(tok) is None


def test_get_verified_user_id_header_fallback(monkeypatch):
    """Bearer forjado + sin cookie + header X-MF-Session VÁLIDO → uid (path iOS-PWA)."""
    _auth = _import_auth(monkeypatch, verify_return=None)
    good = _auth.mint_session_cookie("uid-h")
    assert _run(_auth.get_verified_user_id("Bearer forged.x", None, good)) == "uid-h"


def test_get_verified_user_id_forged_header_none(monkeypatch):
    """Header firmado con otro secreto → None (firma verificada, fail-secure)."""
    _auth = _import_auth(monkeypatch, verify_return=None)
    import time as _t
    now = int(_t.time())
    forged = jwt.encode(
        {"sub": "victim", "typ": "mf_session", "iat": now, "exp": now + 1000},
        "q" * 48, algorithm="HS256",
    )
    assert _run(_auth.get_verified_user_id(None, None, forged)) is None


def test_set_session_cookie_returns_token(monkeypatch):
    """set_session_cookie devuelve el token (para el body → localStorage)."""
    _auth = _import_auth(monkeypatch)
    class _Resp:
        def set_cookie(self, **kw): self.kw = kw
    r = _Resp()
    tok = _auth.set_session_cookie(r, "uid-x")
    assert isinstance(tok, str) and _auth.verify_session_cookie(tok) == "uid-x"
    assert r.kw["samesite"] == "strict" and r.kw["secure"] and r.kw["httponly"]


def test_secret_rotation_old_accepted(monkeypatch):
    """Durante rotación, una cookie firmada con el secreto ANTERIOR sigue válida;
    una firmada con un secreto desconocido NO."""
    _auth = _import_auth(monkeypatch, secret="new" + "y" * 45)
    old_secret = "old" + "z" * 45
    monkeypatch.setattr(_auth, "_SESSION_SECRET_OLD", old_secret, raising=False)
    import time as _t
    now = int(_t.time())
    rotated = jwt.encode(
        {"sub": "uid-7", "typ": "mf_session", "iat": now, "exp": now + 1000},
        old_secret, algorithm="HS256",
    )
    assert _auth.verify_session_cookie(rotated) == "uid-7"
    unknown = jwt.encode(
        {"sub": "x", "typ": "mf_session", "iat": now, "exp": now + 1000},
        "w" * 48, algorithm="HS256",
    )
    assert _auth.verify_session_cookie(unknown) is None
