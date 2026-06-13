"""[P0-AUDIT-1 · 2026-05-12 · migrado a Neon Auth P1-NEON-AUTH-MIGRATION 2026-06-13]
`backend/auth.py` NO debe aceptar JWTs forjados.

Pre-fix histórico (Supabase era), `get_verified_user_id` tenía un "fallback"
que decodificaba el payload del JWT con `base64.urlsafe_b64decode(...)` y
retornaba `payload["sub"]` SIN verificar la firma → account takeover universal.

Vector de ataque (idéntico bajo cualquier proveedor de auth):
    1. Atacante construye JWT con header/payload válidos + firma arbitraria.
    2. Setea `{"sub": "<UUID-de-victima>"}` en el payload.
    3. Envía `Authorization: Bearer <forged-token>` a cualquier endpoint.
    4. Si el backend decodifica sin verificar firma → `verified_user_id` =
       `victim_id` → todas las defensas IDOR (I2/I3/I6) `AND user_id = %s`
       PASAN porque el id coincide con la víctima.
    5. Atacante lee/muta plans, pantry, perfil, chat history de cualquier user.

Por qué el rol de DB no protege:
    El backend conecta a Neon con un rol que bypassa RLS. Esta función es la
    ÚNICA capa de auth. Por eso el bypass sería catastrófico, no degradable.

Fix vigente (Neon Auth):
    El único validador es `neon_auth.verify_neon_jwt(token)` — valida la firma
    EdDSA (Ed25519) contra el JWKS público de Neon Auth, con el algoritmo FIJO
    `["EdDSA"]` (sin algorithm-confusion ni `none`), más `iss`/`aud`/`exp`.
    Fail-secure: cualquier error → `None`, NUNCA un `sub` no verificado.

Lo que este test enforza:
    A) Parser-based: `auth.py` NO contiene `base64.urlsafe_b64decode(...)`
       seguido de `return payload.get("sub")` sin verificación intermedia.
    B) `verify_neon_jwt(` DEBE estar invocado en `auth.py` — sin ese call no
       hay validación real posible.
    C) `auth.py` NO importa `base64` (el legacy lo usaba solo para el fallback
       inseguro).
    D) Ancla `P0-AUDIT-1` presente (contexto del fix preservado).
    E) Funcional: cuando `verify_neon_jwt` rechaza un token (retorna None),
       `get_verified_user_id` NO retorna el `sub` forjado.
    F) Sin header → None.
    G) Token válido → sub.

Tooltip-anchor sync: `P0-AUDIT-1-AUTH-VERIFY` + `P1-NEON-AUTH-VERIFY` en auth.py.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _run_async(coro):
    return asyncio.run(coro)


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AUTH_PY = _BACKEND_ROOT / "auth.py"


@pytest.fixture(scope="module")
def auth_src() -> str:
    return _AUTH_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A) Parser-based: el patrón inseguro no puede reintroducirse.
# ---------------------------------------------------------------------------

def test_a_no_base64_fallback_returning_unverified_sub(auth_src: str):
    """`base64.urlsafe_b64decode(...)` seguido de `return payload.get("sub")`
    sin un `verify_neon_jwt` intermedio."""
    pattern = re.compile(
        r"base64\.urlsafe_b64decode\([^)]*\)"
        r"(?:(?!verify_neon_jwt).)*?"
        r"return\s+payload\.get\(\s*[\"']sub[\"']\s*\)",
        re.DOTALL,
    )
    match = pattern.search(auth_src)
    assert match is None, (
        "P0-AUDIT-1 regresión: auth.py contiene el patrón inseguro "
        "`base64.urlsafe_b64decode(...)` → `return payload.get('sub')` SIN "
        "verify_neon_jwt en medio. Eso es un bypass de autenticación.\n\n"
        f"Match (300 chars):\n{match.group(0)[:300] if match else ''}..."
    )


def test_b_neon_auth_verify_is_invoked(auth_src: str):
    """Sin invocar `verify_neon_jwt`, NO hay validador real de firma.
    Acepta la forma directa `verify_neon_jwt(` o la indirecta
    `to_thread(verify_neon_jwt` (despacho a thread del fetch JWKS)."""
    invoked = (
        "verify_neon_jwt(" in auth_src
        or re.search(r"to_thread\(\s*verify_neon_jwt\b", auth_src) is not None
    )
    assert invoked, (
        "P0-AUDIT-1/P1-NEON-AUTH regresión: auth.py NO invoca "
        "`verify_neon_jwt`. Es el único validador real (verifica la firma "
        "EdDSA del JWT contra el JWKS de Neon Auth). Sin él, no podemos "
        "confiar en ningún token."
    )
    # Defensa-en-profundidad: el viejo validador Supabase NO debe reaparecer.
    assert "supabase.auth.get_user(" not in auth_src, (
        "auth.py invoca `supabase.auth.get_user(` — Supabase Auth fue "
        "reemplazado por Neon Auth (P1-NEON-AUTH-MIGRATION). El validador "
        "debe ser verify_neon_jwt."
    )


def test_c_no_unused_base64_import(auth_src: str):
    """auth.py no importa `base64` (el legacy lo usaba solo para el bypass)."""
    pattern_top_level = re.compile(r"^\s*(?:import\s+base64|from\s+base64\s+import)", re.MULTILINE)
    pattern_local = re.compile(r"^\s+(?:import\s+base64|from\s+base64\s+import)", re.MULTILINE)
    assert not pattern_top_level.search(auth_src) and not pattern_local.search(auth_src), (
        "P0-AUDIT-1 defense-in-depth: auth.py importa `base64` sin razón. "
        "Si necesitas base64 legítimo, documentar y eliminar este test."
    )


def test_d_anchor_present_in_source(auth_src: str):
    """Anclas `P0-AUDIT-1` y `P1-NEON-AUTH` presentes."""
    assert "P0-AUDIT-1" in auth_src, (
        "auth.py perdió la referencia P0-AUDIT-1 (contexto del fix)."
    )
    assert "P1-NEON-AUTH" in auth_src, (
        "auth.py perdió la referencia P1-NEON-AUTH (contexto del proveedor)."
    )


# ---------------------------------------------------------------------------
# Funcional: simulamos verify_neon_jwt aceptando/rechazando.
# ---------------------------------------------------------------------------

def _build_forged_jwt(forged_sub: str) -> str:
    """JWT con header/payload válidos pero firma basura."""
    header_b64 = base64.urlsafe_b64encode(
        json.dumps({"alg": "EdDSA", "typ": "JWT", "kid": "fake"}).encode()
    ).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(
        json.dumps({"sub": forged_sub, "aud": "authenticated", "exp": 9999999999}).encode()
    ).decode().rstrip("=")
    return f"{header_b64}.{payload_b64}.invalid_signature_attacker_basura"


def _import_auth_with_stubs(monkeypatch, verify_return):
    """Importa auth con `db` y `neon_auth` stubeados. `verify_return` es lo que
    `verify_neon_jwt` retorna (None = rechazado, dict = aceptado)."""
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
    # Hard-set por si auth ya capturó la referencia.
    monkeypatch.setattr(_auth, "verify_neon_jwt", lambda token: verify_return, raising=False)
    return _auth


def test_e_forged_jwt_does_not_return_unverified_sub(monkeypatch):
    """Si `verify_neon_jwt` rechaza (None), get_verified_user_id NO retorna el
    sub forjado — debe dar None (o levantar 401/403)."""
    _auth = _import_auth_with_stubs(monkeypatch, verify_return=None)

    forged_sub = "00000000-0000-0000-0000-000000000099"
    forged_token = _build_forged_jwt(forged_sub)

    from fastapi import HTTPException
    raised = None
    result = None
    try:
        result = _run_async(_auth.get_verified_user_id(f"Bearer {forged_token}"))
    except HTTPException as e:
        raised = e

    if raised is not None:
        assert raised.status_code in (401, 403)
        return
    assert result is None, (
        f"P0-AUDIT-1 violation: retornó '{result}' para un token con firma "
        f"INVÁLIDA (sub forjado '{forged_sub}'). Abre IDOR universal."
    )
    assert result != forged_sub


def test_f_missing_authorization_returns_none(monkeypatch):
    """Sin header Authorization → None (no raise)."""
    _auth = _import_auth_with_stubs(monkeypatch, verify_return=None)
    assert _run_async(_auth.get_verified_user_id(None)) is None
    assert _run_async(_auth.get_verified_user_id("")) is None
    assert _run_async(_auth.get_verified_user_id("NotBearer abc.def.ghi")) is None
    assert _run_async(_auth.get_verified_user_id("Bearer ")) is None


def test_g_valid_token_returns_user_id(monkeypatch):
    """Token válido (verify_neon_jwt retorna payload con sub) → retorna sub."""
    valid_sub = "11111111-2222-3333-4444-555555555555"
    _auth = _import_auth_with_stubs(
        monkeypatch,
        verify_return={"sub": valid_sub, "email": "u@test.local", "name": "U"},
    )
    result = _run_async(_auth.get_verified_user_id("Bearer valid.token.signature"))
    assert result == valid_sub
