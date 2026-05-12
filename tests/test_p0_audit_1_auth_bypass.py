"""[P0-AUDIT-1 · 2026-05-12] `backend/auth.py` NO debe aceptar JWTs forjados.

Pre-fix, `get_verified_user_id` tenía un "fallback" que decodificaba el
payload del JWT con `base64.urlsafe_b64decode(...)` y retornaba
`payload["sub"]` SIN verificar la firma:

    try:
        parts = token.split('.')
        if len(parts) >= 2:
            payload_b64 = parts[1]
            payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64).decode('utf-8')
            payload = json.loads(payload_json)
            return payload.get("sub")    # ← bypass: retorna sub sin verify
    except Exception:
        ...

Vector de ataque:
    1. Atacante construye JWT con header/payload válidos + firma arbitraria.
    2. Setea `{"sub": "<UUID-de-victima>"}` en el payload.
    3. Envía `Authorization: Bearer <forged-token>` a cualquier endpoint.
    4. `get_verified_user_id` decodifica el payload, retorna el `sub` forjado
       → `verified_user_id` = `victim_id`.
    5. Todas las defensas IDOR documentadas (invariantes I2/I3/I6) `AND user_id
       = %s` PASAN porque `verified_user_id` coincide con la víctima.
    6. Atacante lee/muta plans, pantry, perfil, chat history de cualquier user.

Por qué SERVICE_ROLE no protege:
    Backend usa `SUPABASE_KEY = SERVICE_ROLE` (`.env.example:22` lo confirma —
    "backend usa SERVICE_ROLE, bypassa RLS"). RLS está bypassed. Esta función
    es la ÚNICA capa de auth.

Fix:
    Eliminar el fallback base64. El único validador es `supabase.auth.get_user(
    token)` que valida la firma server-side llamando a Supabase. Fail-secure:
    cualquier error → None o HTTPException 403, NUNCA retornar `sub` no
    verificado.

Lo que este test enforza:
    A) Parser-based: `auth.py` NO contiene `base64.urlsafe_b64decode(...)`
       seguido de `return payload.get("sub")` (en cualquier orden de líneas,
       multiline DOTALL).
    B) `supabase.auth.get_user(` DEBE estar invocado en `auth.py` — sin ese
       call no hay validación real posible.
    C) Funcional: cuando `supabase.auth.get_user` rechaza un token (firma
       inválida), `get_verified_user_id` NO retorna el `sub` forjado. Debe
       retornar `None` o levantar `HTTPException 401/403`.
    D) Defensa adicional: `auth.py` NO debe importar `base64` (el legacy lo
       importaba solo para el fallback inseguro; sin él no hay razón para
       que `base64` aparezca en este módulo).

Tooltip-anchor sync: `P0-AUDIT-1-AUTH-VERIFY` aparece en `auth.py` para que
un refactor cosmético no borre la convención sin pasar por este test.
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
    """[P2-AUTH-ASYNC-SLEEP · 2026-05-12] `get_verified_user_id` se
    convirtió a `async def`. Tests funcionales que la invocan deben
    ejecutar el coroutine. Usamos `asyncio.run` per-call para no
    contaminar event loops entre tests.
    """
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
    """`base64.urlsafe_b64decode(...)` seguido (en cualquier punto del módulo)
    de `return payload.get("sub")` sin signature verify intermedio.

    Pattern multiline (DOTALL): la lazy negative-lookahead asegura que NO haya
    `supabase.auth.get_user` entre el decode y el return — si lo hay, el verify
    SÍ ocurrió y el patrón no aplica.
    """
    pattern = re.compile(
        r"base64\.urlsafe_b64decode\([^)]*\)"            # decode call
        r"(?:(?!supabase\.auth\.get_user).)*?"           # SIN verify intermedio
        r"return\s+payload\.get\(\s*[\"']sub[\"']\s*\)",  # retorna sub sin verify
        re.DOTALL,
    )
    match = pattern.search(auth_src)
    assert match is None, (
        "P0-AUDIT-1 regresión: auth.py contiene el patrón inseguro "
        "`base64.urlsafe_b64decode(...)` seguido de "
        "`return payload.get('sub')` SIN supabase.auth.get_user en medio. "
        "Esto es un bypass de autenticación: un atacante construye un JWT "
        "con `sub` = victim_id (header/payload válidos + firma random) y "
        "el backend lo acepta como verified_user_id. SERVICE_ROLE bypassea "
        "RLS, así que esta función es la ÚNICA defensa. Eliminar el "
        "fallback — el único validador debe ser supabase.auth.get_user("
        "token).\n\n"
        f"Match encontrado (primeros 300 chars):\n{match.group(0)[:300] if match else ''}..."
    )


def test_b_supabase_auth_get_user_is_invoked(auth_src: str):
    """Sin `supabase.auth.get_user(token)`, NO hay validador real.
    Asegura que el path correcto sigue presente.
    """
    assert "supabase.auth.get_user(" in auth_src, (
        "P0-AUDIT-1 regresión: auth.py NO invoca `supabase.auth.get_user()`. "
        "Este es el único validador real (verifica la firma del JWT "
        "server-side contra Supabase). Sin él, no podemos confiar en ningún "
        "token. Restaurar el llamado dentro de get_verified_user_id."
    )


def test_c_no_unused_base64_import(auth_src: str):
    """El legacy importaba `base64` SOLO para el fallback inseguro. Una vez
    eliminado el fallback, no hay razón para que `base64` aparezca importado
    en este módulo (defensa contra reintroducción accidental).

    Si en el futuro hay un uso legítimo de base64 acá (ej. parseo de header
    Basic), borrar este test Y documentar el uso en CLAUDE.md.
    """
    # Búsqueda permisiva: `import base64` o `from base64 import ...`.
    pattern_top_level = re.compile(r"^\s*(?:import\s+base64|from\s+base64\s+import)", re.MULTILINE)
    pattern_local = re.compile(r"^\s+(?:import\s+base64|from\s+base64\s+import)", re.MULTILINE)
    assert not pattern_top_level.search(auth_src) and not pattern_local.search(auth_src), (
        "P0-AUDIT-1 defense-in-depth: auth.py importa `base64` sin razón "
        "tras el fix. El legacy lo usaba solo para el fallback inseguro. "
        "Si necesitas base64 para un caso de uso legítimo, documentar en "
        "CLAUDE.md y eliminar este test — pero asegurarte que no sea para "
        "decode unsigned de JWTs."
    )


def test_d_anchor_present_in_source(auth_src: str):
    """Ancla textual `P0-AUDIT-1-AUTH-VERIFY` o `[P0-AUDIT-1` presente —
    sin esto, un futuro refactor cosmético podría borrar el contexto del
    fix sin que nadie note.
    """
    assert "P0-AUDIT-1" in auth_src, (
        "P0-AUDIT-1 regresión: auth.py perdió la referencia al P-fix. "
        "Sin el anchor, un futuro refactor no sabe POR QUÉ se eliminó el "
        "fallback base64 y podría reintroducirlo creyendo que mejora la "
        "robustez frente a outages de Supabase."
    )


# ---------------------------------------------------------------------------
# C) Funcional: simulamos Supabase rechazando un token forjado.
# ---------------------------------------------------------------------------

def _build_forged_jwt(forged_sub: str) -> str:
    """Construye un JWT con header/payload válidos pero firma basura.

    Estructura: header.payload.signature.
    Header y payload se decodifican OK (lo que hacía el legacy fallback).
    Signature es texto random — NO valida contra ningún secret.
    """
    header_b64 = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(
        json.dumps(
            {"sub": forged_sub, "aud": "authenticated", "exp": 9999999999}
        ).encode()
    ).decode().rstrip("=")
    return f"{header_b64}.{payload_b64}.invalid_signature_attacker_basura"


def test_e_forged_jwt_does_not_return_unverified_sub(monkeypatch):
    """Funcional: si `supabase.auth.get_user` lanza excepción (lo que hace
    Supabase real ante una firma inválida), `get_verified_user_id` DEBE:
        - retornar None, O
        - levantar HTTPException 401/403.

    Inaceptable: retornar el `sub` forjado.
    """
    fake_supabase = MagicMock()
    fake_supabase.auth.get_user.side_effect = Exception(
        "invalid JWT signature (simulated)"
    )

    # Stub `db` ANTES de importar auth — auth.py hace `from db import ...`.
    # `monkeypatch.setitem` se auto-limpia al fin del test (no contamina la
    # sesión de pytest, lección P0-AGENT-1).
    db_stub = MagicMock()
    db_stub.supabase = fake_supabase
    db_stub.get_monthly_api_usage = lambda *a, **k: 0
    db_stub.get_user_profile = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "db", db_stub)

    # Reimportar auth con el stub aplicado.
    if "auth" in sys.modules:
        monkeypatch.delitem(sys.modules, "auth", raising=False)
    import auth as _auth  # type: ignore

    # Hard-set por si auth ya capturó la referencia al `supabase` original.
    monkeypatch.setattr(_auth, "supabase", fake_supabase, raising=False)

    forged_sub = "00000000-0000-0000-0000-000000000099"
    forged_token = _build_forged_jwt(forged_sub)

    from fastapi import HTTPException

    raised: HTTPException | None = None
    result: str | None = None
    try:
        result = _run_async(_auth.get_verified_user_id(f"Bearer {forged_token}"))
    except HTTPException as e:
        raised = e

    # Path 1: levantó 401/403 — aceptable.
    if raised is not None:
        assert raised.status_code in (401, 403), (
            f"P0-AUDIT-1: esperaba HTTPException 401/403 al rechazar token "
            f"forjado, recibió {raised.status_code}. Cualquier otro status "
            f"es ambiguo para el cliente."
        )
        return

    # Path 2: retornó algo. Solo None es aceptable.
    assert result is None, (
        f"P0-AUDIT-1 violation: get_verified_user_id retornó '{result}' "
        f"para un token con firma INVÁLIDA. El sub forjado era "
        f"'{forged_sub}'. Cualquier retorno != None abre IDOR universal — "
        f"el caller hace `if verified_user_id != target_user_id` y el "
        f"check pasa porque verified_user_id == victim_id (forjado).\n\n"
        f"El fix debe rechazar el token (retornar None o raise 403)."
    )
    assert result != forged_sub  # redundante pero explícito


def test_f_missing_authorization_returns_none(monkeypatch):
    """Contrato: sin header Authorization, retorna None (no raise)."""
    fake_supabase = MagicMock()
    db_stub = MagicMock()
    db_stub.supabase = fake_supabase
    db_stub.get_monthly_api_usage = lambda *a, **k: 0
    db_stub.get_user_profile = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "db", db_stub)
    if "auth" in sys.modules:
        monkeypatch.delitem(sys.modules, "auth", raising=False)
    import auth as _auth  # type: ignore
    monkeypatch.setattr(_auth, "supabase", fake_supabase, raising=False)

    assert _run_async(_auth.get_verified_user_id(None)) is None
    assert _run_async(_auth.get_verified_user_id("")) is None
    assert _run_async(_auth.get_verified_user_id("NotBearer abc.def.ghi")) is None
    # Bearer sin token tampoco debe crashear.
    assert _run_async(_auth.get_verified_user_id("Bearer ")) is None


def test_g_valid_token_returns_user_id(monkeypatch):
    """Contrato: token válido (supabase lo acepta) retorna user.id."""
    fake_user = MagicMock()
    fake_user.id = "11111111-2222-3333-4444-555555555555"
    fake_user_res = MagicMock()
    fake_user_res.user = fake_user

    fake_supabase = MagicMock()
    fake_supabase.auth.get_user.return_value = fake_user_res

    db_stub = MagicMock()
    db_stub.supabase = fake_supabase
    db_stub.get_monthly_api_usage = lambda *a, **k: 0
    db_stub.get_user_profile = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "db", db_stub)
    if "auth" in sys.modules:
        monkeypatch.delitem(sys.modules, "auth", raising=False)
    import auth as _auth  # type: ignore
    monkeypatch.setattr(_auth, "supabase", fake_supabase, raising=False)

    result = _run_async(_auth.get_verified_user_id("Bearer valid.token.signature"))
    assert result == "11111111-2222-3333-4444-555555555555"
