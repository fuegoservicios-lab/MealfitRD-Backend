"""[P1-AUDIT-3 · 2026-05-12] Path params `{user_id}` / `{session_id}` deben
rechazar UUIDs malformados con HTTP 400 antes de tocar la DB.

Contexto del audit:
    Logs prod 2026-05-12 mostraron `invalid input syntax for type uuid:
    "0000"` y `"test"` — externals (security scanners / probes) hitteando
    endpoints autenticados con valores no-UUID. El handler ejecutaba
    `SELECT … WHERE id = %s` (column `uuid`) → postgres ERROR → 500 al
    cliente (sin info útil) + polución de logs.

Fix:
    Helper `path_validators.assert_valid_uuid(value, allow_guest=False)`
    aplicado al inicio de los handlers en `routers/chat.py` y
    `routers/diary.py` que reciben `{user_id}`/`{session_id}`. Retorna 400
    explícito antes de SQL. Sentinel `"guest"` permitido cuando el handler
    lo acepta (chat sessions, diary consumed).

Lo que este test enforza:
  A) `backend/path_validators.py` define `assert_valid_uuid`.
  B) Función levanta `HTTPException 400` para entradas no-UUID.
  C) `"guest"` aceptado iff `allow_guest=True`.
  D) Endpoints listados de `routers/chat.py` invocan `assert_valid_uuid`.
  E) `routers/diary.py::/consumed/{user_id}` invoca `assert_valid_uuid`.

Tooltip-anchor: P1-AUDIT-3-UUID-VALIDATOR (en path_validators.py).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PATH_VALIDATORS_PY = _BACKEND_ROOT / "path_validators.py"
_CHAT_PY = _BACKEND_ROOT / "routers" / "chat.py"
_DIARY_PY = _BACKEND_ROOT / "routers" / "diary.py"


@pytest.fixture(scope="module")
def chat_src() -> str:
    return _CHAT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def diary_src() -> str:
    return _DIARY_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A) Helper existe y funciona.
# ---------------------------------------------------------------------------

def test_a_path_validators_module_exists():
    assert _PATH_VALIDATORS_PY.exists(), (
        "P1-AUDIT-3: falta `backend/path_validators.py`. Sin el helper, "
        "los endpoints no pueden validar UUIDs sin duplicar lógica."
    )


def test_b_assert_valid_uuid_defined():
    src = _PATH_VALIDATORS_PY.read_text(encoding="utf-8")
    assert re.search(r"^def assert_valid_uuid\b", src, re.MULTILINE), (
        "P1-AUDIT-3: `path_validators.py` no define `assert_valid_uuid`. "
        "Firma esperada: `assert_valid_uuid(value: str, allow_guest: bool = "
        "False) -> str`."
    )


def test_b2_assert_valid_uuid_uses_uuid_stdlib():
    """El validator debe usar el módulo `uuid` stdlib — no regex casero."""
    src = _PATH_VALIDATORS_PY.read_text(encoding="utf-8")
    assert "uuid" in src and "UUID(" in src, (
        "P1-AUDIT-3: `assert_valid_uuid` debe usar `uuid.UUID(value)` del "
        "stdlib (rechaza formatos no-UUID por parseo, no por regex). "
        "Un regex casero puede dejar pasar formas inválidas que postgres "
        "rechaza."
    )


# ---------------------------------------------------------------------------
# B) Funcional: validator rechaza/acepta correctamente.
# ---------------------------------------------------------------------------

def test_c_rejects_garbage_input():
    """Entradas no-UUID levantan 400."""
    from fastapi import HTTPException
    from path_validators import assert_valid_uuid

    for bad in ("test", "0000", "abc", "", "123-456", "guest"):
        # "guest" rechazado por default (allow_guest=False).
        with pytest.raises(HTTPException) as exc:
            assert_valid_uuid(bad)
        assert exc.value.status_code == 400


def test_d_accepts_guest_when_allowed():
    from path_validators import assert_valid_uuid
    assert assert_valid_uuid("guest", allow_guest=True) == "guest"


def test_e_accepts_valid_uuid():
    from path_validators import assert_valid_uuid
    valid = "11111111-2222-3333-4444-555555555555"
    assert assert_valid_uuid(valid) == valid
    assert assert_valid_uuid(valid, allow_guest=True) == valid


def test_f_rejects_none_and_empty():
    from fastapi import HTTPException
    from path_validators import assert_valid_uuid

    for bad in (None, ""):
        with pytest.raises(HTTPException) as exc:
            assert_valid_uuid(bad)
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# C) Endpoints aplican el validator.
# ---------------------------------------------------------------------------

# Endpoints de chat.py con path params expuestos.
_CHAT_PROTECTED_ENDPOINTS = [
    # (route, decorator_method, allow_guest_expected)
    ("/sessions/{user_id}", "get", True),
    ("/sessions/{user_id}", "delete", True),
    ("/session/{session_id}", "put", False),
    ("/history/{session_id}", "get", False),
    ("/session/{session_id}", "delete", False),
]


@pytest.mark.parametrize("route,method,allow_guest", _CHAT_PROTECTED_ENDPOINTS)
def test_g_chat_endpoint_calls_assert_valid_uuid(chat_src: str, route: str, method: str, allow_guest: bool):
    """Cada endpoint protegido en chat.py debe invocar `assert_valid_uuid`."""
    # Localizar el bloque del endpoint: decorador + función + cuerpo hasta
    # el próximo `@router.` o EOF.
    decorator_pattern = re.compile(
        rf'@router\.{method}\(\s*["\']{re.escape(route)}["\']\s*\)'
        rf'.*?'
        rf'(?=@router\.|$)',
        re.DOTALL,
    )
    m = decorator_pattern.search(chat_src)
    assert m, (
        f"P1-AUDIT-3: no localicé `@router.{method}('{route}')` en chat.py. "
        f"Estructura del archivo cambió, actualizar este test."
    )
    body = m.group(0)
    assert "assert_valid_uuid(" in body, (
        f"P1-AUDIT-3 regresión: endpoint `{method.upper()} {route}` en "
        f"chat.py NO invoca `assert_valid_uuid(...)`. Sin la validación, "
        f"un probe con valor no-UUID (ej. 'test', '0000') hace que "
        f"postgres rechace el SELECT con error y el handler devuelva 500 "
        f"en lugar de 400 explícito."
    )

    if allow_guest:
        assert re.search(r"assert_valid_uuid\([^)]*allow_guest\s*=\s*True", body), (
            f"P1-AUDIT-3: endpoint `{method.upper()} {route}` debe pasar "
            f"`allow_guest=True` (acepta el sentinel 'guest' para sesiones "
            f"anónimas)."
        )


def test_h_diary_consumed_calls_assert_valid_uuid(diary_src: str):
    pattern = re.compile(
        r'@router\.get\(\s*["\']/consumed/\{user_id\}["\']\s*\).*?(?=@router\.|$)',
        re.DOTALL,
    )
    m = pattern.search(diary_src)
    assert m, "P1-AUDIT-3: no localicé `/consumed/{user_id}` en diary.py."
    body = m.group(0)
    assert "assert_valid_uuid(" in body, (
        "P1-AUDIT-3 regresión: `/api/diary/consumed/{user_id}` no invoca "
        "`assert_valid_uuid`. Un probe con user_id='test' produce el "
        "error UUID cast que el audit prod observó."
    )
    assert re.search(r"assert_valid_uuid\([^)]*allow_guest\s*=\s*True", body), (
        "P1-AUDIT-3: `/api/diary/consumed/{user_id}` debe pasar "
        "`allow_guest=True` — el handler ya devuelve `{meals: []}` para "
        "user_id='guest'."
    )


# ---------------------------------------------------------------------------
# D) Anchor en el helper.
# ---------------------------------------------------------------------------

def test_i_anchor_present_in_validator():
    src = _PATH_VALIDATORS_PY.read_text(encoding="utf-8")
    assert "P1-AUDIT-3" in src, (
        "P1-AUDIT-3 regresión: `path_validators.py` perdió el anchor "
        "textual `P1-AUDIT-3`. Sin el anchor, un futuro refactor pierde "
        "el contexto del audit."
    )
