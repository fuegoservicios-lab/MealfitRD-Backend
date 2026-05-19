"""[P2-HIST-AUDIT-6 · 2026-05-09] Tests: ``api_rename_plan`` rechaza
nombres con control chars (0x00-0x1F + DEL 0x7F).

Bug original (audit historial 2026-05-08):
    El endpoint de rename validaba `name` no-string / vacío post-trim
    / cap 200 chars, pero NO control chars. Casos rotos:
      - NUL byte (\\u0000): Postgres jsonb lanza
        `unsupported Unicode escape sequence` al hacer
        `to_jsonb('foo\\x00bar'::text)` → ROLLBACK silencioso del
        UPDATE → 500 sin detalle útil.
      - CR/LF/TAB interiores: rompen render del UI (card del Historial
        muestra primera línea, control char invisible queda en el
        nombre persistido → copy/paste lo arrastra a otros contextos).
      - DEL (0x7F): legacy artifact, no printable.

Fix:
    Validación O(n) sobre el name post-trim. Si encuentra char con
    codepoint < 0x20 o == 0x7F → 400 con mensaje específico. Caracteres
    no-ASCII printables (ñ, á, emoji) son OK.

Cobertura:
    - 400 con NUL byte (el caso patológico que rompe jsonb).
    - 400 con CR / LF / TAB / VT / otros 0x01-0x1F.
    - 400 con DEL (0x7F).
    - 200 con caracteres unicode no-ASCII (ñ, á, emoji).
    - 200 con espacios interiores (codepoint 0x20+).
    - El cap 200 chars sigue funcionando (sin regresión).
    - El check empty post-trim sigue funcionando.
    - Anchor del marker.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER = "11111111-1111-1111-1111-111111111111"
_PLAN_ID = "dddddddd-dddd-dddd-dddd-dddddddddddd"


# Mocks (igual que test_p1_hist_5)
class _CursorRecorder:
    def __init__(self, fetchall_returns=None):
        self.calls = []
        self._fetchall_returns = list(fetchall_returns or [])

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchall(self):
        if self._fetchall_returns:
            return self._fetchall_returns.pop(0)
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _ConnRecorder:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def cursor(self, *a, **kw):
        return self.cursor_obj

    def transaction(self):
        class _Tx:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, exc_type, *a):
                return False
        return _Tx()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _build_pool_mock(cursor):
    from unittest.mock import MagicMock
    pool = MagicMock()
    pool.connection.return_value = _ConnRecorder(cursor)
    return pool


def _client():
    from auth import verify_api_quota, get_verified_user_id
    from routers.plans import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER

    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER
    return client


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_endpoint():
    from routers.plans import api_rename_plan
    src = inspect.getsource(api_rename_plan)
    assert "P2-HIST-AUDIT-6" in src


# ---------------------------------------------------------------------------
# 2. NUL byte (el caso patológico que rompe jsonb)
# ---------------------------------------------------------------------------
def test_rename_rejects_nul_byte():
    """`\\x00` en el name rompe Postgres jsonb. Sin esta validación,
    el endpoint cae al UPDATE → ROLLBACK → 500 sin mensaje útil.
    """
    client = _client()
    bad_names = [
        "Plan\x00Malicious",      # interior
        "\x00Malicious",          # leading
        "Malicious\x00",          # trailing
        "Plan con \x00 NUL byte",  # interior con espacios
    ]
    for bad in bad_names:
        r = client.patch(
            f"/api/plans/{_PLAN_ID}/name",
            json={"name": bad},
        )
        assert r.status_code == 400, (
            f"name {bad!r} debió rechazar (NUL byte rompe jsonb); "
            f"got {r.status_code} {r.text}"
        )
        assert "control" in r.json().get("detail", "").lower() or \
               "invalid" in r.json().get("detail", "").lower()


# ---------------------------------------------------------------------------
# 3. Otros control chars 0x01-0x1F + DEL (0x7F)
# ---------------------------------------------------------------------------
def test_rename_rejects_cr_lf_tab_inside():
    """CR (\\r), LF (\\n), TAB (\\t) interiores rompen el render UI."""
    client = _client()
    for bad in [
        "Plan\rcon CR",
        "Plan\ncon LF",
        "Plan\tcon TAB",
        "Plan\r\ncon CRLF",
    ]:
        r = client.patch(
            f"/api/plans/{_PLAN_ID}/name",
            json={"name": bad},
        )
        assert r.status_code == 400, (
            f"name {bad!r} debió rechazar (control char interior); "
            f"got {r.status_code}"
        )


def test_rename_rejects_other_control_chars():
    """Codepoints 0x01-0x1F (SOH, STX, BEL, FF, etc.) son input
    malicioso o bug cliente — un nombre legítimo nunca los tiene."""
    client = _client()
    for codepoint in [0x01, 0x07, 0x0B, 0x0C, 0x0E, 0x1B, 0x1F]:
        bad = f"Plan{chr(codepoint)}weird"
        r = client.patch(
            f"/api/plans/{_PLAN_ID}/name",
            json={"name": bad},
        )
        assert r.status_code == 400, (
            f"codepoint {hex(codepoint)} debió rechazar; got {r.status_code}"
        )


def test_rename_rejects_del_0x7f():
    """DEL (0x7F) es legacy unicode artifact, no printable."""
    client = _client()
    bad = "Plan\x7Fweird"
    r = client.patch(
        f"/api/plans/{_PLAN_ID}/name",
        json={"name": bad},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 4. Caracteres unicode válidos (NO debe rechazar)
# ---------------------------------------------------------------------------
def test_rename_accepts_unicode_printable():
    """ñ, á, é, ü, emoji 🍳 son válidos (codepoints >= 0x20)."""
    client = _client()
    cursor = _CursorRecorder(fetchall_returns=[[{"id": _PLAN_ID}]])
    pool_mock = _build_pool_mock(cursor)
    valid_names = [
        "Plan Sintético",
        "Mi plan con ñ",
        "Plan con áéíóú",
        "Pollo guisado 🍳",
        "Plan #1 — Lunes",       # em-dash
        "Plan «con guillemets»",  # comillas francesas
    ]
    for valid in valid_names:
        # Reset cursor para cada llamada.
        cursor._fetchall_returns = [[{"id": _PLAN_ID}]]
        cursor.calls = []
        with patch("db_core.connection_pool", pool_mock):
            r = client.patch(
                f"/api/plans/{_PLAN_ID}/name",
                json={"name": valid},
            )
        assert r.status_code == 200, (
            f"name {valid!r} (codepoints todos >= 0x20) debió aceptar; "
            f"got {r.status_code} {r.text}"
        )
        # Trim + persist preserva el nombre tal cual.
        assert r.json()["name"] == valid


def test_rename_accepts_spaces_interior():
    """Codepoint 0x20 (SPACE) es válido, no debe rechazarse."""
    client = _client()
    cursor = _CursorRecorder(fetchall_returns=[[{"id": _PLAN_ID}]])
    pool_mock = _build_pool_mock(cursor)
    with patch("db_core.connection_pool", pool_mock):
        r = client.patch(
            f"/api/plans/{_PLAN_ID}/name",
            json={"name": "Plan con muchos    espacios"},
        )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# 5. Sin regresión de validaciones previas
# ---------------------------------------------------------------------------
def test_cap_200_still_enforced():
    """Cap 200 chars sigue rechazando antes de llegar a la
    validación de control chars (orden de checks)."""
    client = _client()
    too_long = "X" * 201
    r = client.patch(
        f"/api/plans/{_PLAN_ID}/name",
        json={"name": too_long},
    )
    assert r.status_code == 400
    assert "200" in r.json().get("detail", "")


def test_empty_post_trim_still_enforced():
    """Strings whitespace-only siguen rechazándose."""
    client = _client()
    r = client.patch(
        f"/api/plans/{_PLAN_ID}/name",
        json={"name": "   "},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 6. Comentario load-bearing: explica el caso jsonb NUL
# ---------------------------------------------------------------------------
def test_comment_explains_jsonb_nul_case():
    """El comentario debe explicar que el NUL byte rompe jsonb —
    sin esto, un mantenedor podría borrar la validación pensando
    que es over-zealous."""
    from routers.plans import api_rename_plan
    src = inspect.getsource(api_rename_plan)
    # Buscamos uno de: jsonb, NUL byte, Unicode escape.
    anchors = ("jsonb", "NUL", "Unicode")
    assert any(a in src for a in anchors), (
        f"El comentario debe mencionar uno de {anchors} para explicar "
        f"el caso patológico del NUL byte."
    )
