"""[P3-1 · 2026-05-07] Smoke tests para `safe_error_detail`.

Cierra P3-1 del audit `project_audit_p0_p1_close_2026_05_07.md` (backlog):
42 call sites en 7 archivos (app.py + 6 routers) hacían
`raise HTTPException(status_code=500, detail=str(e))`, lo que filtra el
texto bruto de Postgres exceptions (constraint names, schema, SQL fragments)
al cliente HTTP. Wrapper `safe_error_detail` centraliza el gate.

Contrato:
  - default (knob unset / cualquier valor != "true"): devuelve mensaje
    genérico `"Internal server error (ref: <8-hex>)"` y loggea el exc real
    en el módulo `error_utils` con el correlation_id.
  - `MEALFIT_LEAK_DB_ERRORS=true`: devuelve `str(exc)` tal cual (modo dev).
  - El knob se relee en cada invocación (toggle sin redeploy).
"""
import re
import logging

import pytest

from error_utils import safe_error_detail


REF_RE = re.compile(r"^Internal server error \(ref: ([0-9a-f]{8})\)$")


# ---------------------------------------------------------------------------
# Modo prod (default)
# ---------------------------------------------------------------------------
def test_default_returns_generic_with_correlation_id(monkeypatch):
    """Sin knob → mensaje genérico + correlation_id de 8 hex."""
    monkeypatch.delenv("MEALFIT_LEAK_DB_ERRORS", raising=False)
    detail = safe_error_detail(RuntimeError("psycopg2.errors.UndefinedColumn: column foo"))
    m = REF_RE.match(detail)
    assert m is not None, f"expected generic message + cid, got: {detail}"
    cid = m.group(1)
    assert len(cid) == 8


def test_default_does_not_leak_exception_text(monkeypatch):
    """El str(exc) NO debe aparecer en el detail devuelto."""
    monkeypatch.delenv("MEALFIT_LEAK_DB_ERRORS", raising=False)
    secret_text = "user_inventory_pk_constraint_violation_uid_42"
    detail = safe_error_detail(ValueError(secret_text))
    assert secret_text not in detail


def test_default_logs_exception_with_correlation_id(monkeypatch, caplog):
    """En modo prod debe loggear el exc real con el cid para correlación SRE."""
    monkeypatch.delenv("MEALFIT_LEAK_DB_ERRORS", raising=False)
    with caplog.at_level(logging.ERROR, logger="error_utils"):
        detail = safe_error_detail(KeyError("hidden_internal"), context="diary.upload")
    cid = REF_RE.match(detail).group(1)
    assert any(
        f"cid={cid}" in rec.message and "diary.upload" in rec.message and "KeyError" in rec.message
        for rec in caplog.records
    ), f"no se encontró log paralelo con cid={cid}"


# ---------------------------------------------------------------------------
# Modo dev (knob=true)
# ---------------------------------------------------------------------------
def test_leak_true_returns_raw_exception_text(monkeypatch):
    monkeypatch.setenv("MEALFIT_LEAK_DB_ERRORS", "true")
    detail = safe_error_detail(RuntimeError("raw_postgres_error_x"))
    assert detail == "raw_postgres_error_x"


def test_leak_true_case_insensitive(monkeypatch):
    monkeypatch.setenv("MEALFIT_LEAK_DB_ERRORS", "TRUE")
    detail = safe_error_detail(RuntimeError("visible"))
    assert detail == "visible"


@pytest.mark.parametrize("val", ["false", "0", "no", "", "off", "garbage"])
def test_leak_disabled_for_non_true_values(monkeypatch, val):
    """Sólo `true` (case-insensitive) habilita leak — cualquier otro valor → genérico."""
    monkeypatch.setenv("MEALFIT_LEAK_DB_ERRORS", val)
    detail = safe_error_detail(RuntimeError("must_not_leak"))
    assert REF_RE.match(detail), f"valor {val!r} debería caer a default; got: {detail}"
    assert "must_not_leak" not in detail


# ---------------------------------------------------------------------------
# Toggle dinámico (sin redeploy)
# ---------------------------------------------------------------------------
def test_knob_re_read_each_call(monkeypatch):
    """Cambiar el env entre llamadas debe alternar el comportamiento sin reload."""
    exc = RuntimeError("payload_secret")

    monkeypatch.setenv("MEALFIT_LEAK_DB_ERRORS", "true")
    assert safe_error_detail(exc) == "payload_secret"

    monkeypatch.setenv("MEALFIT_LEAK_DB_ERRORS", "false")
    detail = safe_error_detail(exc)
    assert REF_RE.match(detail)
    assert "payload_secret" not in detail


# ---------------------------------------------------------------------------
# Correlation ids son únicos por invocación
# ---------------------------------------------------------------------------
def test_correlation_id_unique_per_call(monkeypatch):
    monkeypatch.delenv("MEALFIT_LEAK_DB_ERRORS", raising=False)
    cids = set()
    for _ in range(20):
        detail = safe_error_detail(RuntimeError("x"))
        cids.add(REF_RE.match(detail).group(1))
    assert len(cids) == 20, "correlation_ids no son únicos"
