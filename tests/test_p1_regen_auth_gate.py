"""[P1-REGEN-AUTH-GATE · 2026-06-18] (audit fresco P1-B) Guard de auth en endpoints de recovery de chunks.

`/{plan_id}/regen-degraded` y `/{plan_id}/chunks/{chunk_id}/regenerate-simplified` usaban solo
`Depends(verify_api_quota)`, que retorna None sin auth (no levanta 401). Su ownership check es CONDICIONAL
(`if verified_user_id and ...`) → se salta cuando no hay auth. Un no-autenticado que conozca/adivine un
plan_id (UUID) podía forzar el re-encolado de chunks ajenos → llamadas LLM sin billing (cost-amp/DoS).

Fix: `if not verified_user_id: raise HTTPException(401)` al inicio de ambos handlers (espejo de /retry-chunk,
P0-HIST-IDOR-1) — vuelve el ownership check incondicional. Parser-based: ancla el guard en el source para que
un refactor que lo elimine falle el test.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_PLANS_PATH = Path(__file__).resolve().parent.parent / "routers" / "plans.py"


def _extract_function_body(src: str, func_name: str) -> str:
    """Extrae el cuerpo de una función top-level por nombre (hasta el próximo `def `/`@router`
    a indentación de módulo o EOF). Suficiente para anclar el guard al inicio del handler."""
    m = re.search(rf"\ndef {re.escape(func_name)}\(", src)
    assert m is not None, f"No se encontró `def {func_name}(` en {_PLANS_PATH}"
    start = m.start()
    # El próximo decorador/def a nivel de módulo (columna 0) marca el fin del cuerpo.
    nxt = re.search(r"\n(?:@router\.|def |class )", src[start + 1:])
    end = (start + 1 + nxt.start()) if nxt else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def src() -> str:
    return _PLANS_PATH.read_text(encoding="utf-8")


_HANDLERS = ["api_regen_degraded_chunks", "api_regenerate_dead_lettered_simplified"]


@pytest.mark.parametrize("func_name", _HANDLERS)
def test_handler_has_401_guard(src, func_name):
    body = _extract_function_body(src, func_name)
    assert "if not verified_user_id:" in body, (
        f"{func_name} no tiene el guard `if not verified_user_id:` — un no-autenticado "
        f"podría forzar el re-encolado de chunks ajenos (P1-REGEN-AUTH-GATE)."
    )
    assert "status_code=401" in body, f"{func_name} no levanta HTTPException 401 sin auth."


@pytest.mark.parametrize("func_name", _HANDLERS)
def test_401_guard_precedes_db_access(src, func_name):
    """El guard debe estar ANTES de cualquier SELECT/UPDATE: si corre después del ownership
    check condicional, el daño (re-encolado) ya pudo ocurrir."""
    body = _extract_function_body(src, func_name)
    guard_idx = body.find("if not verified_user_id:")
    assert guard_idx != -1
    # El primer acceso a DB (SELECT del plan o execute_sql_*) debe venir DESPUÉS del guard.
    first_db = min(
        (i for i in (body.find("execute_sql_query"), body.find("execute_sql_write"),
                     body.find('SELECT user_id')) if i != -1),
        default=-1,
    )
    assert first_db != -1, f"{func_name}: no se halló acceso a DB (¿refactor?)."
    assert guard_idx < first_db, (
        f"{func_name}: el guard 401 debe preceder al primer acceso a DB."
    )


def test_anchor_marker(src):
    assert "P1-REGEN-AUTH-GATE" in src
