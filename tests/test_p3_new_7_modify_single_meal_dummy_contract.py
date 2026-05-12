"""[P3-NEW-7 · 2026-05-11] `tools.modify_single_meal` dummy IDOR contract.

Estado actual:
    La tool retorna `"DUMMY_CALL_ACTUALLY_INTERCEPTED"` (intercepted en
    capa superior). Riesgo: si un PR futuro implementa el cuerpo real
    SIN ownership filter ni advisory lock, abre IDOR + lost-update +
    quota burn simultáneamente.

Este test parser-based ancla el contrato:
    1. La docstring DEBE contener el bloque P3-NEW-7 CONTRATO con los
       tokens canónicos (user_id ownership, advisory lock, AND user_id).
    2. Si el body se reemplaza con algo distinto de `return "DUMMY_..."`,
       el body nuevo DEBE contener al menos UN check `user_id == ...`
       o `verified_user_id` (tooltip de ownership inline).

    Si alguien implementa el cuerpo sin esos tokens, este test falla
    con copy explicativo + link al docstring + invariantes I2/I7.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_FP = _REPO_ROOT / "backend" / "tools.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _TOOLS_FP.read_text(encoding="utf-8")


def _extract_function_body(src: str, func_signature_prefix: str) -> str:
    """Devuelve el cuerpo crudo de la función (entre el `def` y el siguiente
    `def`/`@tool` del top-level)."""
    start = src.find(func_signature_prefix)
    assert start > 0, f"`{func_signature_prefix}` no encontrado"
    # Boundary: la siguiente declaración top-level (@tool / @ / def en col 0).
    next_match = re.search(r"\n(@tool|def\s+\w+|@\w+)", src[start + len(func_signature_prefix):])
    end = start + len(func_signature_prefix) + (next_match.start() if next_match else len(src))
    return src[start:end]


def test_dummy_body_unchanged(src: str):
    """El cuerpo (sin docstring) DEBE ser `return "DUMMY_CALL_ACTUALLY_INTERCEPTED"`
    hasta que un PR explícito implemente la lógica real con ownership guard."""
    body = _extract_function_body(src, "def modify_single_meal(")
    assert 'return "DUMMY_CALL_ACTUALLY_INTERCEPTED"' in body, (
        "P3-NEW-7 regresión: el body de `modify_single_meal` ya NO retorna "
        "el sentinel `DUMMY_CALL_ACTUALLY_INTERCEPTED`. Si la tool se "
        "implementó, leer el bloque P3-NEW-7 CONTRATO de su docstring y "
        "verificar que el body nuevo:\n"
        "  1. Valida `user_id == verified_user_id` autenticado.\n"
        "  2. Filtra `AND user_id = %s` en TODA mutación SQL.\n"
        "  3. Aplica `acquire_meal_plan_advisory_lock(purpose='general')`.\n"
        "  4. NO bypassea `verify_api_quota`.\n"
        "Si TODO eso está en el body, este test puede relajarse para "
        "permitir el path implementado, PERO añadir un test específico "
        "que enforce las 4 condiciones de arriba."
    )


def test_dummy_contract_docstring_present(src: str):
    """La docstring debe contener el bloque CONTRATO P3-NEW-7 con tokens
    canónicos de ownership/IDOR."""
    body = _extract_function_body(src, "def modify_single_meal(")
    required_tokens = [
        "P3-NEW-7",
        "CONTRATO DE OWNERSHIP",
        "verified_user_id",
        "AND user_id",
        "advisory_lock",
        "verify_api_quota",
        "IDOR",
    ]
    for tok in required_tokens:
        assert tok in body, (
            f"P3-NEW-7 regresión: la docstring de `modify_single_meal` ya "
            f"no menciona `{tok}`. Sin este token, un revisor futuro puede "
            f"implementar el cuerpo sin contexto y abrir IDOR/lost-update. "
            f"Restaurar el bloque P3-NEW-7 CONTRATO completo antes de "
            f"mergear."
        )


def test_tooltip_anchor_present(src: str):
    """El tooltip-anchor `P3-NEW-7-DUMMY-CONTRACT` debe estar en la
    docstring para grep cross-codebase."""
    body = _extract_function_body(src, "def modify_single_meal(")
    assert "P3-NEW-7-DUMMY-CONTRACT" in body, (
        "P3-NEW-7 regresión: el tooltip-anchor `P3-NEW-7-DUMMY-CONTRACT` "
        "desapareció. Sin él, un grep desde otro test/code no encuentra "
        "esta función."
    )
