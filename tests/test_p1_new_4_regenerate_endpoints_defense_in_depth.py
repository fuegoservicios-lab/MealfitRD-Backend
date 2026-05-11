"""[P1-NEW-4 · 2026-05-10] `UPDATE meal_plans` en `regenerate-simplified` y
`regen-degraded` DEBE filtrar por `(id, user_id)` (defense-in-depth).

Bug original (audit 2026-05-10):
    - `regenerate-simplified` (plans.py:7072) UPDATE `meal_plans` solo
      `WHERE id = %s` puro.
    - `regen-degraded` (plans.py:7202) idem.
    Ambos endpoints SÍ tienen ownership check explícito al inicio del
    handler (SELECT user_id + comparación con verified_user_id). Sin
    embargo el UPDATE no replica el filtro: un futuro refactor que
    rompa el SELECT inicial sin tocar el UPDATE re-introduce IDOR.

    Mismo patrón que cerró P0-HIST-IDOR-1 retry-chunk:4119-4123 y
    P0-HIST-IDOR-2 chunk-status:3473.

Fix:
    Añadir `AND user_id = %s` al WHERE de ambos UPDATE, bind
    `verified_user_id` ya en scope.

Estrategia del test (parser estático sobre plans.py):
    1. Localizar los dos handlers.
    2. Verificar que ambos UPDATE a `meal_plans` filtran por user_id.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(
            f"No se encontró `def {fn_name}(` en plans.py — endpoint "
            f"renombrado/eliminado. Si es intencional, actualizar test."
        )
    start = m.start()
    next_def = re.search(
        r"\n(?:@router\.|@app\.|def\s)",
        src[start + 1:],
    )
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


def test_regenerate_simplified_update_filters_by_user_id(plans_src: str):
    """El handler `api_regenerate_dead_lettered_simplified` debe tener el UPDATE
    sobre `meal_plans` con `WHERE id = %s AND user_id = %s`."""
    body = _extract_function_body(plans_src, "api_regenerate_dead_lettered_simplified")

    # Extraer todos los UPDATE meal_plans del cuerpo y verificar que cada
    # uno tenga ambos predicados.
    update_blocks = re.findall(
        r"UPDATE\s+meal_plans\s+SET.*?WHERE\s+([^;\"]*?)(?:\"\"\"|''')",
        body,
        re.DOTALL,
    )
    assert update_blocks, (
        "P1-NEW-4 regresión: no se encontró ningún `UPDATE meal_plans` "
        "en `api_regenerate_dead_lettered_simplified`. Si la persistencia se movió a "
        "otro helper, actualizar este parser."
    )

    for where_clause in update_blocks:
        has_id = bool(re.search(r"id\s*=\s*%s", where_clause))
        has_user = bool(re.search(r"user_id\s*=\s*%s", where_clause))
        assert has_id and has_user, (
            f"P1-NEW-4 regresión: UPDATE meal_plans en "
            f"`api_regenerate_dead_lettered_simplified` con WHERE "
            f"`{where_clause.strip()[:150]}...` NO filtra por user_id. "
            f"Defense-in-depth roto vs. P0-HIST-IDOR-1 retry-chunk pattern."
        )


def test_regen_degraded_update_filters_by_user_id(plans_src: str):
    """El handler `api_regen_degraded_chunks` debe tener el UPDATE
    sobre `meal_plans` con `WHERE id = %s AND user_id = %s`."""
    body = _extract_function_body(plans_src, "api_regen_degraded_chunks")

    update_blocks = re.findall(
        r"UPDATE\s+meal_plans\s+SET.*?WHERE\s+([^;\"]*?)(?:\"\"\"|''')",
        body,
        re.DOTALL,
    )
    assert update_blocks, (
        "P1-NEW-4 regresión: no se encontró `UPDATE meal_plans` en "
        "`api_regen_degraded_chunks`."
    )

    for where_clause in update_blocks:
        has_id = bool(re.search(r"id\s*=\s*%s", where_clause))
        has_user = bool(re.search(r"user_id\s*=\s*%s", where_clause))
        assert has_id and has_user, (
            f"P1-NEW-4 regresión: UPDATE meal_plans en "
            f"`api_regen_degraded_chunks` con WHERE "
            f"`{where_clause.strip()[:150]}...` NO filtra por user_id. "
            f"Defense-in-depth roto."
        )


def test_p1_new_4_anchor_present(plans_src: str):
    """Anchor `P1-NEW-4` en comentarios de ambos handlers — `grep`
    debe poder localizar el fix."""
    body_simplified = _extract_function_body(plans_src, "api_regenerate_dead_lettered_simplified")
    body_degraded = _extract_function_body(plans_src, "api_regen_degraded_chunks")
    assert "P1-NEW-4" in body_simplified, (
        "P1-NEW-4 regresión: anchor `P1-NEW-4` desapareció de "
        "`api_regenerate_dead_lettered_simplified`."
    )
    assert "P1-NEW-4" in body_degraded, (
        "P1-NEW-4 regresión: anchor `P1-NEW-4` desapareció de "
        "`api_regen_degraded_chunks`."
    )
