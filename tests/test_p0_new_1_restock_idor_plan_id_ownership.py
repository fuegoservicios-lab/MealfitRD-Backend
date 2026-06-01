"""[P0-NEW-1 · 2026-05-10] `POST /api/plans/restock` con `plan_id` user-provided
DEBE chequear ownership; SELECT y UPDATE deben filtrar por user_id.

Bug original (audit 2026-05-10):
    El handler `api_restock` (plans.py:3113) aceptaba `plan_id` en el body
    del request y lo usaba sin ownership check:
      - SELECT `meal_plans` solo `WHERE id = %s` → leía plan_data ajeno.
      - UPDATE `meal_plans` solo `WHERE id = real_plan_id` → corrompía
        `plan_data` de la víctima con `is_restocked=True`,
        `restocked_at_iso=NOW`, `restocked_items={keys: NOW}` arbitrarios.
    Resultado: la lista de compras de la víctima omitía perecederos por
    hasta `MEALFIT_PERISHABLE_CYCLE_DAYS` (default 7 días). Misma familia
    que P0-HIST-IDOR-1 (retry-chunk) y P0-HIST-IDOR-2 (chunk-status)
    cerrados el 2026-05-10; el audit inicial no cubrió `/restock`.

Estrategia del test (parser estático, mismo patrón que
`test_p0_hist_idor_1_retry_chunk_ownership.py`):
    1. Localizar `api_restock` en plans.py (signature + body) hasta el
       siguiente top-level `def`/`@router`.
    2. Verificar que el SELECT en la rama `if plan_id` filtra por user_id.
    3. Verificar que el UPDATE final filtra por user_id (defense-in-depth).
    4. Verificar que un plan_id no resoluble lanza HTTPException 404.
    5. Verificar re-raise explícito de HTTPException antes del catch genérico
       (sin esto, el 404 se re-wrappea a 500 y el cliente pierde la señal).

Drift detection bidireccional:
    - Si alguien revierte el `.eq("user_id", user_id)` del SELECT → falla
      `test_restock_select_filters_by_user_id`.
    - Si alguien quita el `.eq("user_id", user_id)` del UPDATE → falla
      `test_restock_update_filters_by_user_id_defense_in_depth`.
    - Si alguien quita el 404 → falla
      `test_restock_404_on_plan_id_ownership_mismatch`.
    - Si el `except HTTPException: raise` desaparece → falla
      `test_restock_reraises_httpexception_before_generic_catch`.
    - Si alguien renombra `api_restock` → AssertionError explícita.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    """Devuelve `def <fn>(` hasta el siguiente top-level `def `/`@router`/
    `@app`. Mismo helper que test_p0_hist_idor_1.
    """
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(
            f"No se encontró `def {fn_name}(` en plans.py — el endpoint "
            f"fue renombrado/eliminado. Si el rename es intencional, "
            f"actualizar este test con el nuevo nombre. Si fue eliminado, "
            f"verificar que el patrón IDOR del audit P0-NEW-1 no se "
            f"reintrodujo en otro endpoint."
        )
    start = m.start()
    next_def = re.search(
        r"\n(?:@router\.|@app\.|def\s)",
        src[start + 1:],
    )
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def restock_body() -> str:
    src = _PLANS_PY.read_text(encoding="utf-8")
    return _extract_function_body(src, "api_restock")


def test_restock_select_filters_by_user_id(restock_body: str):
    """El SELECT en la rama `if plan_id` DEBE filtrar por user_id.

    Forma aceptada (supabase-py builder):
        .eq("id", plan_id).eq("user_id", user_id)
    En cualquier orden — el contrato es que ambos predicados coexisten.
    """
    # Buscamos el bloque `if plan_id:` y verificamos que en sus líneas
    # subsiguientes aparezca tanto .eq("id", plan_id) como .eq("user_id", user_id).
    branch_match = re.search(
        r"if\s+plan_id\s*:[\r\n]+(?P<body>(?:\s+.*[\r\n]+){1,30})",
        restock_body,
    )
    assert branch_match, (
        "P0-NEW-1 regresión: no se encontró la rama `if plan_id:` en "
        "`api_restock`. Si la lógica de detección del plan_id cambió, "
        "actualizar este parser."
    )
    branch_body = branch_match.group("body")

    has_id_filter = bool(re.search(r'\.eq\(\s*["\']id["\']\s*,\s*plan_id\s*\)', branch_body))
    has_user_filter = bool(re.search(r'\.eq\(\s*["\']user_id["\']\s*,\s*user_id\s*\)', branch_body))

    assert has_id_filter and has_user_filter, (
        "P0-NEW-1 regresión: el SELECT dentro de `if plan_id:` NO "
        "filtra simultáneamente por `id` y `user_id`. "
        f"has_id_filter={has_id_filter}, has_user_filter={has_user_filter}. "
        "Sin `.eq('user_id', user_id)` un atacante autenticado puede leer "
        "(y subsecuentemente corromper) `plan_data` de planes ajenos."
    )


def test_restock_404_on_plan_id_ownership_mismatch(restock_body: str):
    """Si el `plan_id` user-provided no es resoluble para este user,
    debe lanzar HTTPException 404 (no 403 — no filtrar la existencia
    del plan ajeno; mismo contrato que P0-HIST-IDOR-1/2).
    """
    # Patrón: dentro de la rama `if plan_id:`, después del SELECT, debe
    # haber un raise HTTPException(status_code=404 cuando data está vacío.
    pattern = re.compile(
        r"if\s+(?:not\s+\(?plan_res|not\s+plan_res\.data|not\s+\([^)]*plan_res[^)]*\))"
        r".*?raise\s+HTTPException\s*\(\s*status_code\s*=\s*404",
        re.IGNORECASE | re.DOTALL,
    )
    assert pattern.search(restock_body), (
        "P0-NEW-1 regresión: no se encontró `raise HTTPException("
        "status_code=404)` para el caso `plan_id` user-provided no "
        "resoluble. Sin este 404 explícito, el handler caería al "
        "fallback latest (wrong-plan persist) o silenciaría el IDOR. "
        "Mismo contrato que retry-chunk:4106 y chunk-status:3474."
    )


def test_restock_update_filters_by_user_id_defense_in_depth(restock_body: str):
    """El persist de `plan_data` DEBE filtrar por user_id (defense-in-depth:
    aunque el SELECT arriba ya cerró el IDOR, mirroring del patrón
    P0-HIST-IDOR-1 retry-chunk:4119-4123 protege contra futuros refactors
    que rompan el ownership check sin tocar el persist).

    [P1-RESTOCK-LOSTUPDATE · 2026-05-30] La persistencia migró de full-overwrite
    `supabase.table("meal_plans").update({"plan_data": plan_data}).eq("id",...)
    .eq("user_id", user_id)` a `update_plan_data_atomic(real_plan_id, _restock_mutator,
    user_id=user_id)` (SELECT … FOR UPDATE para cerrar la ventana lost-update I7).
    El filtro de ownership SE PRESERVA: `update_plan_data_atomic` incluye
    `AND user_id = %s` en su SELECT y UPDATE internos (P2-OPEN-1) cuando se le
    pasa `user_id=`. Este test ahora acepta CUALQUIERA de las dos formas, pero
    EXIGE que el persist esté gateado por user_id de un modo u otro.
    """
    # Forma A (legacy supabase-py): `.table("meal_plans").update(...).eq("id",...).eq("user_id", user_id)`.
    update_pattern = re.compile(
        r'\.table\(\s*["\']meal_plans["\']\s*\)\s*\.update\([^)]*\)'
        r'(?P<chain>(?:\s*\.eq\([^)]+\))+)',
        re.DOTALL,
    )
    legacy_updates = update_pattern.findall(restock_body)

    # Forma B (atómica P1-RESTOCK-LOSTUPDATE): `update_plan_data_atomic(real_plan_id, ..., user_id=user_id)`.
    atomic_match = re.search(
        r"update_plan_data_atomic\s*\(\s*real_plan_id\s*,(?P<args>.*?)\)",
        restock_body,
        re.DOTALL,
    )
    atomic_has_user = bool(
        atomic_match and re.search(r"user_id\s*=\s*user_id", atomic_match.group("args"))
    )

    assert legacy_updates or atomic_has_user, (
        "P0-NEW-1 / P1-RESTOCK-LOSTUPDATE regresión: el persist de plan_data en "
        "`api_restock` no usa NI el patrón legacy `.table('meal_plans').update(...)"
        ".eq(...)` NI `update_plan_data_atomic(real_plan_id, ..., user_id=user_id)`. "
        "El ownership/lost-update guard desapareció."
    )

    if atomic_has_user:
        # Path canónico actual: el filtro user_id vive dentro de
        # update_plan_data_atomic; basta con que se le pase user_id=user_id.
        return

    # Path legacy: cada UPDATE chain debe filtrar id + user_id.
    for chain in legacy_updates:
        has_id = bool(re.search(r'\.eq\(\s*["\']id["\']\s*,\s*\w+\s*\)', chain))
        has_user = bool(re.search(r'\.eq\(\s*["\']user_id["\']\s*,\s*user_id\s*\)', chain))
        assert has_id and has_user, (
            f"P0-NEW-1 regresión: UPDATE meal_plans con chain `{chain.strip()[:200]}` "
            f"NO filtra simultáneamente por id y user_id. "
            f"has_id={has_id}, has_user={has_user}. Defense-in-depth roto."
        )


def test_restock_reraises_httpexception_before_generic_catch(restock_body: str):
    """El `except Exception` exterior re-wrappea a 500. Para que el 404
    del ownership check propague, debe existir `except HTTPException:
    raise` ANTES del catch genérico.
    """
    pattern = re.compile(
        r"except\s+HTTPException\s*:\s*[\r\n]+(?:\s*#[^\r\n]*[\r\n]+)*\s*raise\b",
    )
    assert pattern.search(restock_body), (
        "P0-NEW-1 regresión: falta `except HTTPException: raise` antes "
        "del `except Exception as e` final. Sin este guard, el 404 del "
        "ownership check se re-wrappea a 500 vía `safe_error_detail(e)` "
        "y el cliente/test pierde la señal del IDOR."
    )


def test_restock_endpoint_anchor_present(restock_body: str):
    """Anchor textual `P0-NEW-1` en el comentario del SELECT para que un
    `grep` rápido localice el fix. Mismo patrón que P0-HIST-IDOR-1/2.
    """
    assert "P0-NEW-1" in restock_body, (
        "P0-NEW-1 regresión: el anchor textual `P0-NEW-1` desapareció "
        "del cuerpo de `api_restock`. Si alguien refactorizó los "
        "comentarios, restaurar la referencia para que `grep -r P0-NEW-1` "
        "sigue mapeando al fix."
    )
