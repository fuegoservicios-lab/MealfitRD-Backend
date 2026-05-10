"""[P0-HIST-IDOR-1 · 2026-05-10] `POST /{plan_id}/retry-chunk/{chunk_id}`
DEBE chequear ownership del plan y filtrar todos los UPDATE por user_id.

Bug original (audit Historial 2026-05-10):
    El handler `api_retry_chunk` (plans.py:3923) ejecutaba tres
    `UPDATE` consecutivos sobre `plan_chunk_queue` y `meal_plans`
    filtrando solo por `plan_id`/`chunk_id` — cero validación de
    `verified_user_id == plan.user_id`. Cualquier usuario authenticated
    podía:
      1. resetear `plan_chunk_queue` por `(chunk_id, plan_id)` ajeno
         → forzar re-ejecución de chunks de otros usuarios.
      2. revivir todos los `cancelled` de un plan ajeno por `meal_plan_id`.
      3. mutar `meal_plans` con `WHERE id = %s` puro → setear
         `generation_status='partial'` en cualquier plan del sistema y
         forzar polling sobre planes de víctimas.
    Adicional: `verify_api_quota` cobra cuota al ATACANTE (no al dueño)
    → DOS amplificable contra cualquier plan ajeno.

Estrategia del test (parser estático, mismo patrón que
`test_p1_hist_audit_new_1_quota_gate_history.py` y
`test_p3_b_required_fields_js_parser_added.py`):
    1. Localizar la función `api_retry_chunk` en plans.py (signature
       + body) hasta el siguiente `def`/`@router`.
    2. Verificar SELECT ownership previo a los UPDATE.
    3. Verificar que LOS TRES UPDATE referencian `user_id` en su WHERE.
    4. Verificar HTTPException 404 si el ownership check falla.

Drift detection bidireccional:
    - Si alguien revierte el ownership check → falla
      `test_retry_chunk_has_ownership_select`.
    - Si alguien quita uno de los 3 filtros user_id → falla
      `test_retry_chunk_updates_filter_by_user_id`.
    - Si alguien renombra `api_retry_chunk` → falla con AssertionError
      explícita (anchor del slug en el filename).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    """Devuelve `def <fn>(` hasta el siguiente top-level `def `/`@router`/
    `@app`. No hace análisis AST; suficiente para verificar contenido
    textual del body.
    """
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(
            f"No se encontró `def {fn_name}(` en plans.py — el endpoint "
            f"fue renombrado/eliminado. Si el rename es intencional, "
            f"actualizar este test con el nuevo nombre. Si fue eliminado, "
            f"verificar que el patrón IDOR del audit 2026-05-10 no se "
            f"reintrodujo en otro endpoint."
        )
    start = m.start()
    # Buscar el siguiente top-level def/@router/@app a partir del cuerpo.
    next_def = re.search(
        r"\n(?:@router\.|@app\.|def\s)",
        src[start + 1:],
    )
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def retry_chunk_body() -> str:
    src = _PLANS_PY.read_text(encoding="utf-8")
    return _extract_function_body(src, "api_retry_chunk")


def test_retry_chunk_has_ownership_select(retry_chunk_body: str):
    """SELECT ownership PREVIO a cualquier UPDATE. Sin esto, los UPDATE
    pueden tocar filas ajenas aunque tengan filtros user_id (race entre
    auth check y UPDATE inexistente vuelve trivial el bypass).
    """
    # Patrón: SELECT ... FROM meal_plans WHERE id = %s AND user_id = %s
    # con whitespace/casing flexible. Multiline (DOTALL) para tolerar
    # SQL formateado en varias líneas.
    ownership_pattern = re.compile(
        r"SELECT[^;]*FROM\s+meal_plans\s+WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
        re.IGNORECASE | re.DOTALL,
    )
    assert ownership_pattern.search(retry_chunk_body), (
        "P0-HIST-IDOR-1 regresión: `api_retry_chunk` NO contiene un "
        "SELECT ownership previo (`SELECT id FROM meal_plans WHERE "
        "id = %s AND user_id = %s`). Sin este check explícito un "
        "atacante puede resetear chunks/mutar generation_status de "
        "planes ajenos. Mismo patrón requerido en DELETE /{plan_id} "
        "(plans.py:4380-4389) y /blocked_reasons (plans.py:3637-3645)."
    )


def test_retry_chunk_returns_404_on_ownership_mismatch(retry_chunk_body: str):
    """Cuando el SELECT ownership devuelve 0 rows, debe lanzar 404
    (no 403, para no filtrar la existencia del plan ajeno — mismo
    contrato que DELETE /{plan_id}).
    """
    # Buscar `if not <var>` seguido de `raise HTTPException(status_code=404`
    # dentro del cuerpo. Tolera distintos nombres de variable (owner,
    # owner_check, plan).
    pattern = re.compile(
        r"if\s+not\s+\w+\s*:\s*[\r\n]+\s*(?:#[^\r\n]*[\r\n]+\s*)*"
        r"raise\s+HTTPException\s*\(\s*status_code\s*=\s*404",
        re.IGNORECASE,
    )
    assert pattern.search(retry_chunk_body), (
        "P0-HIST-IDOR-1 regresión: tras el SELECT ownership, no se "
        "encontró el `raise HTTPException(status_code=404, ...)` "
        "para cuando el plan no existe o no pertenece al usuario. "
        "Devolver 403 leak la existencia del plan; usar 404 explícito."
    )


def test_retry_chunk_updates_filter_by_user_id(retry_chunk_body: str):
    """Los 3 UPDATE de `api_retry_chunk` deben filtrar por user_id
    (defense-in-depth incluso con ownership check previo: race entre
    check y UPDATE no debe permitir mutación parcial de filas ajenas).

    Forma aceptada por UPDATE:
      A) `AND user_id = %s` directo (para meal_plans).
      B) `AND meal_plan_id IN (SELECT id FROM meal_plans WHERE user_id = %s)`
         (para plan_chunk_queue, que no tiene columna user_id propia).
    """
    update_blocks = re.findall(
        r"UPDATE\s+(plan_chunk_queue|meal_plans)\s+SET[^;]*?(?=\"\"\"|''')",
        retry_chunk_body,
        re.IGNORECASE | re.DOTALL,
    )
    assert len(update_blocks) >= 3, (
        f"P0-HIST-IDOR-1 regresión: se esperaban 3 UPDATE en "
        f"`api_retry_chunk` (1× failed→pending, 1× cancelled→pending, "
        f"1× generation_status→partial). Encontrados: {len(update_blocks)}. "
        f"Si la lógica del retry cambió, actualizar este conteo."
    )

    # Re-extraer cada UPDATE con su WHERE completo para inspección.
    update_pattern = re.compile(
        r"UPDATE\s+(plan_chunk_queue|meal_plans)\s+SET.*?WHERE(.*?)(?:\"\"\"|''')",
        re.IGNORECASE | re.DOTALL,
    )
    matches = update_pattern.findall(retry_chunk_body)
    assert len(matches) >= 3, (
        f"P0-HIST-IDOR-1 regresión: parser no pudo extraer WHERE de "
        f"los 3 UPDATE. Encontrados: {len(matches)}. Probable cambio "
        f"de quoting en SQL — ajustar parser."
    )

    direct_user_id_re = re.compile(r"user_id\s*=\s*%s", re.IGNORECASE)
    subquery_user_id_re = re.compile(
        r"meal_plan_id\s+IN\s*\(\s*SELECT\s+id\s+FROM\s+meal_plans\s+WHERE\s+user_id\s*=\s*%s\s*\)",
        re.IGNORECASE | re.DOTALL,
    )

    for table, where_clause in matches:
        has_direct = bool(direct_user_id_re.search(where_clause))
        has_subq = bool(subquery_user_id_re.search(where_clause))
        assert has_direct or has_subq, (
            f"P0-HIST-IDOR-1 regresión: UPDATE `{table.strip()}` con "
            f"WHERE `{where_clause.strip()[:200]}...` NO filtra por "
            f"user_id (ni directo `AND user_id = %s` ni subquery "
            f"`AND meal_plan_id IN (SELECT id FROM meal_plans WHERE "
            f"user_id = %s)`). Defense-in-depth: si una race entre "
            f"el ownership check y este UPDATE permite que un chunk "
            f"huérfano se mute, perdemos la garantía de aislamiento "
            f"cross-tenant."
        )


def test_retry_chunk_authenticated_required(retry_chunk_body: str):
    """`if not verified_user_id: raise 401` — sin auth no se debería
    siquiera ejecutar el SELECT ownership. Mismo guard que tiene
    `api_delete_plan` (plans.py:4366-4367) y `api_restore_plan`
    (plans.py:4004-4005).
    """
    pattern = re.compile(
        r"if\s+not\s+verified_user_id\s*:\s*[\r\n]+\s*"
        r"raise\s+HTTPException\s*\(\s*status_code\s*=\s*401",
        re.IGNORECASE,
    )
    assert pattern.search(retry_chunk_body), (
        "P0-HIST-IDOR-1 regresión: falta el guard explícito "
        "`if not verified_user_id: raise HTTPException(401, ...)` "
        "al inicio del handler. `Depends(verify_api_quota)` puede "
        "devolver None si auth falló silently — sin el guard, el "
        "SELECT ownership con user_id=None devuelve 0 rows y caemos "
        "al 404, ofuscando el error real (debería ser 401)."
    )


def test_marker_anchor_present():
    """El nombre de este archivo contiene el slug del marker
    `P0-HIST-IDOR-1` para que `test_p2_hist_audit_14_marker_test_link`
    lo correlacione con `_LAST_KNOWN_PFIX` (app.py).
    """
    expected_slug = "p0_hist_idor_1_retry_chunk_ownership"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo de test debe contener el slug del "
        "P-fix para que `test_p2_hist_audit_14_marker_test_link` lo "
        "matchee con el marker `_LAST_KNOWN_PFIX` del app.py."
    )
