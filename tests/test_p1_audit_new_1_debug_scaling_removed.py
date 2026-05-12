"""[P1-AUDIT-NEW-1 · 2026-05-12] El endpoint debug `/debug-scaling/{user_id}`
fue ELIMINADO de routers/plans.py. Este test enforza que no vuelva.

Bug original (audit comprehensivo 2026-05-12, gap P1):
    `@router.get("/debug-scaling/{user_id}")` vivía en producción marcado
    como "TEMPORARY DEBUG (REMOVE AFTER DIAGNOSIS)" sin auth. Dos defectos:

      1. Sin `Depends(get_verified_user_id)` ni `_verify_admin_token`:
         cualquiera con la URL podía leer `plan_data` de cualquier user_id.

      2. Fallback IDOR: si el user_id no tenía plan, ejecutaba
         `SELECT id, user_id, plan_data FROM meal_plans ORDER BY created_at
         DESC LIMIT 1` y retornaba el plan más reciente de OTRO usuario,
         incluyendo `found_user_id` ajeno. Enumerador trivial de UUIDs.

Cierre:
    Endpoint eliminado completo (cero callers cross-codebase). Bloque de
    comentario explicativo dejado en el sitio + reemplazo SRE-side por
    el script local `backend/check_scaling.py` que se ejecuta contra
    una DB readonly sin exponer endpoint HTTP. Si en el futuro hace falta
    inspección live de scaling, montar bajo `/admin/plans/...` con
    `_verify_admin_token` (mismo patrón que `/admin/chunks/stuck`) y SIN
    fallback cross-user.

Lo que este test enforza:
    A) El path `/debug-scaling` ya NO está registrado como ruta de FastAPI.
    B) La función `debug_scaling(` ya NO existe en routers/plans.py.
    C) El fallback IDOR `SELECT id, user_id, plan_data FROM meal_plans
       ORDER BY created_at DESC` no aparece en NINGÚN endpoint de
       routers/plans.py (incluso si alguien intentara recrear el patrón
       con otro nombre).
    D) Si alguien re-añade un endpoint `debug-*` SIN
       `_verify_admin_token`, el test falla con copy explicativo.

Tooltip-anchor: P1-AUDIT-NEW-1-DEBUG-ENDPOINT-REMOVED
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


def test_debug_scaling_route_removed(plans_src: str) -> None:
    """A) La ruta `/debug-scaling/...` no existe como decorador."""
    pattern = re.compile(
        r'@router\.(get|post|put|patch|delete)\(\s*["\']\/?debug-scaling',
        re.IGNORECASE,
    )
    match = pattern.search(plans_src)
    assert match is None, (
        "P1-AUDIT-NEW-1 violation: el endpoint `/debug-scaling/...` fue "
        "re-añadido a routers/plans.py. Este endpoint NO debe existir — "
        "tenía IDOR (sin auth + fallback que retornaba plan de otro user). "
        "Si necesitas inspección live de scaling, monta bajo "
        "`/admin/plans/scaling-inspect/{plan_id}` con `_verify_admin_token`. "
        f"Match encontrado: {match.group(0)!r}"
    )


def test_debug_scaling_function_removed(plans_src: str) -> None:
    """B) La función `debug_scaling(...)` no existe."""
    pattern = re.compile(r'^\s*def\s+debug_scaling\s*\(', re.MULTILINE)
    match = pattern.search(plans_src)
    assert match is None, (
        "P1-AUDIT-NEW-1 violation: la función `debug_scaling(...)` fue "
        "re-añadida a routers/plans.py. Ver bloque de comentario "
        "P1-AUDIT-NEW-1-DEBUG-ENDPOINT-REMOVED en el sitio original."
    )


def test_no_idor_fallback_select_meal_plans_unfiltered(plans_src: str) -> None:
    """C) Defensa-en-profundidad: el patrón SQL del fallback IDOR
    `SELECT id, user_id, plan_data FROM meal_plans ORDER BY created_at
    DESC LIMIT 1` no aparece en NINGUNA parte de routers/plans.py.

    Justificación: aunque el endpoint ya se eliminó, ese SQL específico
    (sin `WHERE user_id = %s`) es el vector que convertía el debug en
    un enumerador cross-user. Si alguien lo replica en otro handler,
    estaría reabriendo la misma clase de bug.

    NOTA: aceptamos `SELECT ... FROM meal_plans ORDER BY created_at`
    con filtro WHERE user_id antes — el regex exige que entre `FROM
    meal_plans` y `ORDER BY` NO haya un `WHERE`.
    """
    # Buscar `FROM meal_plans` seguido de `ORDER BY created_at` sin un
    # `WHERE` entre medio. Usamos lookahead negativo simple basado en
    # caracteres entre los dos anchors.
    pattern = re.compile(
        r'FROM\s+meal_plans\s+ORDER\s+BY\s+created_at\s+DESC\s+LIMIT\s+1',
        re.IGNORECASE,
    )
    matches = pattern.findall(plans_src)
    assert not matches, (
        "P1-AUDIT-NEW-1 violation: detectado el patrón SQL del fallback "
        "IDOR original: `FROM meal_plans ORDER BY created_at DESC LIMIT 1` "
        "sin filtro WHERE user_id. Este patrón retorna el plan más reciente "
        "de la flota completa — usable como enumerador cross-user. "
        "Si necesitas el plan más reciente DE UN USUARIO, filtra con "
        "`WHERE user_id = %s` (helper `get_latest_meal_plan_with_id(user_id)`). "
        f"Matches: {matches}"
    )


def test_debug_routes_require_admin_gate(plans_src: str) -> None:
    """D) Cualquier futuro endpoint `debug-*` o `/debug/...` debe estar
    admin-gateado o vivir bajo `/admin/...` (que ya es admin-gateado por
    convención). Captura intentos de re-introducir endpoints debug
    similares con otro nombre.

    Tolera 1 acepción: si el endpoint tiene `_verify_admin_token` en su
    cuerpo o en su signature dependencies, está OK. Si no, falla.
    """
    # Buscar todos los handlers cuyo path contenga "/debug" (case-insensitive)
    # PERO que NO empiecen por "/admin/" (los admin/* ya están convencionalmente
    # gateados — el test_p1_audit_3 enforza eso del lado de admin endpoints).
    debug_pattern = re.compile(
        r'@router\.(get|post|put|patch|delete)\(\s*["\'](\/?(?!admin\/)[^"\']*debug[^"\']*)["\']',
        re.IGNORECASE,
    )
    offenders = []
    for m in debug_pattern.finditer(plans_src):
        route = m.group(2)
        # Buscar el cuerpo del handler hasta la próxima función o
        # decorador de router → asegurar que llame `_verify_admin_token`.
        start = m.end()
        # Encuentra el siguiente `@router.` o `def ` top-level (~ 200 líneas máx)
        next_decorator = plans_src.find('\n@router.', start)
        next_top_def = re.search(r'\n(?:def|async def) \w+\(', plans_src[start:])
        next_top_def_pos = (start + next_top_def.start()) if next_top_def else len(plans_src)
        # Ventana razonable de búsqueda
        end = min(
            next_decorator if next_decorator != -1 else len(plans_src),
            next_top_def_pos if next_top_def_pos != -1 else len(plans_src),
            start + 5000,
        )
        body = plans_src[start:end]
        if '_verify_admin_token' not in body:
            offenders.append(route)

    assert not offenders, (
        "P1-AUDIT-NEW-1 violation: detectado(s) endpoint(s) con `debug` en "
        "el path SIN `_verify_admin_token` en el cuerpo del handler. "
        "Endpoints debug deben vivir bajo `/admin/...` o gatearse "
        "explícitamente con `_verify_admin_token` (mismo patrón que "
        "`/admin/chunks/stuck`). "
        f"Offending routes: {offenders}"
    )
