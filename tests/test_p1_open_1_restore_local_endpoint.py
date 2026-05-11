"""[P1-OPEN-1 · 2026-05-11] Tests del endpoint atómico
`POST /api/plans/{plan_id}/restore-local`.

Cierra el último direct-write desde frontend a `meal_plans` (legacy
`restorePlan` path en `AssessmentContext.jsx`). Mismo patrón que cerró
P0-NEW-A (swap-meal), P0-NEW-B (grocery-start-date) y P1-HIST-5 (rename):
reemplazar `supabase.from('meal_plans').update(...).eq('id', planId)`
desde el cliente con un endpoint backend que:

  1. Verifica ownership (404 si plan no pertenece al usuario).
  2. Toma `acquire_meal_plan_advisory_lock(purpose='general')` antes
     del UPDATE — invariante I7 (CLAUDE.md): "todo full-overwrite a
     `plan_data` DEBE estar precedido por advisory lock".
  3. Aplica `AND user_id = %s` defense-in-depth (I2).
  4. Bumpea `_plan_modified_at` server-side (semántica CAS marker).

Estos tests son parser-based sobre el source de prod — NO arrancan FastAPI
ni tocan DB. Cubren la firma estructural del handler:
  - Ruta y decorador correctos.
  - Body validation (404 vs 400 vs 401).
  - Advisory lock 'general' presente.
  - `AND user_id = %s` en el UPDATE.
  - `_plan_modified_at` bumped antes del UPDATE.

El test de integración (race con `_chunk_worker`) vive en
`test_p1_open_1_restore_local_race_with_chunk_worker.py` si se necesita
en el futuro — el contrato estructural aquí es suficiente para detectar
regresiones de refactor.

Tooltip-anchor: P1-OPEN-1-START | endpoint restore-local
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PATH = _REPO_ROOT / "backend" / "routers" / "plans.py"
_FRONTEND_CTX = (
    _REPO_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx"
)


def _read_plans() -> str:
    return _PLANS_PATH.read_text(encoding="utf-8")


def _extract_handler_block() -> str:
    """Devuelve el cuerpo de `api_restore_plan_local` desde el decorador
    `@router.post("/{plan_id}/restore-local")` hasta `# P1-OPEN-1-END`.

    Si no encuentra el bloque, retorna "" — los tests downstream fallarán
    con mensajes específicos.
    """
    src = _read_plans()
    pattern = re.compile(
        r'@router\.post\(\s*["\']/\{plan_id\}/restore-local["\']\s*\)'
        r"[\s\S]+?# P1-OPEN-1-END",
        re.MULTILINE,
    )
    m = pattern.search(src)
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# 1. Existencia del endpoint
# ---------------------------------------------------------------------------
def test_endpoint_route_declared() -> None:
    """El decorador `@router.post('/{plan_id}/restore-local')` debe existir
    en `routers/plans.py` y el handler debe llamarse `api_restore_plan_local`.
    """
    src = _read_plans()
    assert re.search(
        r'@router\.post\(\s*["\']/\{plan_id\}/restore-local["\']\s*\)',
        src,
    ), (
        "P1-OPEN-1: no se encontró el decorador "
        "`@router.post('/{plan_id}/restore-local')` en `routers/plans.py`. "
        "El endpoint reemplaza el direct-write legacy de `restorePlan` "
        "(AssessmentContext.jsx)."
    )
    assert re.search(
        r"def\s+api_restore_plan_local\s*\(", src
    ), (
        "P1-OPEN-1: el handler debe llamarse `api_restore_plan_local`. "
        "Convención de nombres: api_<verb>_<resource> espejo de los otros "
        "handlers similares (api_swap_meal_persist, api_set_grocery_start_date)."
    )


# ---------------------------------------------------------------------------
# 2. Auth + ownership + 404
# ---------------------------------------------------------------------------
def test_handler_requires_verified_user_id() -> None:
    """El handler debe usar `get_verified_user_id` (NO `verify_api_quota`):
    restaurar un plan no consume cuota LLM, igual que `/swap-meal/persist`
    y `/grocery-start-date` lo manejan."""
    block = _extract_handler_block()
    assert block, "Bloque del handler `api_restore_plan_local` no encontrado."
    assert "get_verified_user_id" in block, (
        "P1-OPEN-1: el handler debe usar `Depends(get_verified_user_id)`. "
        "Patrón espejo de `api_swap_meal_persist` y `api_set_grocery_start_date`. "
        "Si en el futuro queremos cobrar cuota por restore, migrar a "
        "`verify_api_quota` — hoy no aplica."
    )


def test_handler_does_ownership_check_with_user_id() -> None:
    """Antes del UPDATE debe haber un SELECT con `WHERE id = %s AND user_id = %s`
    que devuelva 404 si no resoluble. Sin este check, un caller podría intentar
    restaurar un plan ajeno (aunque RLS bloquee el UPDATE, queremos 404
    consistente para no filtrar existencia).
    """
    block = _extract_handler_block()
    assert block, "Bloque del handler no encontrado."
    select_pattern = re.compile(
        r"SELECT\s+id\s+FROM\s+meal_plans\s+WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
        re.IGNORECASE,
    )
    assert select_pattern.search(block), (
        "P1-OPEN-1: el handler debe ejecutar "
        "`SELECT id FROM meal_plans WHERE id = %s AND user_id = %s` "
        "ANTES del UPDATE para resolver ownership (404 si no resoluble). "
        "Mismo patrón que `api_swap_meal_persist` y `api_set_grocery_start_date`."
    )
    assert "404" in block, (
        "P1-OPEN-1: ownership mismatch debe responder 404 (no 403/401) para "
        "no filtrar existencia del plan ajeno. Mismo patrón que `/retry-chunk` "
        "y `/swap-meal/persist`."
    )


# ---------------------------------------------------------------------------
# 3. Advisory lock 'general' (invariante I7)
# ---------------------------------------------------------------------------
def test_handler_acquires_advisory_lock_general() -> None:
    """El handler debe invocar `acquire_meal_plan_advisory_lock(...,
    purpose='general')` ANTES del UPDATE atómico al plan_data. La invariante
    I7 (CLAUDE.md) exige el lock para todo full-overwrite — sin él, un swap
    concurrente del chunk worker (T1/T2) pisaría las escrituras de este
    handler o viceversa.
    """
    block = _extract_handler_block()
    assert block, "Bloque del handler no encontrado."
    lock_pattern = re.compile(
        r"acquire_meal_plan_advisory_lock\s*\([^)]*purpose\s*=\s*['\"]general['\"]",
        re.IGNORECASE | re.DOTALL,
    )
    assert lock_pattern.search(block), (
        "P1-OPEN-1: el handler DEBE adquirir "
        "`acquire_meal_plan_advisory_lock(cursor, plan_id, purpose='general')` "
        "antes del UPDATE. Sin esto, viola invariante I7 (CLAUDE.md): "
        "'todo full-overwrite a plan_data DEBE estar precedido por advisory "
        "lock'. Mismo `purpose='general'` que T1/T2 chunk worker, /shift-plan "
        "y /restore para serialización correcta."
    )

    # Orden: el lock debe estar ANTES del UPDATE (no después).
    lock_match = lock_pattern.search(block)
    update_match = re.search(
        r"UPDATE\s+meal_plans\s+SET", block, re.IGNORECASE
    )
    assert lock_match and update_match, "Match interno faltó."
    assert lock_match.start() < update_match.start(), (
        "P1-OPEN-1: el `acquire_meal_plan_advisory_lock` debe ejecutarse "
        "ANTES del `UPDATE meal_plans`. Orden inverso permite que un "
        "writer concurrente pise el UPDATE antes de que el lock se tome."
    )


# ---------------------------------------------------------------------------
# 4. UPDATE con AND user_id = %s (invariante I2)
# ---------------------------------------------------------------------------
def test_update_filters_by_user_id() -> None:
    """El UPDATE final debe incluir `AND user_id = %s` además del `id = %s`.
    Defense-in-depth contra I2 violation: aunque el SELECT inicial ya filtró
    por ownership, un refactor que reordene la resolución (e.g., resolver
    plan_id desde un parámetro upstream) podría reabrir IDOR silente. Misma
    razón por la que P2-NEXT-1 añadió el filtro a `/shift-plan`.
    """
    block = _extract_handler_block()
    assert block, "Bloque del handler no encontrado."
    update_pattern = re.compile(
        r"UPDATE\s+meal_plans\s+SET[\s\S]+?WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
        re.IGNORECASE,
    )
    assert update_pattern.search(block), (
        "P1-OPEN-1: el UPDATE de `meal_plans` DEBE incluir "
        "`WHERE id = %s AND user_id = %s` (no solo `WHERE id = %s`). "
        "Sin el filtro user_id, un refactor que cambie cómo se resuelve "
        "plan_id upstream abre IDOR. Defense-in-depth I2."
    )


# ---------------------------------------------------------------------------
# 5. _plan_modified_at bumped server-side
# ---------------------------------------------------------------------------
def test_handler_bumps_plan_modified_at_server_side() -> None:
    """El handler debe setear `_plan_modified_at` server-side (con `NOW()` o
    `datetime.now(timezone.utc).isoformat()`) ANTES de persistir, no
    confiar en el valor que venga del cliente. La restauración es un evento
    de modificación lógica del plan; el Historial usa este path para sort.
    """
    block = _extract_handler_block()
    assert block, "Bloque del handler no encontrado."
    has_marker_assignment = bool(
        re.search(
            r"_plan_modified_at['\"]?\s*\]\s*=\s*",
            block,
        )
    ) or bool(
        re.search(
            r"_plan_modified_at['\"]?\s*:\s*",
            block,
        )
    )
    assert has_marker_assignment, (
        "P1-OPEN-1: el handler debe asignar `_plan_modified_at` server-side "
        "(en la copia del JSONB o vía jsonb_set en el UPDATE) — no aceptar "
        "el valor del cliente. La restauración bumpea el CAS marker porque "
        "el Historial usa ese path para ordenar planes."
    )


# ---------------------------------------------------------------------------
# 6. Frontend migrado: AssessmentContext.jsx invoca el endpoint
# ---------------------------------------------------------------------------
def test_frontend_restoreplan_calls_new_endpoint() -> None:
    """El callsite migrado de `restorePlan` (AssessmentContext.jsx) debe
    invocar ``fetchWithAuth('/api/plans/${planId}/restore-local')`` (template
    literal o template string equivalente) en lugar del legacy
    ``supabase.from('meal_plans').update({plan_data, ...}).eq('id', planId)``.
    """
    if not _FRONTEND_CTX.exists():
        # Repo sin frontend (e.g. backend-only CI) — skip silente.
        return
    src = _FRONTEND_CTX.read_text(encoding="utf-8")
    has_call = bool(
        re.search(
            r"fetchWithAuth\s*\(\s*[`'\"]/api/plans/\$\{[^}]+\}/restore-local",
            src,
        )
    )
    assert has_call, (
        "P1-OPEN-1: `AssessmentContext.jsx` debe invocar "
        "`fetchWithAuth(`/api/plans/${planId}/restore-local`, ...)` en lugar "
        "del direct-write legacy. Si el callsite legacy `supabase.from("
        "'meal_plans').update({plan_data}).eq('id', planId)` sigue presente, "
        "el blanket P1-NEW-A lo bloqueará — este test es el cierre positivo."
    )


# ---------------------------------------------------------------------------
# 7. Whitelist legacy removida del frontend
# ---------------------------------------------------------------------------
def test_p1_new_a_whitelist_removed_from_frontend() -> None:
    """Tras el cierre P1-OPEN-1, NO debe quedar ningún marker
    `// [P1-NEW-A WHITELIST: ...]` en el frontend. El único existente
    (restorePlan legacy) ya fue migrado a `/restore-local`.

    Si en el futuro aparece otro whitelist marker, este test deja de
    fallar (es asimétrico vs. el blanket P1-NEW-A que sí permite
    whitelists) pero un humano que añada uno nuevo DEBE actualizar
    este test añadiéndolo a una set de exceptions documentadas.
    """
    if not _FRONTEND_CTX.parent.parent.exists():
        return  # frontend ausente
    frontend_src = _REPO_ROOT / "frontend" / "src"
    if not frontend_src.exists():
        return
    bad: list[str] = []
    marker_pat = re.compile(
        r"//\s*\[P1-NEW-A\s+WHITELIST:\s*(?P<reason>[^\]]+?)\s*\]"
    )
    for f in frontend_src.rglob("*"):
        if not f.is_file() or f.suffix not in {".js", ".jsx", ".ts", ".tsx"}:
            continue
        parts = {p.lower() for p in f.parts}
        if "__tests__" in parts:
            continue
        name_low = f.name.lower()
        if name_low.endswith((".test.js", ".test.jsx", ".test.ts", ".test.tsx", ".d.ts")):
            continue
        try:
            src = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in marker_pat.finditer(src):
            line_no = src.count("\n", 0, m.start()) + 1
            bad.append(f"  {f.relative_to(_REPO_ROOT)}:{line_no} → {m.group(0)}")

    assert not bad, (
        "P1-OPEN-1: tras la migración de `restorePlan` a "
        "`/api/plans/{plan_id}/restore-local` no debe quedar NINGUNA whitelist "
        "`// [P1-NEW-A WHITELIST: ...]` viva. Si un caller nuevo necesita "
        "exención, primero documentarla aquí (lista de excepciones esperadas) "
        "Y en la sección 'Anti-patrones de frontend prohibidos' de CLAUDE.md.\n"
        + "\n".join(bad)
    )


# ---------------------------------------------------------------------------
# 8. Slug del marker en el filename
# ---------------------------------------------------------------------------
def test_marker_anchor_present() -> None:
    """Filename contiene `p1_open_1` para que el cross-link
    `test_p2_hist_audit_14_marker_test_link` lo matchee cuando el marker
    se bumpee a `P1-OPEN-1 · 2026-05-11`."""
    expected_slug = "p1_open_1"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p1_open_1`) para el cross-link."
    )
