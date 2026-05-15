"""[P1-AUDIT-1 · 2026-05-15] Lock-the-contract: los 3 callsites restantes de
`update_meal_plan_data` (post P1-RECALC-LOSTUPDATE) migran a
`update_plan_data_atomic` (FOR UPDATE row lock + re-SELECT fresh +
callback + UPDATE en la misma transacción).

Cierra el follow-up natural documentado en `test_p1_recalc_lostupdate.py:
341-343` (la memoria P1-RECALC-LOSTUPDATE deja explícito el plan de
migrar los 3 callsites restantes en un P-fix siguiente). Audit
production-readiness 2026-05-15 lo reactivó como P1.

Callsites cubiertos:
    1. `routers/plans.py::api_expand_recipe`     (`/recipe/expand`)
    2. `proactive_agent.py::_trigger_week2_background_generation` (JIT week-2)
    3. `tools.py::execute_modify_single_meal`    (agent_tool helper invocado
                                                  por `@tool modify_single_meal`)

Pre-fix flow (común a los 3):
    t=0  SELECT plan_data sin lock (helper distinto en cada callsite).
    t=1  Mutación in-memory (`days[...].meals[...]`, `days[...]`, aggregated
         lists, etc.).
    t=2  `update_meal_plan_data(plan_id, plan_data, user_id=user_id)` que
         adquiere advisory lock (P1-NEXT-1) y hace UPDATE full-overwrite.

Ventana lost-update entre t=0 y t=2: si un endpoint hermano (`/swap-meal/persist`,
`/grocery-start-date`, `/{plan_id}/name`, `/recalculate-shopping-list`,
`_chunk_worker T2`) muta `plan_data` quirúrgico con su propio lock entre
nuestro SELECT y nuestro UPDATE, esa mutación se pierde silenciosamente.

Fix: `update_plan_data_atomic(plan_id, callback, user_id=...)` (P0-2,
db_plans.py) re-SELECTea plan_data FRESH bajo `SELECT … FOR UPDATE` row
lock e invoca el callback. El FOR UPDATE row lock conflicta con cualquier
UPDATE posterior: las mutaciones quirúrgicas concurrentes completan ANTES
del lock o esperan DETRÁS de nuestro UPDATE.

Drift detection (parser-based, los 3 callsites comparten el contrato):
    - Cada handler invoca `update_plan_data_atomic` (no `update_meal_plan_data`).
    - El call pasa `user_id=user_id` (defense-in-depth I2 / P2-OPEN-1).
    - Existe un callback definido como función anidada (`_apply_*`) que
      recibe `plan_data_fresh`.
    - Ninguno de los 3 contextos contiene `update_meal_plan_data(plan_id,
      ...)` activo (el patrón pre-fix está erradicado).

Whitelist:
    Ninguna prevista. Si un refactor futuro reemplaza el callback por
    `jsonb_set` quirúrgico, los asserts pueden simplificarse — pero
    entonces el riesgo lost-update está intrínsecamente cerrado y los
    asserts de `update_plan_data_atomic` pueden retirarse intencionalmente.

Cross-link convention (P2-HIST-AUDIT-14): el slug del marker
`P1-AUDIT-1` → `p1_audit_1` matchea este archivo
`test_p1_audit_1_update_meal_plan_data_lostupdate.py`. Test único para
los 3 callsites — comparten contrato semántico (mismo helper SSOT, mismo
shape de callback) y son cerrados en un solo P-fix.

Tooltip-anchor: P1-AUDIT-1-START | gap audit 2026-05-15
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"
_PROACTIVE_PY = _BACKEND_ROOT / "proactive_agent.py"
_TOOLS_PY = _BACKEND_ROOT / "tools.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    """Aísla el cuerpo de `def fn_name(...)` hasta el siguiente `def`,
    `@router.`, `@app.` o `@tool` top-level. Defensivo contra decoradores
    adyacentes y funciones anidadas — el rango incluye `def _bg_task`/`def
    _apply_*` (anidadas) que también pertenecen al cuerpo del handler.
    """
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    assert m is not None, (
        f"Función `{fn_name}` no encontrada — refactor del callsite? "
        f"Si renombró, actualizar este test."
    )
    start = m.start()
    next_marker = re.search(
        r"\n(?:@router\.|@app\.|@tool|def\s)",
        src[start + 1:],
    )
    end = (start + 1 + next_marker.start()) if next_marker else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def proactive_src() -> str:
    return _PROACTIVE_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _TOOLS_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures por callsite (cuerpos de handler aislados)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def recipe_expand_body(plans_src: str) -> str:
    return _extract_function_body(plans_src, "api_expand_recipe")


@pytest.fixture(scope="module")
def week2_body(proactive_src: str) -> str:
    return _extract_function_body(
        proactive_src, "_trigger_week2_background_generation"
    )


@pytest.fixture(scope="module")
def modify_meal_body(tools_src: str) -> str:
    return _extract_function_body(tools_src, "execute_modify_single_meal")


# ---------------------------------------------------------------------------
# Contrato común: 3 asserts por callsite.
#
# Cada bloque cubre el mismo trío de garantías:
#   A. El handler invoca `update_plan_data_atomic(...)`.
#   B. La invocación pasa `user_id=user_id` (defense-in-depth I2).
#   C. Existe un callback anidado `def _apply_*(plan_data_fresh)` que
#      corre la mutación bajo el FOR UPDATE row lock.
#   D. El call legacy `update_meal_plan_data(plan_id, ...)` ya NO aparece
#      activo (sólo en comentarios narrativos).
# ---------------------------------------------------------------------------


def _assert_uses_atomic_helper(body: str, callsite: str) -> None:
    pattern = re.compile(
        r"update_plan_data_atomic\s*\(\s*\w+\s*,",
    )
    # En `tools.py` el helper se importa con alias
    # `update_plan_data_atomic as _update_plan_data_atomic_tool`; aceptamos
    # ambas formas. Mismo para plans.py (alias `_update_plan_data_atomic_re`).
    alias_pattern = re.compile(
        r"_update_plan_data_atomic_\w+\s*\(\s*\w+\s*,",
    )
    assert pattern.search(body) or alias_pattern.search(body), (
        f"P1-AUDIT-1 regresión: el handler `{callsite}` no invoca "
        f"`update_plan_data_atomic(plan_id, ...)`. Sin este helper, el "
        f"handler usa el patrón pre-fix `update_meal_plan_data(plan_id, "
        f"plan_data, user_id=...)` que aplica la copia leída ANTES del "
        f"lock y reabre la ventana lost-update contra mutaciones "
        f"quirúrgicas concurrentes."
    )


def _assert_passes_user_id(body: str, callsite: str) -> None:
    """`update_plan_data_atomic(..., user_id=user_id)` o alias. Tolera
    saltos de línea entre args."""
    pattern = re.compile(
        r"(?:update_plan_data_atomic|_update_plan_data_atomic_\w+)"
        r"\s*\([^)]*user_id\s*=\s*user_id",
        re.DOTALL,
    )
    assert pattern.search(body), (
        f"P1-AUDIT-1 regresión: la invocación de `update_plan_data_atomic` "
        f"en `{callsite}` no pasa `user_id=user_id` como kwarg. Sin esto, "
        f"el helper emite log warning `[I2-MISS]` y degrada el SELECT/UPDATE "
        f"al `WHERE id` solo (P2-OPEN-1). Fix: añadir `user_id=user_id` a la "
        f"lista de kwargs."
    )


def _assert_defines_inner_callback(body: str, callsite: str) -> None:
    """Función anidada `def _<name>(plan_data...)` con `plan_data` o
    `plan_data_fresh` como primer arg."""
    pattern = re.compile(
        r"^\s+def\s+_\w+\s*\(\s*plan_data\w*\s*[:)]",
        re.MULTILINE,
    )
    assert pattern.search(body), (
        f"P1-AUDIT-1 regresión: no se encontró función anidada "
        f"`def _<name>(plan_data_fresh...)` dentro de `{callsite}`. El "
        f"callback es donde corre la mutación bajo el FOR UPDATE row lock — "
        f"si no existe, el handler está mutando plan_data fuera del lock. "
        f"Fix: definir `def _apply_*(plan_data_fresh: dict) -> dict:` con "
        f"la lógica de mutación, y pasarla a "
        f"`update_plan_data_atomic(plan_id, _apply_*, user_id=user_id)`."
    )


def _assert_no_legacy_call(body: str, callsite: str) -> None:
    """`update_meal_plan_data(plan_id, ...)` activo (no en comentario)
    debe estar erradicado en el handler. Strip comentarios primero — los
    bloques narrativos del fix referencian el helper legacy por nombre."""
    body_no_comments = re.sub(r"#[^\n]*", "", body)
    pattern = re.compile(
        r"update_meal_plan_data\s*\(\s*\w+",
    )
    matches = pattern.findall(body_no_comments)
    assert not matches, (
        f"P1-AUDIT-1 regresión: el handler `{callsite}` todavía contiene "
        f"un callsite activo `update_meal_plan_data(plan_id, ...)`. Eso "
        f"reintroduce el patrón pre-fix full-overwrite con copia stale, "
        f"reabriendo la ventana lost-update. Fix: borrar el callsite legacy "
        f"y usar exclusivamente `update_plan_data_atomic(plan_id, _apply_*, "
        f"user_id=user_id)`. Matches encontrados: {matches}"
    )


# ---------------------------------------------------------------------------
# 1. /recipe/expand
# ---------------------------------------------------------------------------
def test_recipe_expand_uses_atomic_helper(recipe_expand_body: str):
    _assert_uses_atomic_helper(recipe_expand_body, "api_expand_recipe")


def test_recipe_expand_passes_user_id(recipe_expand_body: str):
    _assert_passes_user_id(recipe_expand_body, "api_expand_recipe")


def test_recipe_expand_defines_inner_callback(recipe_expand_body: str):
    _assert_defines_inner_callback(recipe_expand_body, "api_expand_recipe")


def test_recipe_expand_no_legacy_call(recipe_expand_body: str):
    _assert_no_legacy_call(recipe_expand_body, "api_expand_recipe")


def test_recipe_expand_mutations_inside_callback(recipe_expand_body: str):
    """Las asignaciones `recipe`/`isExpanded` deben aparecer DENTRO del
    callback (entre `def _<...>:` y la llamada a `update_plan_data_atomic`).
    Si aparecen FUERA, están corriendo sobre la copia stale."""
    callback_def_re = re.compile(
        r"^\s+def\s+_\w+\s*\(\s*plan_data\w*\s*[:)]",
        re.MULTILINE,
    )
    helper_call_re = re.compile(
        r"(?:update_plan_data_atomic|_update_plan_data_atomic_\w+)\s*\(",
    )
    cb_m = callback_def_re.search(recipe_expand_body)
    call_m = helper_call_re.search(recipe_expand_body)
    assert cb_m and call_m
    assert cb_m.start() < call_m.start(), (
        "Callback debe definirse ANTES de la llamada al helper (Python "
        "scoping)."
    )
    # Asignaciones `["recipe"] = expanded_steps` y `["isExpanded"] = True`.
    recipe_assign_re = re.compile(
        r"\[\s*['\"](?:recipe|isExpanded)['\"]\s*\]\s*=",
    )
    matches = list(recipe_assign_re.finditer(recipe_expand_body))
    assert matches, (
        "No se encontraron asignaciones `['recipe'|'isExpanded'] = ...` "
        "en el handler — refactor inesperado, revisar."
    )
    outside = [
        m for m in matches
        if not (cb_m.start() < m.start() < call_m.start())
    ]
    assert not outside, (
        f"P1-AUDIT-1 regresión: encontradas {len(outside)} asignaciones a "
        f"`['recipe'|'isExpanded']` FUERA del callback. Estas corren sobre "
        f"la copia stale leída fuera del lock — fix incompleto. Mover "
        f"DENTRO del callback `_apply_recipe_expansion`."
    )


# ---------------------------------------------------------------------------
# 2. proactive_agent week-2
# ---------------------------------------------------------------------------
def test_week2_uses_atomic_helper(week2_body: str):
    _assert_uses_atomic_helper(week2_body, "_trigger_week2_background_generation")


def test_week2_passes_user_id(week2_body: str):
    _assert_passes_user_id(week2_body, "_trigger_week2_background_generation")


def test_week2_defines_inner_callback(week2_body: str):
    _assert_defines_inner_callback(
        week2_body, "_trigger_week2_background_generation"
    )


def test_week2_no_legacy_call(week2_body: str):
    _assert_no_legacy_call(week2_body, "_trigger_week2_background_generation")


def test_week2_days_append_inside_callback(week2_body: str):
    """La asignación `plan_data_fresh["days"] = existing_days_fresh + new_days`
    debe ocurrir DENTRO del callback. Si está fuera, el append corre sobre
    `existing_plan_data` (parámetro leído por el cron) — esa copia puede
    estar stale por minutos/horas mientras run_plan_pipeline ejecuta."""
    callback_def_re = re.compile(
        r"^\s+def\s+_\w+\s*\(\s*plan_data\w*\s*[:)]",
        re.MULTILINE,
    )
    helper_call_re = re.compile(r"update_plan_data_atomic\s*\(")
    cb_m = callback_def_re.search(week2_body)
    call_m = helper_call_re.search(week2_body)
    assert cb_m and call_m
    days_assign_re = re.compile(
        r"\[\s*['\"]days['\"]\s*\]\s*=",
    )
    matches = list(days_assign_re.finditer(week2_body))
    assert matches, "No se encontraron asignaciones a `['days']` — refactor?"
    inside = [
        m for m in matches
        if cb_m.start() < m.start() < call_m.start()
    ]
    assert inside, (
        "P1-AUDIT-1 regresión: la asignación `['days'] = existing + new_days` "
        "está FUERA del callback en `_trigger_week2_background_generation`. "
        "Mover DENTRO del callback `_apply_week2_append` para que el append "
        "corra sobre `plan_data_fresh.days` (post-merge bajo el FOR UPDATE "
        "row lock) y no sobre `existing_plan_data` que el cron pasó "
        "minutos/horas atrás."
    )


# ---------------------------------------------------------------------------
# 3. tools.modify_single_meal
# ---------------------------------------------------------------------------
def test_modify_meal_uses_atomic_helper(modify_meal_body: str):
    _assert_uses_atomic_helper(modify_meal_body, "execute_modify_single_meal")


def test_modify_meal_passes_user_id(modify_meal_body: str):
    _assert_passes_user_id(modify_meal_body, "execute_modify_single_meal")


def test_modify_meal_defines_inner_callback(modify_meal_body: str):
    _assert_defines_inner_callback(modify_meal_body, "execute_modify_single_meal")


def test_modify_meal_no_legacy_call(modify_meal_body: str):
    _assert_no_legacy_call(modify_meal_body, "execute_modify_single_meal")


def test_modify_meal_aggregated_writes_inside_callback(modify_meal_body: str):
    """Las 4 asignaciones `plan_data_fresh["aggregated_shopping_list*"] = ...`
    deben aparecer DENTRO del callback. Si aparecen FUERA, escriben a la
    copia local `plan_data` que ya no se persiste — efectivamente perdidas."""
    callback_def_re = re.compile(
        r"^\s+def\s+_\w+\s*\(\s*plan_data\w*\s*[:)]",
        re.MULTILINE,
    )
    helper_call_re = re.compile(
        r"(?:update_plan_data_atomic|_update_plan_data_atomic_\w+)\s*\(",
    )
    cb_m = callback_def_re.search(modify_meal_body)
    call_m = helper_call_re.search(modify_meal_body)
    assert cb_m and call_m
    aggregated_re = re.compile(
        r"plan_data_fresh\s*\[\s*['\"]aggregated_shopping_list"
        r"(?:_weekly|_biweekly|_monthly)?['\"]\s*\]\s*=",
    )
    matches = list(aggregated_re.finditer(modify_meal_body))
    # Aceptamos 0 matches solo si las variables aggr_* son None (fallback
    # silencioso); el comportamiento canónico es 4 matches DENTRO del callback.
    # Si hay matches, todas deben estar dentro.
    outside = [
        m for m in matches
        if not (cb_m.start() < m.start() < call_m.start())
    ]
    assert not outside, (
        f"P1-AUDIT-1 regresión: {len(outside)} asignaciones a "
        f"`plan_data_fresh['aggregated_shopping_list*']` FUERA del callback "
        f"en `modify_single_meal`. Mover DENTRO del callback "
        f"`_apply_meal_modification`."
    )


def test_modify_meal_no_aggregated_writes_to_local_plan_data(modify_meal_body: str):
    """La copia local `plan_data` NO debe recibir asignaciones a
    `aggregated_shopping_list*` — esas escrituras se perderían porque
    `plan_data` ya no se persiste (sólo `plan_data_fresh` lo hace via
    callback). Si quedan, el operador podría asumir que tienen efecto."""
    body_no_comments = re.sub(r"#[^\n]*", "", modify_meal_body)
    pattern = re.compile(
        r"\bplan_data\s*\[\s*['\"]aggregated_shopping_list",
    )
    matches = pattern.findall(body_no_comments)
    assert not matches, (
        f"P1-AUDIT-1 regresión: `plan_data['aggregated_shopping_list*'] = "
        f"...` activo en el handler. Esa escritura va a la copia local que "
        f"no se persiste — falsa sensación de mutación. Mover DENTRO del "
        f"callback como `plan_data_fresh[...] = ...`. Matches: {len(matches)}"
    )


# ---------------------------------------------------------------------------
# Cross-cutting: db_plans.update_plan_data_atomic sigue siendo SSOT
# ---------------------------------------------------------------------------
def test_helper_signature_intact():
    """El helper `update_plan_data_atomic(plan_id, mutator, *, user_id=None)`
    sigue exportado con la signature que el bundle test asume. Si renombra
    o cambia el shape del callback, este test falla loud."""
    db_plans_src = (_BACKEND_ROOT / "db_plans.py").read_text(encoding="utf-8")
    sig_re = re.compile(
        r"def\s+update_plan_data_atomic\s*\(\s*"
        r"plan_id\s*:\s*str\s*,\s*"
        r"mutator\s*,",
        re.DOTALL,
    )
    assert sig_re.search(db_plans_src), (
        "P1-AUDIT-1 regresión: signature de `update_plan_data_atomic` "
        "cambió. Si renombró `mutator` o reordenó args, actualizar los 3 "
        "callsites + este test."
    )
    # `user_id` keyword-only (tras `*`).
    user_id_kwonly_re = re.compile(
        r"def\s+update_plan_data_atomic\s*\([^)]*\*\s*,[^)]*user_id\s*:",
        re.DOTALL,
    )
    assert user_id_kwonly_re.search(db_plans_src), (
        "P1-AUDIT-1 regresión: `user_id` ya no es keyword-only en "
        "`update_plan_data_atomic`. Si se relajó, los callsites pueden "
        "olvidar el kwarg silenciosamente. Restaurar `*, user_id=...`."
    )
