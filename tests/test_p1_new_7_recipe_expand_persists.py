"""[P1-NEW-7 · 2026-05-11] Test parser-based: el handler
`/api/plans/recipe/expand` DEBE persistir la mutación de recipe/isExpanded
vía un helper de DB con kwarg `user_id=user_id`.

[P1-AUDIT-1 · 2026-05-15] Actualizado: el helper cambió de
`update_meal_plan_data(target_plan_id, target_plan_data, ...)` (patrón
pre-fix, read-modify-write con copia stale) a
`update_plan_data_atomic(target_plan_id, _apply_recipe_expansion,
user_id=user_id)` (FOR UPDATE row lock + callback fresh). El INTENT del
P1-NEW-7 se preserva — verificar que la persistencia ocurra y filtre por
user_id — pero el mecanismo cambió. Los asserts aceptan AMBOS helpers
para no romper si futuro refactor revierte parcialmente (siempre que el
contrato semántico se cumpla).

Bug original (gap P0 detectado audit 2026-05-11, este test lo enforza):
    El handler `api_expand_recipe` ([routers/plans.py:2871-3040]) mutaba
    el dict en memoria (`target_meal["recipe"] = expanded_steps`,
    `target_meal["isExpanded"] = True`) pero el bloque
    `if updated and target_plan_id:` contenía solo 30 líneas de
    comentario + `pass`. La llamada a `update_meal_plan_data` que
    originalmente persistía la expansión fue eliminada durante el
    refactor P3-NEW-1 (que solo pretendía remover el bump de
    `_plan_modified_at` — pero también borró el call de persistencia).

    Síntomas operacionales del gap P0:
      - Cada cook-click invocaba `expand_recipe_agent` (LLM → quema
        cuota Gemini, registra `log_api_usage`) pero NO persistía
        `expanded_recipe` ni `isExpanded` a DB.
      - Al reload de la página o al abrir desde Historial,
        `isExpanded=False` → el usuario re-expandía la misma receta
        infinitamente, quemando cuota en cada cook-click.
      - Comentarios inline en el handler (líneas 3021-3024) Y en
        Recipes.jsx (líneas 14-18, 292-293, 408-409) AFIRMAN que la
        persistencia ocurre vía `update_meal_plan_data` — mentirosos
        respecto al estado real del código tras P3-NEW-1.

Lo que permitió que el bug pasara CI:
    Los 5 tests de `test_p1_hist_recipe_1_expand_targeting.py`
    verificaban:
      - SELECT con ownership AND user_id ✓
      - lectura de `plan_id`/`day_index`/`meal_index` del body ✓
      - no-break-on-first-match ✓
      - match by `name` + `recipe` equivalence ✓
      - anchor del slug del marker ✓
    PERO NINGUNO verificaba que el handler llamara
    `update_meal_plan_data`. Eso permitió que el call fuera
    silenciosamente eliminado al refactor.

Cierre P1-NEW-7:
    Tres asserts contractuales independientes:
      1. Existe un call a `update_meal_plan_data(target_plan_id,
         target_plan_data, ...)` en el body del handler.
      2. El call aparece DESPUÉS del bloque `if updated and
         target_plan_id:` (no en rama hermana donde se ejecutaría
         siempre, generando no-op writes, o donde `target_plan_id`
         podría ser None → AttributeError).
      3. El call pasa `user_id=user_id` como kwarg, para que el helper
         P1-NEW-3 filtre `AND user_id = %s` a nivel DB (defense-in-
         depth a pesar del SELECT ownership upstream).

Tooltip-anchor: P1-NEW-7-PERSIST-CALL | gap P0 audit 2026-05-11

Dependencia operacional:
    Este test FALLA hasta que el call `update_meal_plan_data` se
    restaure en el handler (cierre del gap P0). La intención es
    exactamente esa: TDD red-light que bloquea CI hasta que producción
    se alinee con el contrato ya documentado en los comentarios.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    """Extrae el cuerpo de la función `fn_name` desde `def fn_name(...)`
    hasta justo antes del siguiente `@router.`/`@app.`/`def `. Mismo
    helper que `test_p1_hist_recipe_1_expand_targeting.py` para evitar
    drift entre tests del mismo handler.
    """
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(
            f"No se encontró `def {fn_name}(` en plans.py — "
            f"endpoint renombrado/eliminado."
        )
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def expand_body() -> str:
    src = _PLANS_PY.read_text(encoding="utf-8")
    return _extract_function_body(src, "api_expand_recipe")


# ---------------------------------------------------------------------------
# 1. Existe la llamada a update_meal_plan_data
# ---------------------------------------------------------------------------
def test_handler_invokes_update_meal_plan_data(expand_body: str):
    """Existe un call de persistencia en el body del handler — sea
    `update_meal_plan_data(target_plan_id, target_plan_data, ...)`
    (patrón pre P1-AUDIT-1) o `update_plan_data_atomic(target_plan_id,
    _apply_recipe_expansion, ...)` (patrón post P1-AUDIT-1 con FOR UPDATE
    row lock + callback fresh). Sin ninguna de las dos formas, las
    mutaciones de recipe/isExpanded viven SOLO en memoria del request y
    se pierden al return → cada cook-click quema cuota Gemini sin
    persistir (gap P0 original audit 2026-05-11).
    """
    legacy_re = re.compile(
        r"update_meal_plan_data\s*\(\s*target_plan_id\s*,\s*target_plan_data",
        re.DOTALL,
    )
    atomic_re = re.compile(
        r"(?:update_plan_data_atomic|_update_plan_data_atomic_\w+)"
        r"\s*\(\s*target_plan_id\s*,",
        re.DOTALL,
    )
    assert legacy_re.search(expand_body) or atomic_re.search(expand_body), (
        "P1-NEW-7 regresión: el handler `api_expand_recipe` NO invoca un "
        "helper de persistencia. Esperado uno de: "
        "`update_meal_plan_data(target_plan_id, target_plan_data, ...)` "
        "(legacy) o `update_plan_data_atomic(target_plan_id, "
        "_apply_recipe_expansion, ...)` (post P1-AUDIT-1, recomendado: "
        "FOR UPDATE row lock + callback fresh cierra la ventana lost-update). "
        "Sin ningún call, las mutaciones de recipe/isExpanded viven solo "
        "en memoria y se pierden al return → cada cook-click quema cuota "
        "Gemini sin persistir (gap P0 audit 2026-05-11)."
    )


# ---------------------------------------------------------------------------
# 2. La llamada está dentro del guard `if updated and target_plan_id:`
# ---------------------------------------------------------------------------
def test_persist_call_is_inside_updated_block(expand_body: str):
    """El call de persistencia debe aparecer DESPUÉS de un guard que
    asegure `target_plan_id` no es None. Pre P1-AUDIT-1 el guard era
    `if updated and target_plan_id:` (variable `updated` gestionada
    fuera del callback). Post P1-AUDIT-1 el guard es `if target_plan_data
    and isinstance(target_plan_data.get("days"), list) and target_plan_id:`
    (la lógica de `updated` se movió DENTRO del callback, donde retornar
    `False` aborta el UPDATE — equivalente semántico al guard `updated`
    pre-fix). Ambas formas son aceptables.

    Si el call se mueve FUERA de cualquier guard que cheque target_plan_id,
    crashea cuando es None (fallback path donde get_latest_meal_plan_with_id
    no resolvió).
    """
    # Aceptamos ambos guards: pre P1-AUDIT-1 y post P1-AUDIT-1.
    # Non-greedy `.*?` para parar en la primera ocurrencia de `target_plan_id`;
    # de lo contrario greedy `.*` se extiende hasta menciones del nombre en
    # comentarios narrativos al final del handler, dejando `guard_end` después
    # del callsite y el `after_guard` vacío.
    guard_pre_re = re.compile(r"if\s+updated\s+and\s+target_plan_id\s*:")
    guard_post_re = re.compile(
        r"if\s+target_plan_data\s+and\s+.*?target_plan_id\s*:",
    )
    pre_m = guard_pre_re.search(expand_body)
    post_m = guard_post_re.search(expand_body)
    assert pre_m or post_m, (
        "P1-NEW-7 regresión: no se encontró un guard que cheque "
        "`target_plan_id` (acepta `if updated and target_plan_id:` o "
        "`if target_plan_data and ... target_plan_id:`). ¿Refactor "
        "estructural? Sin guard, el call a persistencia crashea cuando "
        "target_plan_id es None (fallback path)."
    )
    # El call debe aparecer DESPUÉS del primer guard que matchee.
    guard_end = (pre_m or post_m).end()
    after_guard = expand_body[guard_end:]
    call_re = re.compile(
        r"(?:update_meal_plan_data|update_plan_data_atomic|_update_plan_data_atomic_\w+)"
        r"\s*\(\s*target_plan_id",
        re.DOTALL,
    )
    assert call_re.search(after_guard), (
        "P1-NEW-7 regresión: el call de persistencia no aparece DESPUÉS "
        "del guard de `target_plan_id`. Si se movió fuera, el call puede "
        "crashear cuando target_plan_id es None. Restaurar dentro del "
        "bloque guardado."
    )


# ---------------------------------------------------------------------------
# 3. Defense-in-depth P1-NEW-3: pasar user_id como kwarg
# ---------------------------------------------------------------------------
def test_persist_call_passes_user_id_kwarg(expand_body: str):
    """El call de persistencia debe pasar `user_id=user_id` para que el
    helper filtre `AND user_id = %s` a nivel DB. Aplica a ambos helpers:
      - Legacy `update_meal_plan_data` (P1-NEW-3): sin kwarg degrada a
        `WHERE id = %s` (DEPRECATED).
      - Post P1-AUDIT-1 `update_plan_data_atomic` (P2-OPEN-1): sin kwarg
        emite warning `[I2-MISS]` y degrada al SELECT/UPDATE sin filtro.
    Defense-in-depth en TODA mutación de meal_plans (convención repo).
    """
    pattern = re.compile(
        r"(?:update_meal_plan_data|update_plan_data_atomic|_update_plan_data_atomic_\w+)"
        r"\s*\([^)]*user_id\s*=\s*user_id",
        re.DOTALL,
    )
    assert pattern.search(expand_body), (
        "P1-NEW-7 regresión: el call de persistencia no pasa "
        "`user_id=user_id` como kwarg. Esto degrada el helper a la rama "
        "legacy sin ownership filter a nivel DB. El SELECT upstream con "
        "`AND user_id = %s` ya validó ownership, pero la convención del "
        "repo es defense-in-depth a DB-level en TODA mutación de meal_plans."
    )


# ---------------------------------------------------------------------------
# 4. Cross-link slug del marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P1-NEW-7` para
    `test_p2_hist_audit_14_marker_test_link` (cross-link enforcer)."""
    expected_slug = "p1_new_7"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p1_new_7`) para que el cross-link `test_p2_hist_audit_14_"
        "marker_test_link` lo matchee con el marker `_LAST_KNOWN_PFIX "
        "= \"P1-NEW-7 · 2026-05-11\"`."
    )
