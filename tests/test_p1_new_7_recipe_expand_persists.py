"""[P1-NEW-7 · 2026-05-11] Test parser-based: el handler
`/api/plans/recipe/expand` DEBE invocar `update_meal_plan_data(
target_plan_id, target_plan_data, user_id=user_id)` tras mutar
`target_plan_data` en memoria.

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
    """Existe `update_meal_plan_data(target_plan_id, target_plan_data,
    ...)` en el body del handler. Sin esta llamada las mutaciones a
    recipe/isExpanded sobre el dict Python viven SOLO en memoria del
    request y se pierden al `return` → cada cook-click quema cuota
    Gemini sin persistir.
    """
    call_pattern = re.compile(
        r"update_meal_plan_data\s*\(\s*target_plan_id\s*,\s*target_plan_data",
        re.DOTALL,
    )
    assert call_pattern.search(expand_body), (
        "P1-NEW-7 regresión: el handler `api_expand_recipe` NO llama "
        "`update_meal_plan_data(target_plan_id, target_plan_data, ...)`. "
        "Sin esta llamada, las mutaciones de recipe/isExpanded sobre "
        "target_plan_data viven solo en el dict Python del request y "
        "se pierden al return → cada cook-click quema cuota Gemini sin "
        "persistir (gap P0 audit 2026-05-11). Restaurar la llamada "
        "DENTRO del bloque `if updated and target_plan_id:` antes del "
        "`# P1-HIST-RECIPE-1-END`."
    )


# ---------------------------------------------------------------------------
# 2. La llamada está dentro del guard `if updated and target_plan_id:`
# ---------------------------------------------------------------------------
def test_persist_call_is_inside_updated_block(expand_body: str):
    """El call a `update_meal_plan_data` debe aparecer DESPUÉS del
    `if updated and target_plan_id:`. Si se mueve fuera del guard:
      - Se ejecuta también cuando no hubo cambios → no-op write.
      - Si `target_plan_id is None` (fallback path donde el helper
        get_latest_meal_plan_with_id no resolvió) → AttributeError.
    """
    if_match = re.search(
        r"if\s+updated\s+and\s+target_plan_id\s*:",
        expand_body,
    )
    assert if_match, (
        "P1-NEW-7 regresión: no se encontró el bloque `if updated and "
        "target_plan_id:` en el handler. ¿Fue refactorizado? Si la "
        "estructura cambió, actualizar este test SIN perder la "
        "condición: la persistencia debe ejecutarse SOLO cuando hubo "
        "mutaciones que guardar (`updated`) y plan resuelto "
        "(`target_plan_id`)."
    )
    after_if = expand_body[if_match.end():]
    call_pattern = re.compile(
        r"update_meal_plan_data\s*\(\s*target_plan_id",
        re.DOTALL,
    )
    assert call_pattern.search(after_if), (
        "P1-NEW-7 regresión: el call a `update_meal_plan_data` no "
        "aparece DESPUÉS del `if updated and target_plan_id:` en el "
        "handler. Si se movió fuera del guard, el call ejecuta no-op "
        "writes cuando no hubo cambios, o crashea cuando "
        "target_plan_id es None (fallback path). Restaurar dentro del "
        "bloque."
    )


# ---------------------------------------------------------------------------
# 3. Defense-in-depth P1-NEW-3: pasar user_id como kwarg
# ---------------------------------------------------------------------------
def test_persist_call_passes_user_id_kwarg(expand_body: str):
    """El call debe pasar `user_id=user_id` para que el helper P1-NEW-3
    filtre `AND user_id = %s` a nivel DB. Sin el kwarg, el helper
    degrada a la rama legacy `WHERE id = %s` (documentado como
    DEPRECATED en db_plans.py:927 + warning de log).
    """
    pattern = re.compile(
        r"update_meal_plan_data\s*\([^)]*user_id\s*=\s*user_id[^)]*\)",
        re.DOTALL,
    )
    assert pattern.search(expand_body), (
        "P1-NEW-7 regresión: el call a `update_meal_plan_data` no pasa "
        "`user_id=user_id` como kwarg. Sin esto el helper degrada a la "
        "rama legacy `WHERE id = %s` sin ownership filter a nivel DB "
        "(documentado DEPRECATED en db_plans.py:927). El SELECT "
        "upstream con `AND user_id = %s` ya validó ownership, pero la "
        "convención del repo (P1-NEW-3) es defense-in-depth a DB-level "
        "en TODA mutación de meal_plans."
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
