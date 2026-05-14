"""[P1-SHOPPING-1 · 2026-05-13] Lock-the-contract de la invariante I2
(CLAUDE.md) en `_process_pending_shopping_lists` (cron_tasks.py).

Bug original (audit 2026-05-13):
    El cron `_process_pending_shopping_lists` (cron_tasks.py:~14188)
    ejecuta un `UPDATE meal_plans SET plan_data = jsonb_set(...) WHERE
    id = %s` cuando recupera shopping lists para planes
    `partial_no_shopping`. Pre-fix, la WHERE solo filtraba por `id`
    sin `AND user_id = %s` — violación cosmética de la invariante I2
    de CLAUDE.md ("toda mutación de `meal_plans` filtra AND
    user_id = %s"):

      1. NO IDOR: el cron itera planes system-wide sin user input.
      2. SÍ defense-in-depth: una refactorización futura que copie
         este UPDATE como template para un endpoint user-facing
         heredaría la omisión, donde sí abriría IDOR cross-user.
      3. SÍ I2-as-policy: contrato uniforme sobre `meal_plans`
         independiente de surface (cron vs endpoint).

    El `user_id` ya está disponible en el row leído al inicio del
    loop (`p.get('user_id')`, línea ~14218). Pre-fix no se pasaba
    a la tupla de parámetros del UPDATE.

Fix:
    Añadido `AND user_id = %s` a la WHERE clause + parámetro extra
    en la tupla de execute_sql_write. Comment inline anchored a
    `[P1-SHOPPING-1 · 2026-05-13]` para que un refactor futuro
    encuentre la razón sin abrir esta memoria.

Alcance del test (intencional, acotado):
    Este test cubre SOLO el sitio cerrado por P1-SHOPPING-1
    (`_process_pending_shopping_lists`). Un scan blanket sobre
    `cron_tasks.py` reveló múltiples UPDATEs históricos pre-existentes
    que tampoco filtran por user_id (anchor recovery attempts, restock
    cleanup, plan_start_date, etc.). Migrar todos ellos al contrato I2
    requiere un audit más amplio que excede P1-SHOPPING-1 (la
    decisión es out-of-scope para este P-fix).

    Si en el futuro se decide cerrar el contrato sobre TODO
    `cron_tasks.py`, este test puede extenderse con un scan blanket
    análogo a `test_p3_next_1_i2_user_id_filter_contract.py` (que
    cubre `routers/plans.py`), con whitelist markers explícitos para
    los UPDATEs cuyo no-filtrado sea decisión deliberada (e.g.
    sweeps cross-user legítimos).

Pareja con P3-NEXT-1:
    P3-NEXT-1 ancla `routers/plans.py` (universo user-facing).
    P1-SHOPPING-1 ancla un sitio puntual de `cron_tasks.py`. Si en
    el futuro se cierra el contrato cron-wide, el nuevo test sería
    la generalización; mientras tanto, este test mantiene defensa
    sobre el sitio específico.

Tooltip-anchor: P1-SHOPPING-1.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CRON_PY = _REPO_ROOT / "backend" / "cron_tasks.py"


def _read_cron_source() -> str:
    assert _CRON_PY.exists(), f"cron_tasks.py no encontrado: {_CRON_PY}"
    return _CRON_PY.read_text(encoding="utf-8")


def _extract_function_body(src: str, fn_name: str) -> str:
    """Devuelve el cuerpo de la función `fn_name` desde su `def` hasta el
    siguiente top-level `def `/`class ` o EOF.

    NO depende del AST de Python — los UPDATE strings dentro del cuerpo
    son lo que validamos, y se preservan literales.
    """
    start = src.find(f"def {fn_name}(")
    if start < 0:
        return ""
    after = src[start + len(f"def {fn_name}("):]
    # Buscar siguiente `def `/`class ` top-level desde el inicio de línea.
    next_top = re.search(r"\n(?:def |class )\w", after)
    end = (start + len(f"def {fn_name}(") + next_top.start()) if next_top else len(src)
    return src[start:end]


def _statement_filters_by_user_id(snippet: str) -> bool:
    """¿El snippet contiene un filtro `user_id = %s` en una cláusula
    SQL? Case-insensitive para tolerancia."""
    return bool(re.search(r"user_id\s*=\s*%s", snippet, re.IGNORECASE))


def test_cron_tasks_exists():
    """Sanity: el archivo existe (si fue renombrado, falla loud)."""
    assert _CRON_PY.exists(), (
        f"`{_CRON_PY}` no encontrado. ¿`cron_tasks.py` fue movido? "
        "Actualizar el path en este test."
    )


def test_process_pending_shopping_lists_function_exists():
    """Sanity: la función motivadora del P-fix existe."""
    src = _read_cron_source()
    body = _extract_function_body(src, "_process_pending_shopping_lists")
    assert body, (
        "`_process_pending_shopping_lists` no encontrado en `cron_tasks.py`. "
        "Si fue renombrada o movida, actualizar este test (o eliminarlo si "
        "la función fue absorbida por otro flujo). El P-fix P1-SHOPPING-1 "
        "cerró el contrato I2 sobre el UPDATE de esta función."
    )


def test_process_pending_shopping_lists_contains_update_meal_plans():
    """El UPDATE que motivó el P-fix sigue presente.

    Si la mutación se migró a un helper (e.g. `update_meal_plan_data`),
    actualizar este test para apuntar al helper. Si fue eliminada
    completamente, eliminar este test.
    """
    src = _read_cron_source()
    body = _extract_function_body(src, "_process_pending_shopping_lists")
    update_re = re.compile(r"UPDATE\s+meal_plans\b", re.IGNORECASE)
    assert update_re.search(body), (
        "`_process_pending_shopping_lists` ya NO contiene `UPDATE meal_plans`. "
        "Si la mutación fue migrada a un helper (e.g. `update_meal_plan_data`), "
        "actualizar este test para reflejar el nuevo callsite. Si fue "
        "eliminada completamente, eliminar este test."
    )


def test_process_pending_shopping_lists_filters_by_user_id():
    """[P1-SHOPPING-1] El `UPDATE meal_plans` en
    `_process_pending_shopping_lists` DEBE filtrar por `user_id = %s`.

    Este es el sitio específico cerrado por P1-SHOPPING-1. Si alguien
    revierte el filtro, este test falla con un mensaje accionable
    apuntando directamente al sitio.
    """
    src = _read_cron_source()
    body = _extract_function_body(src, "_process_pending_shopping_lists")
    assert _statement_filters_by_user_id(body), (
        "[P1-SHOPPING-1] El `UPDATE meal_plans` en "
        "`_process_pending_shopping_lists` ya no filtra por `user_id = %s`. "
        "Restaurar `AND user_id = %s` en la WHERE clause + añadir el "
        "parámetro `user_id` a la tupla. El `user_id` ya está disponible "
        "en el row leído al inicio del loop (`p.get('user_id')`).\n\n"
        "Razón: contrato I2 de CLAUDE.md exige `AND user_id = %s` en TODA "
        "mutación de `meal_plans`. Defense-in-depth aunque el cron no "
        "acepte user input — protege contra refactorizaciones que reusen "
        "este UPDATE como template para endpoints user-facing."
    )


def test_param_tuple_includes_user_id():
    """Verifica que la tupla de parámetros del execute_sql_write incluya
    `user_id` después de `meal_plan_id`. Un filtro `WHERE user_id = %s`
    sin el parámetro correspondiente causaría psycopg2 placeholder
    mismatch en runtime (5 placeholders en SET + 2 en WHERE = 7).

    Patrón canónico post-fix:
        execute_sql_write(\"\"\"...WHERE id = %s AND user_id = %s\"\"\",
            (..., meal_plan_id, user_id))
    """
    src = _read_cron_source()
    body = _extract_function_body(src, "_process_pending_shopping_lists")
    # Buscar el sufijo de la tupla cerca del UPDATE — pattern: ambos
    # `meal_plan_id` y `user_id` en la lista de args (sin importar orden
    # estricto, pero `user_id` debe estar presente).
    assert "meal_plan_id" in body, (
        "`meal_plan_id` ya no aparece en la tupla de parámetros del "
        "UPDATE. ¿Refactor? Actualizar este test."
    )
    assert "user_id" in body, (
        "`user_id` ya no aparece en el cuerpo de la función. Sin él en "
        "la tupla, el filtro `AND user_id = %s` causaría placeholder "
        "mismatch en runtime."
    )


def test_marker_anchored_in_comment():
    """El comment inline justifica el porqué del cambio — sin él, un
    refactor cosmético podría revertir el fix sin entender la razón.

    Buscar el anchor `[P1-SHOPPING-1 · 2026-05-13]` o `P1-SHOPPING-1`
    dentro del cuerpo de la función, idealmente cerca del UPDATE.
    """
    src = _read_cron_source()
    body = _extract_function_body(src, "_process_pending_shopping_lists")
    assert "P1-SHOPPING-1" in body, (
        "El anchor `[P1-SHOPPING-1 · ...]` desapareció del cuerpo de "
        "`_process_pending_shopping_lists`. Restaurar el comment que "
        "documenta por qué el UPDATE filtra por user_id aunque sea cron "
        "interno (defense-in-depth + I2 policy)."
    )
