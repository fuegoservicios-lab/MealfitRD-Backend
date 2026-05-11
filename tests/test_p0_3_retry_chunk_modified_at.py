"""[P0-3 · 2026-05-10] Regression guard: `retry-chunk` sella
`plan_data._plan_modified_at` además del `updated_at` físico.

Bug original (audit 2026-05-10):
    `[backend/routers/plans.py:4138-4143]` retry-chunk hacía
    `UPDATE meal_plans SET plan_data = jsonb_set(..., generation_status,
    'partial'), updated_at = NOW()`. La columna `updated_at` no existía
    (cerrado por P0-2), pero AÚN tras P0-2 quedaba un gap:
    `plan_data._plan_modified_at` (SSOT semántico de "última edición
    del contenido del plan") NO se actualizaba.

    Consecuencia visible al usuario: el Historial (P1-HIST-4) ordena por
    `_plan_modified_at` extraído del jsonb client-side. Un usuario que
    retryea un chunk fallido a las 14:00 veía su plan ordenado bajo
    otros planes intactos creados el mismo día — la acción reciente NO
    burbujeaba arriba aunque el plan SÍ había mutado.

Fix:
    `jsonb_set` anidado en el UPDATE: además de `generation_status`,
    setea `_plan_modified_at = NOW()::text`. Mismo patrón que
    `cron_tasks.py:370-381` (legacy chunk learning persist) y que los
    cierres P1-HIST-AUDIT-1 (restore) y P1-HIST-AUDIT-2 (rename).

Cobertura de este test (parser-based, no DB):
    1. El endpoint `retry-chunk` contiene `'{_plan_modified_at}'` en su
       UPDATE de meal_plans.
    2. El path completo `jsonb_set(..., to_jsonb(NOW()::text), ...)`
       está presente (forma idiomática del repo).
    3. La columna física `updated_at = NOW()` SIGUE explícita
       (defense-in-depth; el trigger P0-2 la cubriría, pero el SET
       documenta la intención del callsite).

Out of scope (gaps para P-fixes posteriores):
    - Otros mutators no enumerados aún por P1-HIST-AUDIT-* (ej.
      pause-chunk, cancel-chunk, dead-letter-resolve) podrían tener el
      mismo gap. Auditarlos individualmente.
    - `swap_meal` (agent.py) NO toca plan_data persistido en este path
      (solo retorna el plato alternativo). Decisión P2-2 documentada.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_ROUTER_PATH = _BACKEND_ROOT / "routers" / "plans.py"


def _extract_retry_chunk_block() -> str:
    """Devuelve el cuerpo del endpoint retry-chunk como string.

    Estrategia: regex desde `@router.post("/{plan_id}/retry-chunk/...")`
    hasta el siguiente `@router.` o el final del archivo. Robusto frente
    a refactors que añadan comentarios o helpers entre medias.
    """
    src = _PLANS_ROUTER_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"@router\.post\(\"/\{plan_id\}/retry-chunk/\{chunk_id\}\".*?(?=@router\.|\Z)",
        src, re.DOTALL,
    )
    assert match is not None, (
        f"Endpoint retry-chunk no encontrado en {_PLANS_ROUTER_PATH}. "
        f"¿Fue renombrado/movido? Actualizar este test."
    )
    return match.group(0)


def test_retry_chunk_sets_plan_modified_at_in_jsonb():
    """El UPDATE de meal_plans debe incluir `'{_plan_modified_at}'`
    como path de jsonb_set."""
    block = _extract_retry_chunk_block()
    assert "'{_plan_modified_at}'" in block, (
        "retry-chunk NO setea `_plan_modified_at` en su UPDATE de "
        "meal_plans. Sin esto, el Historial no reordena al plan tras "
        "el retry — la acción del usuario queda invisible.\n\n"
        "Patrón esperado (mirror de cron_tasks.py:370-381):\n"
        "    UPDATE meal_plans\n"
        "    SET plan_data = jsonb_set(\n"
        "            jsonb_set(plan_data, '{generation_status}', '\"partial\"'),\n"
        "            '{_plan_modified_at}',\n"
        "            to_jsonb(NOW()::text),\n"
        "            true\n"
        "        ),\n"
        "        updated_at = NOW()\n"
        "    WHERE id = %s AND user_id = %s"
    )


def test_retry_chunk_uses_to_jsonb_now_idiom():
    """La forma `to_jsonb(NOW()::text)` es la canónica del repo para
    embeber timestamps en jsonb (vs. literal strings que pierden zona
    horaria). Si alguien usa otra forma (`'now()'`, `current_timestamp`),
    queda como divergencia detectable."""
    block = _extract_retry_chunk_block()
    assert re.search(r"to_jsonb\(\s*NOW\(\)\s*::\s*text\s*\)", block, re.IGNORECASE), (
        "retry-chunk debe usar `to_jsonb(NOW()::text)` para el "
        "`_plan_modified_at`. Patrón canónico (cron_tasks.py:379)."
    )


def test_retry_chunk_keeps_explicit_updated_at_set():
    """Defense-in-depth: aunque el trigger P0-2 cubre `updated_at`,
    el SET explícito documenta la intención del callsite y previene
    olvidos si alguien dropea el trigger en el futuro."""
    block = _extract_retry_chunk_block()
    assert re.search(r"updated_at\s*=\s*NOW\(\)", block, re.IGNORECASE), (
        "retry-chunk debe mantener `updated_at = NOW()` explícito en "
        "el UPDATE. El trigger P0-2 lo cubriría, pero la presencia "
        "explícita es contrato del callsite."
    )


def test_retry_chunk_jsonb_set_nested_correctly():
    """El jsonb_set debe estar ANIDADO (jsonb_set dentro de jsonb_set),
    no concatenado con coma — esto preserva atómicamente la mutación
    de ambas keys (`generation_status` y `_plan_modified_at`) en un
    único UPDATE."""
    block = _extract_retry_chunk_block()
    # Cuenta cuántos `jsonb_set(` aparecen en el UPDATE de meal_plans.
    # Esperamos al menos 2 (anidados).
    update_block_match = re.search(
        r"UPDATE\s+meal_plans.*?WHERE", block, re.DOTALL | re.IGNORECASE,
    )
    assert update_block_match is not None, "UPDATE meal_plans no encontrado en retry-chunk."
    update_block = update_block_match.group(0)
    jsonb_set_count = len(re.findall(r"\bjsonb_set\s*\(", update_block))
    assert jsonb_set_count >= 2, (
        f"retry-chunk debe usar jsonb_set ANIDADO (2+ llamadas en el mismo "
        f"UPDATE). Encontrado: {jsonb_set_count}. Si solo hay 1, una de "
        f"las dos keys (`generation_status` o `_plan_modified_at`) "
        f"NO se está seteando — la mutación es parcial."
    )
