"""[P1-RESTOCK-LOSTUPDATE · 2026-05-30] `POST /api/plans/restock` debe persistir
`plan_data` con `update_plan_data_atomic` (SELECT … FOR UPDATE + mutator sobre
plan_data FRESH), NO con full-overwrite `supabase.table("meal_plans").update(
{"plan_data": plan_data})`.

Bug original (audit nevera + lista-de-compras 2026-05-30):
    `api_restock` leía `plan_data` a t=0 vía SELECT plano (sin FOR UPDATE ni
    advisory lock), lo mutaba in-memory (is_restocked / restocked_at_iso /
    restocked_items) y reescribía el JSONB ENTERO a t=2. Era la ÚLTIMA escritura
    full-overwrite de `plan_data` en routers/plans.py y la única violación de I7
    (CLAUDE.md) restante — todos los endpoints hermanos ya se habían migrado
    (/swap-meal/persist P1-SWAP-PERSIST-ATOMIC, /recalculate-shopping-list
    P1-RECALC-LOSTUPDATE, /restore-local P1-OPEN-1).

    Ventana lost-update REAL (no hipotética): entre el SELECT t=0 y el UPDATE t=2,
    `_chunk_worker` puede persistir `days[8..14]` de un plan multi-semana bajo
    advisory lock + el cron VISIÓN-C poda `restocked_items` vencidos vía jsonb_set.
    El full-overwrite a t=2 los CLOBBEA silenciosamente (pérdida de comidas
    generadas + reactivación de perecederos ya consumidos). RLS/ownership filtra
    IDOR pero NO lost-update (mismo user_id).

Fix:
    Migrar el persist a `update_plan_data_atomic(real_plan_id, _restock_mutator,
    user_id=user_id)`. El mutator aplica SOLO las 3 keys que /restock posee,
    MERGEANDO sobre el plan_data fresh re-leído bajo FOR UPDATE — preserva
    days/_chunk_lessons/aggregated_shopping_list* y prunes concurrentes. El
    `user_id=` preserva el filtro AND user_id=%s del SELECT+UPDATE internos
    (defensa-en-profundidad del ownership P0-NEW-1 + invariante I2).

Estrategia del test (parser estático): localizar `api_restock` y verificar:
    1. Usa `update_plan_data_atomic(real_plan_id, ..., user_id=user_id)`.
    2. NO reintroduce el full-overwrite `.update({"plan_data": ...})`.
    3. El mutator es PURO: no contiene IO/DB dentro de su cuerpo
       (no `supabase.table`, no `execute_sql_*`, no `update_plan_data_atomic`).
    4. El mutator merge-a `restocked_items` sobre el `fresh` (no sobre un
       snapshot t=0) y respeta el flag de self-heal.
    5. Anchors textuales presentes (P1-RESTOCK-LOSTUPDATE-START/END).
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
    assert m, f"No se encontró `def {fn_name}(` en plans.py — ¿renombrado?"
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def restock_body() -> str:
    src = _PLANS_PY.read_text(encoding="utf-8")
    return _extract_function_body(src, "api_restock")


def test_restock_uses_update_plan_data_atomic(restock_body: str):
    """El persist DEBE usar `update_plan_data_atomic(real_plan_id, _restock_mutator,
    user_id=user_id)`."""
    m = re.search(
        r"update_plan_data_atomic\s*\(\s*real_plan_id\s*,(?P<args>.*?)\)",
        restock_body,
        re.DOTALL,
    )
    assert m, (
        "P1-RESTOCK-LOSTUPDATE regresión: `api_restock` ya no llama a "
        "`update_plan_data_atomic(real_plan_id, ...)`. Si la persistencia "
        "volvió al full-overwrite, se reabre la ventana lost-update I7."
    )
    assert re.search(r"user_id\s*=\s*user_id", m.group("args")), (
        "P1-RESTOCK-LOSTUPDATE regresión: la llamada a update_plan_data_atomic "
        "NO pasa `user_id=user_id` → se pierde el filtro AND user_id=%s "
        "(defensa-en-profundidad del ownership P0-NEW-1 / invariante I2)."
    )


def test_restock_no_full_overwrite_of_plan_data(restock_body: str):
    """El persist NO debe reintroducir el full-overwrite del JSONB completo."""
    assert not re.search(
        r"\.update\(\s*\{\s*[\"']plan_data[\"']\s*:", restock_body
    ), (
        "P1-RESTOCK-LOSTUPDATE regresión: reapareció "
        "`.update({\"plan_data\": ...})` (full-overwrite) en api_restock. "
        "Usar update_plan_data_atomic con mutator quirúrgico."
    )


def test_restock_mutator_is_pure_no_io(restock_body: str):
    """El mutator corre DENTRO del SELECT … FOR UPDATE (retiene row-lock + slot
    del pool). DEBE ser puro CPU-only (contrato P2-MUTATOR-PURITY): nada de
    supabase/execute_sql/re-entrada al helper atómico en su cuerpo. El cuerpo se
    delimita entre `def _restock_mutator` y su `return fresh` (última línea del
    mutator) — NO usar un regex de indentación que sobre-capture las líneas
    posteriores (la llamada a update_plan_data_atomic vive justo después)."""
    def_idx = restock_body.find("def _restock_mutator")
    assert def_idx != -1, "No se encontró `def _restock_mutator(...)` dentro de api_restock."
    ret_idx = restock_body.find("return fresh", def_idx)
    assert ret_idx != -1, "El mutator `_restock_mutator` no termina en `return fresh`."
    body = restock_body[def_idx:ret_idx]
    for forbidden in ("supabase.table", "execute_sql_write", "execute_sql_query",
                      "update_plan_data_atomic", ".execute()"):
        assert forbidden not in body, (
            f"P1-RESTOCK-LOSTUPDATE / P2-MUTATOR-PURITY regresión: el mutator "
            f"contiene `{forbidden}` — IO/DB dentro del FOR UPDATE retiene el "
            f"row-lock + slot del pool y puede agotar el pool sync. Resolver "
            f"datos externos ANTES de llamar update_plan_data_atomic."
        )


def test_restock_mutator_merges_restocked_items_on_fresh(restock_body: str):
    """El mutator debe MERGEAR `restocked_items` sobre el dict `fresh`
    (re-leído bajo FOR UPDATE), no reconstruir desde un snapshot t=0, y respetar
    el flag de self-heal."""
    def_idx = restock_body.find("def _restock_mutator")
    assert def_idx != -1, "No se encontró `def _restock_mutator(...)`."
    ret_idx = restock_body.find("return fresh", def_idx)
    assert ret_idx != -1, "El mutator no termina en `return fresh`."
    body = restock_body[def_idx:ret_idx]
    assert 'fresh["restocked_items"]' in body or "fresh['restocked_items']" in body, (
        "El mutator no escribe `fresh[\"restocked_items\"]` — debe operar sobre "
        "el plan_data fresh, no sobre el snapshot t=0."
    )
    assert "_restock_self_heal_reset" in body, (
        "El mutator no consulta `_restock_self_heal_reset` — el self-heal "
        "P3-RESTOCK-STALE-DEDUP debe replicarse sobre el fresh (arrancar "
        "restocked_items desde cero cuando la nevera estaba vacía)."
    )


def test_restock_lostupdate_anchors_present(restock_body: str):
    assert "P1-RESTOCK-LOSTUPDATE-START" in restock_body
    assert "P1-RESTOCK-LOSTUPDATE-END" in restock_body
