"""[P2-NEW-1/3/4/5/6 · 2026-05-11] Tests consolidados para los 5 P2
restantes (P2-NEW-7 tiene archivo propio por convención del marker
slug, P2-NEW-2 tiene su propio drift-detection test).

Cubre:
    - P2-NEW-1: visibilitychange en History.jsx respeta TTL local del cache.
    - P2-NEW-3: Recipes.jsx invalida historyCaches tras /recipe/expand.
    - P2-NEW-4: Pantry.jsx pre-check plan freshness antes de recalc.
    - P2-NEW-5: tabla meal_plans_audit definida en migración.
    - P2-NEW-6: cron _gc_orphan_conversation_summaries registrado.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# P2-NEW-1: visibilitychange respeta TTL del cache (no invalida en cada tick)
# ---------------------------------------------------------------------------
def test_p2_new_1_visibilitychange_respects_cache_ttl():
    """History.jsx visibilitychange handler debe SOLO invalidar caches
    del plan abierto si (a) _dirty=true (señal explícita), O (b)
    _stale >= _CACHE_TTL_MS (30min). Sin esto, cada alt-tab ≥60s
    desperdiciaba cuota refetcheando 4 endpoints pesados aunque el
    plan no hubiera cambiado."""
    history_fp = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
    src = history_fp.read_text(encoding="utf-8")
    # El handler debe declarar _CACHE_TTL_MS y usarlo en condición.
    assert "_CACHE_TTL_MS" in src, (
        "P2-NEW-1 regresión: _CACHE_TTL_MS no definido en History.jsx. "
        "Sin alineamiento explícito con TTL del cache singleton, el "
        "visibilitychange invalida en cada alt-tab > 60s."
    )
    assert re.search(
        r"_shouldInvalidateModalCaches\s*=\s*_dirty\s*\|\|\s*_stale\s*>=\s*_CACHE_TTL_MS",
        src,
    ), (
        "P2-NEW-1 regresión: condición `_dirty || _stale >= _CACHE_TTL_MS` "
        "no encontrada. Sin esa condición, los caches se invalidan en "
        "cada visibilitychange con _stale>60s."
    )


# ---------------------------------------------------------------------------
# P2-NEW-3: Recipes.jsx invalida caches tras recipe/expand exitoso
# ---------------------------------------------------------------------------
def test_p2_new_3_recipes_imports_invalidatecachesforplan():
    """Recipes.jsx debe importar `invalidateCachesForPlan` del helper SSOT."""
    recipes_fp = _REPO_ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"
    src = recipes_fp.read_text(encoding="utf-8")
    assert re.search(
        r"import\s*\{[^}]*invalidateCachesForPlan[^}]*\}\s*from\s*['\"][^'\"]*historyCaches['\"]",
        src,
    ), (
        "P2-NEW-3 regresión: Recipes.jsx ya no importa "
        "`invalidateCachesForPlan` desde `../utils/historyCaches`. "
        "Sin esto, el modal del Historial muestra receta pre-expand "
        "hasta 30min tras un cook-click."
    )


def test_p2_new_3_recipes_invokes_invalidate_after_expand():
    """Recipes.jsx debe llamar `invalidateCachesForPlan(planId)` tras
    `/recipe/expand` exitoso, antes de cerrar el handler."""
    recipes_fp = _REPO_ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"
    src = recipes_fp.read_text(encoding="utf-8")
    # Buscar el bloque post-expand exitoso.
    success_idx = src.find("data.expanded_recipe")
    finally_idx = src.find("setIsExpanding(false)", success_idx)
    assert success_idx > 0 and finally_idx > 0
    success_block = src[success_idx:finally_idx]
    assert "invalidateCachesForPlan(planId)" in success_block, (
        "P2-NEW-3 regresión: `invalidateCachesForPlan(planId)` no se "
        "invoca tras `/recipe/expand` exitoso. El cache stale persiste "
        "30min."
    )


# ---------------------------------------------------------------------------
# P2-NEW-4: Pantry pre-check plan freshness antes del recalc
# ---------------------------------------------------------------------------
def test_p2_new_4_pantry_prefetch_plan_freshness():
    """Pantry.jsx `_recalcShoppingListAfterPantryChange` debe prefetch
    el plan actual del usuario ANTES de invocar el recalc — si el
    plan cambió en background, refrescar localStorage primero."""
    pantry_fp = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"
    src = pantry_fp.read_text(encoding="utf-8")
    # Buscar la función y validar pre-check.
    fn_idx = src.find("_recalcShoppingListAfterPantryChange")
    fetch_idx = src.find("recalculate-shopping-list", fn_idx)
    assert fn_idx > 0 and fetch_idx > 0
    section = src[fn_idx:fetch_idx]
    # Debe haber un fetch del plan actual ANTES del fetch al recalc.
    # [P1-NEON-DB-MIGRATION · 2026-06-12] El prefetch ya no es SELECT
    # directo a `meal_plans` con supabase-js (PostgREST prohibido en el
    # frontend tras el cutover a Neon): ahora va por el endpoint backend
    # GET /api/plans-data/latest (routers/user_data.py), mismo shape
    # {id, updated_at, plan_data}. La propiedad protegida es idéntica:
    # leer el plan latest del usuario antes de aplicar el recalc.
    assert re.search(
        r"/api/plans-data/latest",
        section,
    ), (
        "P2-NEW-4 regresión: Pantry no prefetchea el plan latest "
        "(`GET /api/plans-data/latest`) antes del recalc. Si el plan "
        "cambió en background (shift_plan), el householdSize/"
        "groceryDuration enviado es del plan viejo."
    )
    assert "P2-NEW-4" in section, (
        "P2-NEW-4 regresión: marker P2-NEW-4 no presente en el bloque "
        "del prefetch — defensa contra refactor que mueva el guard sin "
        "preservar la intención."
    )


def test_p2_new_4_recalc_request_includes_plan_id():
    """El body del fetch al recalc debe incluir `plan_id` opcional —
    backend puede log/rechazar drift si recibe id que ya no es latest."""
    pantry_fp = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"
    src = pantry_fp.read_text(encoding="utf-8")
    recalc_idx = src.find("recalculate-shopping-list")
    assert recalc_idx > 0
    # Buscar `plan_id` en una ventana alrededor del fetch al recalc.
    # [stale-parser fix · 2026-06-16] El body se extrajo a un const
    # `recalcBody = JSON.stringify({... plan_id: planData?.id })` que se
    # construye ANTES del fetch URL (líneas ~936-949) y se pasa como
    # `body: recalcBody`. Antes el `JSON.stringify` venía inline DESPUÉS
    # de la URL, así que la ventana solo miraba hacia adelante. Ahora la
    # ventana cubre ±1200 chars para capturar el body declarado antes del
    # fetch — la propiedad protegida (request incluye plan_id) es idéntica.
    body_block = src[max(0, recalc_idx - 1200):recalc_idx + 1200]
    assert "plan_id" in body_block, (
        "P2-NEW-4 regresión: request al recalc ya no incluye `plan_id`. "
        "Sin él, backend no puede detectar drift entre client local "
        "plan y server latest."
    )


# ---------------------------------------------------------------------------
# P2-NEW-5: tabla meal_plans_audit definida en migración
# ---------------------------------------------------------------------------
def test_p2_new_5_meal_plans_audit_migration_exists():
    """Migración `p2_new_5_meal_plans_audit_table.sql` debe existir."""
    fp = _REPO_ROOT / "migrations" / "p2_new_5_meal_plans_audit_table.sql"
    assert fp.exists(), (
        "P2-NEW-5 regresión: migración meal_plans_audit ausente. "
        "El SOP P3-AUDIT-6 de CLAUDE.md depende de esta tabla."
    )


def test_p2_new_5_audit_table_has_required_columns():
    """La migración debe definir las columnas críticas del audit log."""
    fp = _REPO_ROOT / "migrations" / "p2_new_5_meal_plans_audit_table.sql"
    sql = fp.read_text(encoding="utf-8")
    required_cols = [
        "meal_plan_id", "plan_data_before", "action", "actor", "created_at",
    ]
    for col in required_cols:
        assert col in sql, (
            f"P2-NEW-5 regresión: columna `{col}` ausente en "
            "meal_plans_audit migration."
        )
    # Action enum cerrado via CHECK constraint.
    assert "CHECK" in sql and "corruption_repair" in sql, (
        "P2-NEW-5 regresión: la columna `action` ya no tiene CHECK "
        "constraint enumerando valores válidos."
    )


def test_p2_new_5_audit_table_has_rls_forced():
    """La tabla audit debe tener RLS habilitado + forced — solo
    service_role puede leer/escribir."""
    fp = _REPO_ROOT / "migrations" / "p2_new_5_meal_plans_audit_table.sql"
    sql = fp.read_text(encoding="utf-8")
    assert "ENABLE ROW LEVEL SECURITY" in sql, (
        "P2-NEW-5 regresión: RLS no habilitado en meal_plans_audit."
    )
    assert "FORCE ROW LEVEL SECURITY" in sql, (
        "P2-NEW-5 regresión: FORCE RLS no aplicado. Sin él, el "
        "postgres role bypasea RLS y el audit log podría leerse "
        "accidentalmente desde un endpoint mal configurado."
    )


# ---------------------------------------------------------------------------
# P2-NEW-6: GC orphan conversation_summaries cron
# ---------------------------------------------------------------------------
def test_p2_new_6_gc_function_defined():
    """`_gc_orphan_conversation_summaries` debe estar definido."""
    cron_fp = _REPO_ROOT / "backend" / "cron_tasks.py"
    src = cron_fp.read_text(encoding="utf-8")
    assert re.search(
        r"^def\s+_gc_orphan_conversation_summaries\s*\(",
        src,
        re.MULTILINE,
    ), (
        "P2-NEW-6 regresión: `_gc_orphan_conversation_summaries` ya "
        "no existe. Summaries con session_id IS NULL acumularían "
        "indefinidamente."
    )


def test_p2_new_6_gc_registered_in_scheduler():
    """El cron debe estar registrado en `register_plan_chunk_scheduler`
    con `id='gc_orphan_conversation_summaries'`."""
    cron_fp = _REPO_ROOT / "backend" / "cron_tasks.py"
    src = cron_fp.read_text(encoding="utf-8")
    assert re.search(
        r"id\s*=\s*[\"\']gc_orphan_conversation_summaries[\"\']",
        src,
    ), (
        "P2-NEW-6 regresión: cron no registrado con id estable. "
        "APScheduler no lo dispara → summaries huérfanas acumulan."
    )


def test_p2_new_6_gc_uses_knob_for_interval():
    """`MEALFIT_ORPHAN_SUMMARIES_GC_INTERVAL_HOURS` debe leerse."""
    cron_fp = _REPO_ROOT / "backend" / "cron_tasks.py"
    src = cron_fp.read_text(encoding="utf-8")
    assert re.search(
        r"_env_int\(\s*[\"\']MEALFIT_ORPHAN_SUMMARIES_GC_INTERVAL_HOURS[\"\']",
        src,
    ), (
        "P2-NEW-6 regresión: knob para interval no se lee — frecuencia "
        "hardcoded rompe configurabilidad."
    )


def test_p2_new_6_gc_kill_switch_knob():
    """Knob `MEALFIT_ORPHAN_SUMMARIES_GC_ENABLED` debe leerse — kill
    switch sin redeploy."""
    cron_fp = _REPO_ROOT / "backend" / "cron_tasks.py"
    src = cron_fp.read_text(encoding="utf-8")
    assert "MEALFIT_ORPHAN_SUMMARIES_GC_ENABLED" in src, (
        "P2-NEW-6 regresión: kill-switch knob removido."
    )


def test_p2_new_6_gc_filters_only_orphans():
    """El cron debe filtrar por `session_id IS NULL` — NO debe tocar
    summaries activos (con session_id válido)."""
    cron_fp = _REPO_ROOT / "backend" / "cron_tasks.py"
    src = cron_fp.read_text(encoding="utf-8")
    m = re.search(
        r"^def\s+_gc_orphan_conversation_summaries\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m
    body = m.group(1)
    assert re.search(r"session_id\s+IS\s+NULL", body, re.I), (
        "P2-NEW-6 regresión: cron no filtra `session_id IS NULL`. "
        "Riesgo: borraría summaries activos."
    )
