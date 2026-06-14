"""[P0-2 · 2026-05-10] Regression guard: `meal_plans.updated_at` existe
en el schema, mantenida por trigger BEFORE UPDATE.

Bug original (audit 2026-05-10):
    1. `[backend/routers/plans.py:4141]` (retry-chunk) ejecutaba
       `UPDATE meal_plans SET plan_data = jsonb_set(...), updated_at = NOW()`.
       La columna NO existía (`information_schema.columns` confirmaba
       11 columnas, ninguna `updated_at`). Cada retry-chunk del usuario
       lanzaba `column "updated_at" of relation "meal_plans" does not exist`
       y devolvía 500.
    2. Cron horario `_aggregate_coherence_block_history_metrics` (P3-B)
       golpeaba `?updated_at=gte=<cutoff>` vía PostgREST cliente → 400.
       Workaround P0-OBS-1 cambió el filtro a `created_at`, pero esto
       cerraba el síntoma sin arreglar la causa (la columna seguía
       faltando, cualquier futuro callsite reproduciría el bug).

Fix (migración `p0_2_meal_plans_updated_at.sql`):
    - ADD COLUMN `updated_at TIMESTAMPTZ NOT NULL DEFAULT now()`.
    - Backfill: filas existentes obtienen `updated_at = created_at`.
    - Trigger `trg_meal_plans_set_updated_at BEFORE UPDATE` la mantiene
      automáticamente — los callsites no necesitan acordarse de incluir
      `SET updated_at = NOW()`.
    - Índice `idx_meal_plans_user_updated_at` para window queries.

Cobertura de este test (parser-based, no DB):
    1. La migración SSOT existe y declara los 4 objetos (columna, función
       trigger, trigger, índice).
    2. retry-chunk en `routers/plans.py` sigue usando `updated_at = NOW()`
       (defense-in-depth: si alguien elimina el `SET updated_at = NOW()`
       confiando solo en el trigger, queda registrado en el test que la
       intención original era explícita).
    3. CLAUDE.md o la fila correspondiente en MEMORY.md referencia la
       migración (validado vía existencia del archivo migration con
       prefijo correcto — el cross-link a memory queda en P3-2).

Out of scope:
    - Validación schema runtime contra DB real: ese checkeo lo hace
      `test_p0_obs_1_coherence_metrics_uses_created_at.py` (snapshot del
      set de columnas reales, ya actualizado en P0-2 para incluir
      `updated_at`).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


# Raíz del repo: este test vive en backend/tests/, subir 2 niveles.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATION_PATH = _REPO_ROOT / "migrations" / "p0_2_meal_plans_updated_at.sql"
_PLANS_ROUTER_PATH = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo requerido no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. La migración existe y declara los 4 objetos canónicos.
# ---------------------------------------------------------------------------
def test_migration_file_exists():
    assert _MIGRATION_PATH.exists(), (
        f"Migración SSOT debe vivir en {_MIGRATION_PATH}. Si la moviste, "
        f"actualiza `_MIGRATION_PATH` aquí + el cross-link en MEMORY.md."
    )


def test_migration_declares_updated_at_column():
    sql = _read(_MIGRATION_PATH)
    # ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ ...
    assert re.search(
        r"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+updated_at\s+TIMESTAMPTZ",
        sql, re.IGNORECASE,
    ), "Migración debe declarar `ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ`."


def test_migration_declares_trigger_function_and_trigger():
    sql = _read(_MIGRATION_PATH)
    assert re.search(
        r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+public\.set_meal_plans_updated_at",
        sql, re.IGNORECASE,
    ), "Migración debe crear la función `public.set_meal_plans_updated_at`."
    assert re.search(
        r"CREATE\s+TRIGGER\s+trg_meal_plans_set_updated_at",
        sql, re.IGNORECASE,
    ), "Migración debe crear el trigger `trg_meal_plans_set_updated_at`."
    # BEFORE UPDATE, FOR EACH ROW — invariantes del diseño.
    assert re.search(r"BEFORE\s+UPDATE", sql, re.IGNORECASE), \
        "Trigger debe ser BEFORE UPDATE (no AFTER)."
    assert re.search(r"FOR\s+EACH\s+ROW", sql, re.IGNORECASE), \
        "Trigger debe correr FOR EACH ROW (cubre UPDATE-multi)."


def test_migration_declares_window_index():
    sql = _read(_MIGRATION_PATH)
    assert re.search(
        r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+idx_meal_plans_user_updated_at",
        sql, re.IGNORECASE,
    ), "Migración debe crear índice `idx_meal_plans_user_updated_at` para window queries."


def test_migration_notifies_pgrst_schema_reload():
    """Sin NOTIFY pgrst, las queries `?updated_at=gte=...` siguen 400 hasta
    que un worker PostgREST haga reload natural (~minutos)."""
    sql = _read(_MIGRATION_PATH)
    assert "NOTIFY pgrst" in sql, (
        "Migración debe incluir `NOTIFY pgrst, 'reload schema'` para que "
        "PostgREST refresque su cache inmediatamente tras aplicar."
    )


# ---------------------------------------------------------------------------
# 2. retry-chunk sigue usando `updated_at = NOW()` explícito.
# El trigger lo cubriría igualmente, pero la presencia explícita es señal
# del contrato: "este endpoint marca el plan como modificado físicamente".
# ---------------------------------------------------------------------------
def test_retry_chunk_explicitly_sets_updated_at():
    src = _read(_PLANS_ROUTER_PATH)
    # Buscamos el bloque del endpoint retry-chunk y verificamos que su UPDATE
    # de meal_plans incluya `updated_at = NOW()`.
    match = re.search(
        r"@router\.post\(\"/\{plan_id\}/retry-chunk/\{chunk_id\}\".*?(?=@router\.|\Z)",
        src, re.DOTALL,
    )
    assert match, "No se encontró el endpoint retry-chunk en plans.py."
    block = match.group(0)
    assert "UPDATE meal_plans" in block, \
        "retry-chunk debe ejecutar `UPDATE meal_plans`."
    assert re.search(r"updated_at\s*=\s*NOW\(\)", block, re.IGNORECASE), (
        "retry-chunk debe incluir `updated_at = NOW()` explícito. El trigger "
        "cubriría el caso, pero esta verificación documenta la intención "
        "del callsite y previene regresiones donde alguien remueva el SET "
        "asumiendo que el trigger es suficiente."
    )
