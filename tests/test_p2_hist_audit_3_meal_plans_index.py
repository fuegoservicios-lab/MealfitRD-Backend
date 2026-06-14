"""[P2-HIST-AUDIT-3 · 2026-05-09] Tests static analysis sobre la
migración SSOT que añade el índice compuesto
``idx_meal_plans_user_created_at`` en `meal_plans`.

Bug original (audit historial 2026-05-08):
    Solo existía `idx_meal_plans_user_id` simple. EXPLAIN del listado
    del Historial y del SELECT target del restore mostraba `Sort`
    después del `Index Scan`, lo que para datasets grandes (tier
    ultra con 100+ planes) introducía costo O(n log n) en memoria.

Fix:
    Migración SSOT idempotente con `CREATE INDEX IF NOT EXISTS
    idx_meal_plans_user_created_at ON meal_plans (user_id,
    created_at DESC)`. Aplicada al remoto (project
    mpoodlmnzaeuuazsazbj) y verificada vía MCP — el planner ya elige
    el nuevo índice.

Cobertura (static analysis del SQL):
    - Migración existe en `migrations/`.
    - Marker P2-HIST-AUDIT-3.
    - CREATE INDEX con `(user_id, created_at DESC)`.
    - Idempotente vía `IF NOT EXISTS`.
    - Comment documental aplicado.
    - Trade-off documentado (sort GREATEST sigue en memoria; nota
      sobre opción B con expression index + función IMMUTABLE
      como mejora futura).
    - Patrón consistente con otros índices compuestos del repo
      (`idx_chunk_lesson_telemetry_user_created_at`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_MIGRATION_PATH = (
    _BACKEND_ROOT.parent
    / "migrations"
    / "p2_hist_audit_3_meal_plans_user_created_at_idx.sql"
)


def _migration_text() -> str:
    return _MIGRATION_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Migración existe y tiene el marker
# ---------------------------------------------------------------------------
def test_migration_file_exists():
    assert _MIGRATION_PATH.exists(), (
        f"No se encontró la migración P2-HIST-AUDIT-3 en {_MIGRATION_PATH}. "
        f"Path esperado por convención (SSOT en migrations/)."
    )


def test_marker_present():
    text = _migration_text()
    assert "P2-HIST-AUDIT-3" in text


# ---------------------------------------------------------------------------
# 2. Contrato del CREATE INDEX
# ---------------------------------------------------------------------------
def test_creates_compound_index_on_user_id_created_at_desc():
    text = _migration_text()
    # Patrón: CREATE INDEX [IF NOT EXISTS] <name> ON public.meal_plans
    # USING btree (user_id, created_at DESC).
    assert re.search(
        r"CREATE\s+INDEX[\s\S]*?ON\s+public\.meal_plans\s+USING\s+btree\s*\(\s*user_id\s*,\s*created_at\s+DESC\s*\)",
        text,
        re.IGNORECASE,
    ), (
        "Migración debe crear `idx_meal_plans_user_created_at` con "
        "(user_id, created_at DESC). Sin el orden DESC explícito, el "
        "planner no puede usar el índice para `ORDER BY created_at DESC`."
    )


def test_index_name_follows_convention():
    """Nombre `idx_meal_plans_user_created_at` consistente con
    `idx_chunk_lesson_telemetry_user_created_at` y otros índices
    compuestos del repo."""
    text = _migration_text()
    assert "idx_meal_plans_user_created_at" in text


# ---------------------------------------------------------------------------
# 3. Idempotencia
# ---------------------------------------------------------------------------
def test_migration_is_idempotent():
    """`IF NOT EXISTS` permite re-aplicar la migración sin error."""
    text = _migration_text()
    assert re.search(
        r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+idx_meal_plans_user_created_at",
        text,
        re.IGNORECASE,
    ), (
        "Falta `IF NOT EXISTS` en CREATE INDEX. Sin esto, re-aplicar "
        "la migración falla con `relation already exists`."
    )


# ---------------------------------------------------------------------------
# 4. Comment documental
# ---------------------------------------------------------------------------
def test_comment_on_index_present():
    """COMMENT ON INDEX para diagnóstico SQL futuro."""
    text = _migration_text()
    assert re.search(
        r"COMMENT\s+ON\s+INDEX\s+public\.idx_meal_plans_user_created_at",
        text,
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# 5. Trade-off documentado
# ---------------------------------------------------------------------------
def test_documents_trade_off_with_greatest_sort():
    """El comentario debe documentar que el sort por GREATEST(...)
    NO se elimina con este índice (deja puerta abierta para opción B
    con expression index + función IMMUTABLE)."""
    text = _migration_text()
    # Buscamos referencia al GREATEST y al concepto de IMMUTABLE.
    assert "GREATEST" in text, (
        "Comentario debe mencionar el sort por `GREATEST` que sigue "
        "en memoria — sin esto, un mantenedor podría asumir que el "
        "índice resuelve el bug del audit completamente."
    )
    assert "IMMUTABLE" in text, (
        "Comentario debe mencionar la opción B (expression index con "
        "función IMMUTABLE) como mejora futura cuando el dataset escale."
    )


# ---------------------------------------------------------------------------
# 6. Scope guard: no toca otros índices ni tablas
# ---------------------------------------------------------------------------
def test_no_drop_of_existing_index():
    """La migración NO debe DROP-ear `idx_meal_plans_user_id`. El
    planner elige el más conveniente automáticamente; el simple es
    útil como fallback y permite rollback fácil del compuesto si
    surge problema.
    """
    text = _migration_text()
    assert not re.search(
        r"DROP\s+INDEX[\s\S]*?idx_meal_plans_user_id",
        text,
        re.IGNORECASE,
    ), (
        "Migración NO debe DROP `idx_meal_plans_user_id`. Mantenerlo "
        "permite rollback del compuesto sin perder cobertura para "
        "queries `WHERE user_id = ?`."
    )


def test_no_alter_to_other_tables():
    text = _migration_text()
    forbidden_tables = [
        "chunk_lesson_telemetry",
        "chunk_user_locks",
        "plan_chunk_queue",
        "plan_chunk_metrics",
        "chunk_deferrals",
        "user_profiles",
    ]
    for table in forbidden_tables:
        # Permitimos menciones en comentarios pero no DDL.
        assert not re.search(
            rf"(CREATE|ALTER|DROP)\s+(INDEX|TABLE)[^;]*\b{re.escape(table)}\b",
            text,
            re.IGNORECASE,
        ), (
            f"P2-HIST-AUDIT-3 no debe modificar `{table}`. Scope "
            f"creep introduce riesgo y dificulta rollback."
        )
