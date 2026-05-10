"""[P2-PERF-1 · 2026-05-10] Regression guard: las COMMENT ON INDEX que
documentan los 3 índices "unused" como KEEP intencional DEBEN existir en
archivos de migración (no solo en runtime).

Bug original (audit 2026-05-10):
    El advisor de Supabase reportaba 3 índices como `unused_index`. Audits
    previos (P2-B 2026-05-07, P1-HIST-NEW-7 2026-05-09) ya los habían
    documentado como falsos positivos vía `COMMENT ON INDEX`, PERO solo 1
    de los 3 (`idx_chunk_lesson_telemetry_plan_week`) tenía el COMMENT en
    archivo de migración. Las otras 2 vivían SOLO en DB runtime — un
    `db reset` desde migrations las perdía y un futuro operador no sabría
    que los índices son intencionales (probable drop equivocado, regresión
    a la lección P2-5).

    Mismo patrón anti-drift que P1-NEW-A: DDL aplicado solo runtime sin
    SOT en migrations es frágil.

Fix:
    Migración `supabase/migrations/p2_perf_1_consolidate_unused_index_comments.sql`
    aplica las 2 COMMENTs faltantes. Esta verificación estática asegura
    que el archivo existe y contiene las referencias esperadas.

Cobertura:
    1. Archivo de migración existe con las 2 COMMENT ON INDEX faltantes.
    2. Texto de cada COMMENT incluye los anchors esperados (palabra `KEEP`,
       lección `P2-5`, mención de `FK ... ON DELETE CASCADE`).
    3. La 3ra COMMENT (idx_chunk_lesson_telemetry_plan_week) sigue en su
       migración original (p1_hist_new_7_*) — defensa contra rollback
       accidental que la elimine.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _BACKEND_ROOT.parent / "supabase" / "migrations"

_NEW_MIGRATION = _MIGRATIONS_DIR / "p2_perf_1_consolidate_unused_index_comments.sql"
_LEGACY_MIGRATION = _MIGRATIONS_DIR / "p1_hist_new_7_recreate_chunk_lesson_telemetry_plan_week_idx.sql"


# ---------------------------------------------------------------------------
# 1. Migración nueva existe y consolida las 2 COMMENT faltantes
# ---------------------------------------------------------------------------
def test_p2_perf_1_migration_file_exists():
    """La migración consolidando las 2 COMMENT runtime DEBE existir.
    Sin ella, un `db reset` perdería las anotaciones y un operador
    futuro podría dropear los índices por mistakenly assumir que son
    bloat (regresión a la lección P2-5)."""
    assert _NEW_MIGRATION.exists(), (
        f"Migración P2-PERF-1 no encontrada: {_NEW_MIGRATION}\n"
        f"Esto reabre el gap original: las COMMENT runtime se pierden "
        f"al recrear DB desde migrations."
    )


@pytest.mark.parametrize("index_name", [
    "idx_failed_inventory_deductions_user_id",
    "idx_nightly_rotation_queue_user_id",
])
def test_migration_contains_index_comment(index_name: str):
    """La migración debe contener un `COMMENT ON INDEX` para cada uno
    de los 2 índices KEEP que estaban solo en runtime."""
    text = _NEW_MIGRATION.read_text(encoding="utf-8")
    pattern = f"COMMENT ON INDEX public.{index_name}"
    assert pattern in text, (
        f"Migración P2-PERF-1 no contiene `{pattern}`. Si lo removiste "
        f"intencionalmente (e.g., decidiste dropear el índice), "
        f"actualizar también este test y la lección P2-5."
    )


@pytest.mark.parametrize("index_name", [
    "idx_failed_inventory_deductions_user_id",
    "idx_nightly_rotation_queue_user_id",
])
def test_migration_comment_has_keep_anchor(index_name: str):
    """Cada COMMENT debe incluir anchors que documentan POR QUÉ es KEEP:
    la palabra `KEEP`, mención del FK CASCADE, y referencia a la lección
    P2-5 (advisor unused_index no observa uso interno por FK).
    """
    text = _NEW_MIGRATION.read_text(encoding="utf-8")
    # Buscar el bloque COMMENT específico para este índice.
    idx_comment_start = text.find(f"COMMENT ON INDEX public.{index_name}")
    assert idx_comment_start >= 0, f"COMMENT para `{index_name}` ausente"
    # Tomar los siguientes 600 chars (el COMMENT body es de ~300-400).
    comment_block = text[idx_comment_start:idx_comment_start + 600]

    for anchor in ["KEEP", "ON DELETE CASCADE", "P2-5"]:
        assert anchor in comment_block, (
            f"COMMENT para `{index_name}` no menciona `{anchor}`. "
            f"Sin este anchor, un futuro operador no entendería el "
            f"contexto del índice y podría dropearlo. Restaurar el "
            f"contexto en el COMMENT."
        )


# ---------------------------------------------------------------------------
# 2. La 3ra COMMENT sigue en su migración original (no migrar)
# ---------------------------------------------------------------------------
def test_legacy_chunk_lesson_telemetry_comment_still_in_p1_hist_new_7():
    """`idx_chunk_lesson_telemetry_plan_week` ya tenía COMMENT en
    p1_hist_new_7_*.sql desde antes. NO debe duplicarse en P2-PERF-1
    (eso introduciría drift entre dos migrations sobre el mismo
    índice). Verificar que el COMMENT sigue en su lugar original."""
    assert _LEGACY_MIGRATION.exists(), (
        f"Migración legacy P1-HIST-NEW-7 no encontrada: {_LEGACY_MIGRATION}"
    )
    text = _LEGACY_MIGRATION.read_text(encoding="utf-8")
    assert "COMMENT ON INDEX public.idx_chunk_lesson_telemetry_plan_week" in text, (
        "La COMMENT del 3er índice fue removida de su migración original. "
        "Si quieres consolidarla en P2-PERF-1, asegurar que NO quede "
        "duplicada — una de las 2 referencias debe desaparecer."
    )


def test_p2_perf_1_does_not_duplicate_legacy_comment():
    """Defensa contra duplicación accidental: P2-PERF-1 NO debe contener
    una COMMENT para `idx_chunk_lesson_telemetry_plan_week` (eso ya lo
    cubre p1_hist_new_7_*.sql). Tener COMMENTs en 2 migrations sobre el
    mismo índice introduce drift cuando una se actualiza y la otra no."""
    text = _NEW_MIGRATION.read_text(encoding="utf-8")
    assert "COMMENT ON INDEX public.idx_chunk_lesson_telemetry_plan_week" not in text, (
        "P2-PERF-1 duplica la COMMENT del índice ya cubierto por "
        "p1_hist_new_7_*.sql. Eliminar de P2-PERF-1 o explicar el motivo "
        "explícito de la duplicación en el header de la migración."
    )
