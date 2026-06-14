"""[P2-USER-DEPLETED-ITEMS-FK-IDX · 2026-05-23] Sanity test del covering
index para el FK `user_depleted_items_master_ingredient_id_fkey`.

Motivación:
    Supabase Performance Advisor flageaba (audit production-readiness
    2026-05-23):
        Table `public.user_depleted_items` has a foreign key
        `user_depleted_items_master_ingredient_id_fkey` without a
        covering index.

    Tabla creada en P3-DEPLETED-BD · 2026-05-22 con 2 unique partial
    indexes para dedupe pero ninguno cubría el FK directamente. Fix:
    migration SSOT `p2_user_depleted_items_fk_idx_2026_05_23.sql` con
    `CREATE INDEX IF NOT EXISTS idx_user_depleted_items_master_ingredient_id`.

Scope del test (parser-based, sin conexión a DB):
    1. La migration EXISTE en AMBOS directorios SSOT (workspace-root +
       backend/) per P3-MIGRATIONS-SSOT.
    2. Los 2 archivos son BYTE-IDÉNTICOS (drift detection).
    3. El SQL contiene los anchors críticos: CREATE INDEX, partial
       WHERE clause, COMMENT ON INDEX justificativo, sanity check
       post-apply.

Aplicación a la DB se hizo via Supabase MCP en el commit del bundle;
los logs de aplicación son la verificación runtime. Este test cierra
el gap "alguien borra/edita la migration y el index queda huérfano
del SSOT".

Tooltip-anchor: P2-USER-DEPLETED-ITEMS-FK-IDX | regression guard 2026-05-23
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MIGRATION_NAME = "p2_user_depleted_items_fk_idx_2026_05_23.sql"
_MIGRATION_ROOT = _REPO_ROOT / "migrations" / _MIGRATION_NAME
_MIGRATION_BACKEND = _REPO_ROOT / "backend" / "migrations" / _MIGRATION_NAME


def test_migration_file_present_in_both_ssot_dirs():
    """P3-MIGRATIONS-SSOT exige que TODA migration viva en ambos
    `migrations/` (workspace) Y `backend/migrations/`
    (backend repo). Si falta en uno, el push del repo correspondiente
    no lleva el cambio."""
    assert _MIGRATION_ROOT.exists(), (
        f"P2-USER-DEPLETED-ITEMS-FK-IDX: migration ausente del SSOT "
        f"workspace-root: {_MIGRATION_ROOT}. Restaurar el archivo (es "
        f"el SSOT de la migration aplicada via MCP)."
    )
    assert _MIGRATION_BACKEND.exists(), (
        f"P2-USER-DEPLETED-ITEMS-FK-IDX: migration ausente del SSOT "
        f"backend repo: {_MIGRATION_BACKEND}. P3-MIGRATIONS-SSOT exige "
        f"copia idéntica en ambos dirs para que `git push` desde el "
        f"backend repo lleve la migration."
    )


def test_migration_files_byte_identical():
    """Si los 2 archivos divergen, hay drift entre lo que el workspace
    push lleva y lo que el backend push lleva. Ambos deben ser la
    misma SSOT — si necesitas cambiar el contenido, edita ambos en
    el mismo commit."""
    root_bytes = _MIGRATION_ROOT.read_bytes()
    backend_bytes = _MIGRATION_BACKEND.read_bytes()
    assert root_bytes == backend_bytes, (
        "P2-USER-DEPLETED-ITEMS-FK-IDX drift: las copias del SSOT en "
        f"{_MIGRATION_ROOT} y {_MIGRATION_BACKEND} divergen "
        f"(byte size {len(root_bytes)} vs {len(backend_bytes)}). "
        "Re-sincronizar (copiar la versión correcta a ambos dirs)."
    )


def test_migration_contains_critical_anchors():
    """El SQL DEBE contener los 4 anchors del fix: CREATE INDEX
    idempotente, partial WHERE NOT NULL, COMMENT ON INDEX justificativo,
    sanity check post-apply. Si alguno desaparece, el index pierde
    semántica o idempotencia."""
    src = _MIGRATION_ROOT.read_text(encoding="utf-8")

    # 1. CREATE INDEX IF NOT EXISTS (idempotencia nativa).
    assert "CREATE INDEX IF NOT EXISTS idx_user_depleted_items_master_ingredient_id" in src, (
        "P2-USER-DEPLETED-ITEMS-FK-IDX: falta `CREATE INDEX IF NOT EXISTS` "
        "para idx_user_depleted_items_master_ingredient_id. Sin el "
        "IF NOT EXISTS, el reaply de la migration falla en un environment "
        "donde el index ya existe."
    )

    # 2. Partial WHERE (NOT NULL) — sólo cubre filas que participan del CASCADE.
    assert "WHERE master_ingredient_id IS NOT NULL" in src, (
        "P2-USER-DEPLETED-ITEMS-FK-IDX: falta el partial `WHERE "
        "master_ingredient_id IS NOT NULL`. Sin él, el index incluye "
        "filas con master_id NULL (items manuales sin canonicalizar) "
        "que NUNCA participan del CASCADE — desperdicio de storage + "
        "writes de mantenimiento."
    )

    # 3. COMMENT ON INDEX justificativo (analogous to los 5 unused_index aceptados).
    assert "COMMENT ON INDEX public.idx_user_depleted_items_master_ingredient_id" in src, (
        "P2-USER-DEPLETED-ITEMS-FK-IDX: falta el `COMMENT ON INDEX` "
        "justificativo. Cuando Supabase reporte este index como "
        "`unused_index` (esperado, igual que los 5 ya aceptados en "
        "CLAUDE.md), el operador necesita el COMMENT para entender "
        "por qué existe sin recurrir al historial git."
    )

    # 4. Sanity check post-apply (RAISE EXCEPTION si no se creó).
    assert "RAISE EXCEPTION 'P2-USER-DEPLETED-ITEMS-FK-IDX sanity" in src, (
        "P2-USER-DEPLETED-ITEMS-FK-IDX: falta el sanity check con "
        "`RAISE EXCEPTION` post-apply. Sin él, la migration puede "
        "completar como success aunque el index no se haya creado "
        "(edge case: WHERE clause syntax error rejected silently en "
        "ciertos pg versions)."
    )
