"""[P2-WHITELIST-AUDIT · 2026-05-12] Test parser-based que ancla la tabla
"Advisors aceptados" del CLAUDE.md a los artefactos SQL que la sustentan.

Anchor: P2-WHITELIST-AUDIT-ADVISORS.

Hoy la tabla en CLAUDE.md documenta 7 advisors aceptados intencionalmente
(security + performance). Cada uno DEBE estar respaldado por al menos un
artefacto SQL (CREATE / COMMENT) en `supabase/migrations/`. Si alguien
renombra un objeto (función, tabla, índice) o borra la migración SSOT,
el COMMENT que justifica el advisor queda huérfano y un futuro auditor
no tiene cómo verificar que la decisión sigue vigente.

El test es parser-based porque NO depende de Supabase live — solo escanea
el árbol de migraciones cargado en disco.

Estrategia:
  - Para cada nombre canónico (función / tabla / índice), grep recursivo
    sobre `supabase/migrations/*.sql`.
  - Falla si algún nombre no aparece en al menos 1 archivo.
  - Para los unused_index entries, verifica adicionalmente que existe al
    menos UN `COMMENT ON INDEX` que mencione el nombre — porque la decisión
    "aceptado intencional" se documenta en el COMMENT, no en el CREATE.
"""
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATIONS_DIR = _REPO_ROOT / "supabase" / "migrations"


def _load_all_migrations() -> dict:
    """Carga todos los .sql del directorio en un dict {filename: contents}."""
    assert _MIGRATIONS_DIR.exists(), (
        f"Directorio de migraciones no encontrado: {_MIGRATIONS_DIR}. "
        f"Si moviste `supabase/migrations/` a otra ubicación (e.g., al repo "
        f"backend/), actualiza esta ruta + el resto de los tests parser-based "
        f"que asumen esta ubicación."
    )
    return {
        f.name: f.read_text(encoding="utf-8", errors="ignore")
        for f in _MIGRATIONS_DIR.glob("*.sql")
    }


# Nombres canónicos extraídos de CLAUDE.md sección "Advisors aceptados".
# Si añades/quitas un advisor de esa tabla, sincronizá acá. El test es la
# segunda mitad del contrato — la docs sin test se desactualiza silenciosa.
_REQUIRED_ANCHORS_ANY = (
    # Security
    "increment_inventory_quantity",   # SECURITY DEFINER function (P2-4)
    "meal_plans_audit",               # tabla operacional append-only (P3-FINAL-1)
)

_REQUIRED_ANCHORS_WITH_COMMENT_ON_INDEX = (
    "idx_chunk_lesson_telemetry_plan_week",   # P1-HIST-NEW-7
    "idx_failed_inventory_deductions_user_id",  # P2-PERF-1
    "idx_nightly_rotation_queue_user_id",     # P2-PERF-1
    "idx_meal_plans_audit_meal_plan_id",      # P3-FINAL-1
    "idx_meal_plans_audit_user_id",           # P3-FINAL-1
    "idx_meal_plans_audit_action_created",    # P3-FINAL-1
)


def test_advisor_anchors_present_in_migrations():
    """Cada nombre canónico (función, tabla) debe aparecer en al menos 1
    migración. Si alguno desaparece, el advisor pierde su SSOT.
    """
    migrations = _load_all_migrations()
    all_text = "\n".join(migrations.values())
    missing = [name for name in _REQUIRED_ANCHORS_ANY if name not in all_text]
    assert not missing, (
        f"Anchors de advisors aceptados ausentes en supabase/migrations/: "
        f"{missing}. CLAUDE.md sección 'Advisors aceptados' los lista pero "
        f"ninguna migración los menciona. Posible regresión: alguien renombró "
        f"el objeto o borró la migración SSOT. Restaurar o actualizar tabla."
    )


def test_unused_index_anchors_have_comment_on_index():
    """Cada unused_index aceptado debe tener al menos UN `COMMENT ON INDEX
    <nombre>` (case-insensitive) en alguna migración — sin el COMMENT el
    advisor reporta como bug pendiente, no como decisión documentada.

    Caso por caso:
      - `idx_chunk_lesson_telemetry_plan_week`: cubre FK + query /lifetime-lessons.
      - `idx_failed_inventory_deductions_user_id`: cubre FK auth.users CASCADE.
      - `idx_nightly_rotation_queue_user_id`: cubre FK user_profiles CASCADE.
      - 3 índices `idx_meal_plans_audit_*`: sirven SOP P3-AUDIT-6 forensics.
    """
    migrations = _load_all_migrations()
    all_text_lower = "\n".join(migrations.values()).lower()

    missing_comments: list = []
    for index_name in _REQUIRED_ANCHORS_WITH_COMMENT_ON_INDEX:
        # Buscar `COMMENT ON INDEX [schema.]<index_name>` (lowercase).
        # Aceptamos prefijo de schema (`public.`) opcional.
        needle_simple = f"comment on index {index_name.lower()}"
        needle_schema = f"comment on index public.{index_name.lower()}"
        if needle_simple not in all_text_lower and needle_schema not in all_text_lower:
            missing_comments.append(index_name)

    assert not missing_comments, (
        f"Índices listados como 'unused_index aceptado' SIN COMMENT ON INDEX "
        f"en migraciones: {missing_comments}. La decisión 'aceptado intencional' "
        f"vive en el COMMENT del índice — si alguien borró el COMMENT, el advisor "
        f"reportará como bug. Restaurar el COMMENT ON INDEX en la migración "
        f"correspondiente (P1-HIST-NEW-7 / P2-PERF-1 / P3-FINAL-1)."
    )


def test_p2_whitelist_audit_advisors_anchor_present():
    """El marker P2-WHITELIST-AUDIT-ADVISORS debe vivir en este archivo para
    que el cross-link slug del test legacy P2-HIST-AUDIT-14 lo encuentre y
    `_LAST_KNOWN_PFIX` bump sea verificable."""
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P2-WHITELIST-AUDIT" in src
    assert "P2-WHITELIST-AUDIT-ADVISORS" in src
