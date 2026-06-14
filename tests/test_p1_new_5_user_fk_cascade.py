"""[P1-NEW-5 · 2026-05-11] Audit sistemático de FK ON DELETE CASCADE
para columnas user_id en tablas user-scoped.

Bug original (audit 2026-05-11):
    Audit DB de 27 tablas con `user_id` detectó 2 con FK declarado
    pero SIN ON DELETE CASCADE:
      - `api_usage.user_id → auth.users(id)` (telemetría paywall)
      - `meal_rejections.user_id → auth.users(id)` (feedback loop)

    Sin CASCADE: al borrar un usuario (GDPR delete, SRE cleanup),
    las filas quedan huérfanas con `user_id` apuntando a UUID
    inexistente. Las queries de SRE / analytics ven rows sin owner
    real y los counts se sesgan.

Fix:
    Migración `p1_new_5_user_fk_cascade_consolidate.sql` aplicada
    a producción. DROP + ADD CONSTRAINT con ON DELETE CASCADE.

Estrategia del test (parser-based sobre la migración SQL):
    1. El archivo de migración existe.
    2. Las 2 tablas afectadas (api_usage, meal_rejections) están
       en la migración con DROP + ADD pattern.
    3. ON DELETE CASCADE explícito (no missing).
    4. COMMENT ON CONSTRAINT documenta el cambio.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATION_FP = (
    _REPO_ROOT / "migrations"
    / "p1_new_5_user_fk_cascade_consolidate.sql"
)


@pytest.fixture(scope="module")
def migration_sql() -> str:
    assert _MIGRATION_FP.exists(), (
        f"Migración P1-NEW-5 ausente: {_MIGRATION_FP}. "
        "Sin el archivo SQL, el fix vive solo en prod (riesgo de "
        "drift schema↔repo)."
    )
    return _MIGRATION_FP.read_text(encoding="utf-8")


def test_migration_file_exists(migration_sql: str):
    """Sanity: archivo presente + no vacío."""
    assert len(migration_sql) > 100, "Migración demasiado corta — ¿stub?"


def test_api_usage_cascade_applied(migration_sql: str):
    """`api_usage.user_id → auth.users(id) ON DELETE CASCADE`
    debe estar declarado en la migración."""
    # Pattern: en cualquier orden ADD CONSTRAINT api_usage_user_id_fkey ... CASCADE
    pattern = re.compile(
        r"ADD\s+CONSTRAINT\s+api_usage_user_id_fkey[\s\S]+?ON\s+DELETE\s+CASCADE",
        re.IGNORECASE,
    )
    assert pattern.search(migration_sql), (
        "P1-NEW-5 regresión: api_usage_user_id_fkey ya no se reemite "
        "con ON DELETE CASCADE. Sin esto, borrar un usuario en "
        "auth.users deja telemetría huérfana."
    )


def test_meal_rejections_cascade_applied(migration_sql: str):
    """`meal_rejections.user_id → auth.users(id) ON DELETE CASCADE`
    debe estar declarado en la migración."""
    pattern = re.compile(
        r"ADD\s+CONSTRAINT\s+meal_rejections_user_id_fkey[\s\S]+?ON\s+DELETE\s+CASCADE",
        re.IGNORECASE,
    )
    assert pattern.search(migration_sql), (
        "P1-NEW-5 regresión: meal_rejections_user_id_fkey ya no se "
        "reemite con ON DELETE CASCADE. Sin esto, los rechazos de "
        "menús quedan huérfanos al borrar un usuario."
    )


def test_migration_uses_drop_if_exists(migration_sql: str):
    """Idempotencia: `DROP CONSTRAINT IF EXISTS` permite re-correr
    la migración sin error si ya se aplicó."""
    drops = re.findall(
        r"DROP\s+CONSTRAINT\s+IF\s+EXISTS",
        migration_sql,
        re.IGNORECASE,
    )
    assert len(drops) >= 2, (
        f"P1-NEW-5 regresión: solo {len(drops)} DROP CONSTRAINT IF EXISTS "
        "encontrados (esperado ≥2 — uno por FK reescrito). Sin esto, "
        "re-correr la migración falla en lugar de ser idempotente."
    )


def test_migration_documents_changes_with_comments(migration_sql: str):
    """Cada FK reescrito debe tener `COMMENT ON CONSTRAINT` documentando
    la razón. Sin esto, un operador que consulta el schema no sabe por
    qué el CASCADE existe."""
    api_comment = re.search(
        r"COMMENT\s+ON\s+CONSTRAINT\s+api_usage_user_id_fkey",
        migration_sql,
        re.IGNORECASE,
    )
    mr_comment = re.search(
        r"COMMENT\s+ON\s+CONSTRAINT\s+meal_rejections_user_id_fkey",
        migration_sql,
        re.IGNORECASE,
    )
    assert api_comment, (
        "P1-NEW-5 regresión: api_usage_user_id_fkey sin COMMENT. "
        "Sin documentación inline, el motivo del CASCADE se pierde."
    )
    assert mr_comment, (
        "P1-NEW-5 regresión: meal_rejections_user_id_fkey sin COMMENT."
    )


def test_migration_references_p1_new_5_marker(migration_sql: str):
    """El header del archivo debe identificar el P-fix (`P1-NEW-5`)
    para que un git blame post-mortem encuentre la migración."""
    assert "P1-NEW-5" in migration_sql, (
        "P1-NEW-5 regresión: la migración no se identifica como P1-NEW-5 "
        "en comentarios. Bump del marker en el header para trazabilidad."
    )


def test_migration_only_touches_known_tables(migration_sql: str):
    """Defensa contra over-broad changes: la migración solo debe tocar
    `api_usage` y `meal_rejections` (las 2 tablas con FK sin CASCADE
    detectadas en el audit). Si toca otras tablas, refactor parcial."""
    altered_tables = set(
        re.findall(
            r"ALTER\s+TABLE\s+(?:public\.)?([a-z_]+)",
            migration_sql,
            re.IGNORECASE,
        )
    )
    expected = {"api_usage", "meal_rejections"}
    extra = altered_tables - expected
    assert not extra, (
        f"P1-NEW-5 regresión: la migración modifica tablas inesperadas: "
        f"{sorted(extra)}. Solo deben tocarse {sorted(expected)} — "
        f"otros cambios requieren migración propia."
    )
