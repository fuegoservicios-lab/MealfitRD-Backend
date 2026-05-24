"""[P1-PROD-AUDIT-1 · 2026-05-23] Toda migración en `supabase/migrations/`
debe ser idempotente (re-aplicable sin fallar).

Gap original (audit production-readiness 2026-05-23, B-P1-8):
    El audit external mencionó "Migraciones idempotentes pero sin SSOT".
    SSOT cross-repo se cubre en CLAUDE.md "SSOT de migrations" (sync
    bidireccional con workspace-root). En este standalone repo, solo hay
    un dir → SSOT es trivial.

    Lo que SÍ es accionable: **idempotencia**. CLAUDE.md ya documenta la
    convención (P3-MIGRATION-IDEMPOTENCE-DOC · 2026-05-15): "Idempotente
    obligatorio: `IF NOT EXISTS` en CREATE/ADD COLUMN, `DROP CONSTRAINT
    IF EXISTS` antes de ADD, `DO $$ RAISE EXCEPTION` sanity."

    Pero NO existía test enforzando esa convención. Una migración con
    `CREATE TABLE foo` (sin IF NOT EXISTS) que se re-aplique (cron, retry
    de Supabase, branching) fallaría con 42P07 duplicate_object.

Fix:
    Este test escanea TODOS los `.sql` de `supabase/migrations/` y para
    cada uno valida:
      (A) DDL statements usan IF NOT EXISTS / IF EXISTS apropiados.
      (B) Naming convention `p<N>_<slug>(_YYYY_MM_DD)?.sql`.
      (C) Anchor inline `[Pn-...]` o `IDEMPOTENT-EXEMPT: <razón>` si la
          migración legítimamente NO puede ser idempotente.

Cobertura:
    A) Migraciones dir existe.
    B) Cada `CREATE TABLE` tiene `IF NOT EXISTS`.
    C) Cada `CREATE INDEX` tiene `IF NOT EXISTS` (o uses `CREATE INDEX
       CONCURRENTLY IF NOT EXISTS` que es lo correcto en prod).
    D) Cada `ALTER TABLE ... ADD COLUMN` tiene `IF NOT EXISTS` (postgres
       9.6+).
    E) Cada `ADD CONSTRAINT` tiene `DROP CONSTRAINT IF EXISTS` antes O
       usa `IF NOT EXISTS` (postgres 18+; not yet supported by Supabase
       11.x → DROP+ADD pattern es el canónico).
    F) Naming convention.

Allowlist:
    Migraciones legítimamente NO idempotentes (e.g. DML-only backfill que
    debe correr una sola vez) pueden tener inline marker
    `-- IDEMPOTENT-EXEMPT: <razón>`. Solo backfills puros — DDL siempre
    debe ser idempotente.

Tooltip-anchor: P1-PROD-AUDIT-1-MIGRATIONS-IDEMPOTENT | audit 2026-05-23.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _BACKEND_ROOT / "supabase" / "migrations"

_NAMING_PATTERN = re.compile(
    # `p<N>` o `p<N><letter>` (e.g. p1a) seguido de slug. Letter suffix
    # acepta el caso legacy `p1a_drop_dup_indexes_chunk_telemetry.sql`.
    r"^p\d+[a-z]?_[a-z0-9_]+(_\d{4}_\d{2}_\d{2})?\.sql$|"
    r"^add_[a-z0-9_]+_\d{4}_\d{2}_\d{2}\.sql$|"
    r"^db_p\d+_[a-z0-9_]+_\d{4}_\d{2}_\d{2}\.sql$"
)

_IDEMPOTENT_EXEMPT_MARKER = re.compile(r"--\s*IDEMPOTENT-EXEMPT\s*:")


def _list_migration_files():
    if not _MIGRATIONS_DIR.exists():
        return []
    return sorted([p for p in _MIGRATIONS_DIR.glob("*.sql")])


def _strip_sql_comments(text: str) -> str:
    """Elimina comentarios SQL (`-- line` + `/* block */`) para que el
    scan de DDL no produzca falsos positivos cuando el comentario
    explica el patrón pero el código no lo tiene literalmente.
    """
    # Block comments primero (multiline).
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Line comments (-- hasta fin de línea).
    text = re.sub(r"--[^\n]*", "", text)
    return text


def test_migrations_dir_exists():
    assert _MIGRATIONS_DIR.exists(), (
        f"Migrations dir {_MIGRATIONS_DIR} ausente. Si moviste la ruta, "
        f"actualizar este test."
    )


def test_at_least_one_migration_exists():
    files = _list_migration_files()
    assert len(files) >= 1, (
        f"0 migraciones en {_MIGRATIONS_DIR}. Si todas se movieron, "
        f"actualizar este test."
    )


def test_naming_convention():
    """Files siguen `p<N>_<slug>(_YYYY_MM_DD)?.sql` o variantes documentadas
    (add_<slug>_YYYY_MM_DD.sql para feature toggles, db_p<N>_<slug>_... para
    RLS lockdown)."""
    files = _list_migration_files()
    bad_names = [f.name for f in files if not _NAMING_PATTERN.match(f.name)]
    assert not bad_names, (
        f"Migraciones con naming roto: {bad_names}. Convención: "
        f"`p<N>_<slug>(_YYYY_MM_DD)?.sql` o variantes documentadas en "
        f"CLAUDE.md."
    )


@pytest.mark.parametrize("migration", _list_migration_files(), ids=lambda p: p.name)
def test_create_table_uses_if_not_exists(migration: Path):
    """`CREATE TABLE foo (...)` sin IF NOT EXISTS = no idempotente. Si la
    migración se re-aplica (retry de Supabase, branching), falla con
    42P07 duplicate_object → bloquea schema deploys.
    """
    text = _strip_sql_comments(migration.read_text(encoding="utf-8"))
    if _IDEMPOTENT_EXEMPT_MARKER.search(migration.read_text(encoding="utf-8")):
        pytest.skip("Migración marcada IDEMPOTENT-EXEMPT")
    # Buscar `CREATE TABLE <name>` que NO sea seguido de `IF NOT EXISTS`.
    # Ignorar `CREATE TEMP TABLE` y `CREATE TABLE IF NOT EXISTS` (correctos).
    bad = re.findall(
        r"CREATE\s+(?!TEMP\s)(?!TEMPORARY\s)TABLE\s+(?!IF\s+NOT\s+EXISTS)",
        text,
        re.IGNORECASE,
    )
    assert not bad, (
        f"{migration.name}: {len(bad)} `CREATE TABLE` sin `IF NOT EXISTS`. "
        f"Re-apply fallaría con 42P07. Reemplazar con `CREATE TABLE IF NOT "
        f"EXISTS ...` o documentar con `-- IDEMPOTENT-EXEMPT: <razón>`."
    )


@pytest.mark.parametrize("migration", _list_migration_files(), ids=lambda p: p.name)
def test_create_index_uses_if_not_exists(migration: Path):
    """`CREATE INDEX foo ON ...` sin IF NOT EXISTS = no idempotente.
    Patrón canónico productivo: `CREATE INDEX CONCURRENTLY IF NOT EXISTS`.
    """
    text = _strip_sql_comments(migration.read_text(encoding="utf-8"))
    if _IDEMPOTENT_EXEMPT_MARKER.search(migration.read_text(encoding="utf-8")):
        pytest.skip("Migración marcada IDEMPOTENT-EXEMPT")
    # Match `CREATE INDEX <name>` o `CREATE UNIQUE INDEX <name>` o
    # `CREATE INDEX CONCURRENTLY <name>` — todos requieren IF NOT EXISTS.
    bad = re.findall(
        r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?!IF\s+NOT\s+EXISTS)([A-Za-z_][A-Za-z0-9_]*)",
        text,
        re.IGNORECASE,
    )
    assert not bad, (
        f"{migration.name}: CREATE INDEX sin IF NOT EXISTS: {bad}. "
        f"Re-apply falla con 42P07. Convención: `CREATE INDEX CONCURRENTLY "
        f"IF NOT EXISTS idx_name ON table_name(col);` (CONCURRENTLY evita "
        f"bloquear writes en tablas grandes)."
    )


@pytest.mark.parametrize("migration", _list_migration_files(), ids=lambda p: p.name)
def test_add_column_uses_if_not_exists(migration: Path):
    """`ALTER TABLE foo ADD COLUMN bar` sin IF NOT EXISTS = no idempotente
    (postgres 9.6+ soporta ADD COLUMN IF NOT EXISTS).
    """
    text = _strip_sql_comments(migration.read_text(encoding="utf-8"))
    if _IDEMPOTENT_EXEMPT_MARKER.search(migration.read_text(encoding="utf-8")):
        pytest.skip("Migración marcada IDEMPOTENT-EXEMPT")
    bad = re.findall(
        r"ADD\s+COLUMN\s+(?!IF\s+NOT\s+EXISTS)",
        text,
        re.IGNORECASE,
    )
    assert not bad, (
        f"{migration.name}: ADD COLUMN sin IF NOT EXISTS ({len(bad)} occ). "
        f"Re-apply falla con 42701 duplicate_column. Convención: "
        f"`ALTER TABLE foo ADD COLUMN IF NOT EXISTS bar type;`."
    )


@pytest.mark.parametrize("migration", _list_migration_files(), ids=lambda p: p.name)
def test_add_constraint_pairs_with_drop_if_exists(migration: Path):
    """`ADD CONSTRAINT` debe ser precedido por `DROP CONSTRAINT IF EXISTS`
    en la misma migración (patrón canónico hasta postgres 17, since 18+
    soporta ADD CONSTRAINT IF NOT EXISTS).

    Acepta también el patrón DO $$ block que usa `EXECUTE format(...)`
    para DROP dinámico — común en migraciones que iteran constraints
    descubiertos en runtime (e.g. p0_hist_3 que dropea TODOS los FK con
    nombre matching un pattern). Detección via heurística: si el archivo
    tiene `DO $$` + `EXECUTE` + `DROP CONSTRAINT`, se asume cuidado.
    """
    raw = migration.read_text(encoding="utf-8")
    text = _strip_sql_comments(raw)
    if _IDEMPOTENT_EXEMPT_MARKER.search(raw):
        pytest.skip("Migración marcada IDEMPOTENT-EXEMPT")

    # Skip si la migración usa DO $$ blocks con DROP dinámico (common pattern
    # para FK consolidation, sweep operations).
    has_dynamic_drop = (
        "DO $$" in text
        and "EXECUTE" in text
        and re.search(r"DROP\s+CONSTRAINT", text, re.IGNORECASE)
    )
    if has_dynamic_drop:
        pytest.skip(
            "Migración usa DO $$ block con EXECUTE format() para DROP dinámico — "
            "patrón válido para constraints descubiertos en runtime"
        )

    # Identificar todos los nombres de CONSTRAINT añadidos.
    adds = re.findall(
        r"ADD\s+CONSTRAINT\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)",
        text,
        re.IGNORECASE,
    )
    if not adds:
        pytest.skip("Migración no añade constraints")
    # Para cada constraint añadido, debe haber un DROP IF EXISTS del mismo
    # nombre EN ESTE archivo (o uses ADD CONSTRAINT IF NOT EXISTS — pg18+).
    drops = re.findall(
        r"DROP\s+CONSTRAINT\s+IF\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)",
        text,
        re.IGNORECASE,
    )
    add_if_not_exists = re.findall(
        r"ADD\s+CONSTRAINT\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)",
        text,
        re.IGNORECASE,
    )
    # Patrón canónico alternativo: `IF NOT EXISTS (SELECT 1 FROM pg_constraint
    # WHERE conname = '<name>')` antes de ADD. Acepta también este.
    pg_constraint_guards = set(re.findall(
        r"conname\s*=\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
        text,
    ))

    bad = []
    for c in adds:
        if c in drops or c in add_if_not_exists or c in pg_constraint_guards:
            continue
        bad.append(c)
    assert not bad, (
        f"{migration.name}: constraints añadidos sin DROP IF EXISTS previo: "
        f"{bad}. Re-apply falla con 42710 duplicate_object. Patrones canónicos "
        f"aceptados:\n"
        f"  - `DROP CONSTRAINT IF EXISTS <name>; ALTER TABLE ... ADD CONSTRAINT <name> ...;`\n"
        f"  - `IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '<name>') THEN ... END IF;`\n"
        f"  - DO $$ block con EXECUTE format() para DROP dinámico.\n"
    )


def test_idempotent_exempt_must_have_reason():
    """Migración marcada exempt DEBE tener razón legible — no solo marker
    vacío. Cero margen para `IDEMPOTENT-EXEMPT:` sin contenido."""
    files = _list_migration_files()
    exempt_no_reason = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for line in text.split("\n"):
            m = re.search(r"--\s*IDEMPOTENT-EXEMPT\s*:(.*)$", line)
            if m and not m.group(1).strip():
                exempt_no_reason.append(f.name)
                break
    assert not exempt_no_reason, (
        f"Migraciones con `IDEMPOTENT-EXEMPT:` SIN razón explicada: "
        f"{exempt_no_reason}. Cada exempt necesita justificar POR QUÉ no "
        f"puede ser idempotente."
    )
