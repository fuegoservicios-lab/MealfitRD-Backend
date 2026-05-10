"""[P1-HIST-AUDIT-6 · 2026-05-09] Tests static analysis sobre la
migración SSOT que aplica `FORCE ROW LEVEL SECURITY` a `meal_plans`
y `chunk_lesson_telemetry`.

Bug original (audit historial 2026-05-08):
    Ambas tablas tenían `relrowsecurity=true` pero
    `relforcerowsecurity=false`. Las otras tablas chunk_*
    (`plan_chunk_queue`, `chunk_user_locks`, `plan_chunk_metrics`,
    `chunk_deferrals`) ya tenían FORCE; estas dos eran
    inconsistentes — un table OWNER (sin BYPASSRLS) podía bypaseasr
    las policies.

Fix:
    Migración SSOT
    `supabase/migrations/p1_hist_audit_6_rls_force.sql` con
    `ALTER TABLE ... FORCE ROW LEVEL SECURITY` para ambas tablas.
    Idempotente vía DO block + check `relforcerowsecurity`.
    Aplicada al remoto (project mpoodlmnzaeuuazsazbj) y verificada.

Cobertura (static analysis del SQL):
    - Migración existe en `supabase/migrations/`.
    - Header documenta el marker P1-HIST-AUDIT-6.
    - SQL aplica FORCE a AMBAS tablas (positivo).
    - SQL es idempotente (check `relforcerowsecurity` antes de ALTER).
    - SQL NO toca otras tablas chunk_* (no scope creep).
    - Comentario documenta el caso edge cubierto.

NO incluido (out of scope test contra DB real):
    - Verificación runtime que `pg_class.relforcerowsecurity = true`
      tras aplicar — eso requiere conexión Postgres y/o testcontainer.
      La aplicación al remoto se documenta vía MCP Supabase
      (`apply_migration` exitoso + `SELECT relforcerowsecurity`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_MIGRATION_PATH = (
    _BACKEND_ROOT.parent
    / "supabase" / "migrations"
    / "p1_hist_audit_6_rls_force.sql"
)


# ---------------------------------------------------------------------------
# 1. Migración existe y tiene el marker
# ---------------------------------------------------------------------------
def test_migration_file_exists():
    assert _MIGRATION_PATH.exists(), (
        f"No se encontró la migración P1-HIST-AUDIT-6 en "
        f"{_MIGRATION_PATH}. Path esperado por convención del repo "
        f"(SSOT migrations en supabase/migrations/)."
    )


def _migration_text() -> str:
    return _MIGRATION_PATH.read_text(encoding="utf-8")


def test_marker_present_in_header():
    text = _migration_text()
    assert "P1-HIST-AUDIT-6" in text, (
        "La migración debe contener el marker `P1-HIST-AUDIT-6` para "
        "trazabilidad cross-archivo (memoria, app.py:_LAST_KNOWN_PFIX)."
    )


# ---------------------------------------------------------------------------
# 2. Aplicación FORCE a ambas tablas
# ---------------------------------------------------------------------------
def test_alter_force_meal_plans():
    text = _migration_text()
    assert re.search(
        r"ALTER\s+TABLE\s+public\.meal_plans\s+FORCE\s+ROW\s+LEVEL\s+SECURITY",
        text,
        re.IGNORECASE,
    ), (
        "Migración no contiene `ALTER TABLE public.meal_plans FORCE ROW "
        "LEVEL SECURITY`. P1-HIST-AUDIT-6 cierra el gap de meal_plans "
        "que tenía RLS habilitado pero NO forzado."
    )


def test_alter_force_chunk_lesson_telemetry():
    text = _migration_text()
    assert re.search(
        r"ALTER\s+TABLE\s+public\.chunk_lesson_telemetry\s+FORCE\s+ROW\s+LEVEL\s+SECURITY",
        text,
        re.IGNORECASE,
    ), (
        "Migración no contiene `ALTER TABLE public.chunk_lesson_telemetry "
        "FORCE ROW LEVEL SECURITY`. P1-HIST-AUDIT-6 cubre AMBAS tablas."
    )


# ---------------------------------------------------------------------------
# 3. Idempotencia: el ALTER se hace dentro de un check
#    `relforcerowsecurity = true` para no fallar al re-aplicar.
# ---------------------------------------------------------------------------
def test_migration_is_idempotent():
    """La migración debe poder re-aplicarse sin error. Patrón:
    DO block que checa `pg_class.relforcerowsecurity` antes del ALTER.
    """
    text = _migration_text()
    # Aceptamos cualquiera de los dos patrones de idempotencia:
    #   - check `relforcerowsecurity = true` antes del ALTER.
    #   - el ALTER FORCE de Postgres es naturalmente idempotente, pero
    #     el patrón explícito documenta intención.
    assert "relforcerowsecurity" in text, (
        "Migración no referencia `relforcerowsecurity` para check de "
        "idempotencia. Patrón canónico del repo (mismo que "
        "p0_hist_3_telemetry_orphan_fk.sql)."
    )


# ---------------------------------------------------------------------------
# 4. Scope guard: solo las dos tablas del audit
# ---------------------------------------------------------------------------
def test_no_scope_creep_to_other_chunk_tables():
    """Las otras tablas chunk_* (`plan_chunk_queue`, `chunk_user_locks`,
    `plan_chunk_metrics`, `chunk_deferrals`) YA tenían FORCE antes de
    P1-HIST-AUDIT-6 — la migración no debe tocarlas (cero scope
    creep, mantiene la migración revertible si surge problema).
    """
    text = _migration_text()
    forbidden_tables = [
        "plan_chunk_queue",
        "chunk_user_locks",
        "plan_chunk_metrics",
        "chunk_deferrals",
    ]
    for table in forbidden_tables:
        # Permitimos menciones en comentarios (-- ... plan_chunk_queue ...),
        # pero NO ALTER TABLE sobre ellas.
        assert not re.search(
            rf"ALTER\s+TABLE[^;]*\b{re.escape(table)}\b",
            text,
            re.IGNORECASE,
        ), (
            f"La migración P1-HIST-AUDIT-6 NO debe modificar `{table}` "
            f"(ya tenía FORCE pre-audit, scope creep introduce riesgo)."
        )


# ---------------------------------------------------------------------------
# 5. Comentario documental — describe POR QUÉ y caso edge cubierto
# ---------------------------------------------------------------------------
def test_migration_documents_bypassrls_assumption():
    """El comentario debe explicar que el backend conecta como `postgres`
    (BYPASSRLS=true) y sigue funcionando — sin esto, un futuro
    mantenedor podría asumir que FORCE rompe el backend y revertir.
    """
    text = _migration_text()
    assert "BYPASSRLS" in text or "bypassrls" in text, (
        "La migración debe documentar que los roles backend "
        "(postgres, service_role) tienen BYPASSRLS=true y siguen "
        "funcionando. Sin ese anchor, la decisión es opaca."
    )


def test_migration_applies_comment_on_table():
    """El COMMENT ON TABLE deja una huella visible en `pg_description`
    para diagnóstico SQL futuro: `SELECT obj_description(...)` muestra
    por qué el FORCE está activo.
    """
    text = _migration_text()
    assert re.search(
        r"COMMENT\s+ON\s+TABLE\s+public\.meal_plans",
        text,
        re.IGNORECASE,
    )
    assert re.search(
        r"COMMENT\s+ON\s+TABLE\s+public\.chunk_lesson_telemetry",
        text,
        re.IGNORECASE,
    )
