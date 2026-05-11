"""[P1-5 · 2026-05-10] Regression guard: `chunk_lesson_telemetry.event`
tiene CHECK constraint de formato como backstop runtime contra typos.

Bug original (audit 2026-05-10):
    La columna `event` era `text` libre sin validación de formato. El
    meta-test `test_p1_hist_audit_5_lesson_event_whitelist.py` ya enforza
    drift semántico vía parsing de cron_tasks.py, pero esa defensa solo
    cubre call sites con literales estáticos. Modos NO cubiertos:
      - F-strings dinámicos: `event=f"lesson_{kind}"` con `kind` runtime.
      - Call sites en módulos no parseados (admin tools, REPL).
      - Hot patches del operador via SQL editor directo.
      - Typos en code-paths sin tests.
    Resultado: un INSERT con `event="Lesson Synthesized"` o
    `event="lesson_synthesized_low_confidence "` (trailing space) pasaba
    silenciosamente — `/lessons-counts` filtra `event = ANY(%s)` exacto,
    así que el row queda invisible.

Fix:
    Migración `p1_5_chunk_lesson_event_format_check.sql` añade:
      CHECK (event ~ '^[a-z][a-z0-9_]+$' AND length(event) BETWEEN 1 AND 100)
    Atrapa runtime: capitales, espacios, hyphens, vacío, longitud
    excesiva, prefix numérico. NO valida valores específicos (esa es
    semántica, sigue en el meta-test parser-based) — solo formato.

Cobertura de este test (parser-based, no DB):
    1. La migración SSOT existe.
    2. Declara la constraint `chunk_lesson_telemetry_event_format`.
    3. Usa la regex canónica `^[a-z][a-z0-9_]+$`.
    4. Incluye el length bound `BETWEEN 1 AND 100`.
    5. Tiene COMMENT que apunta al test cross-language existente.

Out of scope:
    - Test runtime que ejecute INSERT inválido contra DB y verifique
      check_violation. Hecho como smoke durante la aplicación de la
      migración; no se re-corre en CI (requiere DB real).
    - Lista exacta de valores válidos (enum CHECK). Decisión:
      mantener el formato como gate y delegar semántica al meta-test
      P1-HIST-AUDIT-5 — la lista crece sin fricción de migraciones.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATION_PATH = _REPO_ROOT / "supabase" / "migrations" / "p1_5_chunk_lesson_event_format_check.sql"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo requerido no encontrado: {path}"
    return path.read_text(encoding="utf-8")


def test_migration_file_exists():
    assert _MIGRATION_PATH.exists(), (
        f"Migración SSOT debe vivir en {_MIGRATION_PATH}."
    )


def test_migration_adds_constraint():
    sql = _read(_MIGRATION_PATH)
    assert re.search(
        r"ADD\s+CONSTRAINT\s+chunk_lesson_telemetry_event_format",
        sql, re.IGNORECASE,
    ), (
        "Migración debe declarar `ADD CONSTRAINT chunk_lesson_telemetry_event_format`. "
        "Sin esto el backstop runtime no existe."
    )


def test_migration_uses_canonical_regex():
    """La regex canónica matchea lo que el resto del sistema asume sobre
    `event` (lowercase + dígitos + underscore, empieza por letra)."""
    sql = _read(_MIGRATION_PATH)
    # `event ~ '^[a-z][a-z0-9_]+$'`
    assert re.search(
        r"event\s*~\s*[\"']\^\[a-z\]\[a-z0-9_\]\+\$[\"']",
        sql,
    ), (
        "CHECK debe usar la regex canónica `^[a-z][a-z0-9_]+$`. "
        "Cualquier otra forma (case-insensitive, hyphens permitidos) "
        "introduce divergencia con _LESSON_COUNT_EVENT_WHITELIST."
    )


def test_migration_enforces_length_bounds():
    sql = _read(_MIGRATION_PATH)
    assert re.search(
        r"length\(event\)\s+BETWEEN\s+1\s+AND\s+100",
        sql, re.IGNORECASE,
    ), (
        "CHECK debe incluir `length(event) BETWEEN 1 AND 100`. Sin esto, "
        "un string vacío matchea la regex prefix `^[a-z]` por convención "
        "Postgres y un string de 10000 caracteres pasa silenciosamente."
    )


def test_migration_drops_constraint_if_exists():
    """Idempotencia: la migración puede re-correrse sin error duplicado.
    Patrón `DROP CONSTRAINT IF EXISTS` antes del `ADD CONSTRAINT`."""
    sql = _read(_MIGRATION_PATH)
    assert re.search(
        r"DROP\s+CONSTRAINT\s+IF\s+EXISTS\s+chunk_lesson_telemetry_event_format",
        sql, re.IGNORECASE,
    ), (
        "Migración debe ser idempotente vía `DROP CONSTRAINT IF EXISTS` "
        "antes de `ADD CONSTRAINT`."
    )


def test_migration_documents_relationship_to_semantic_test():
    """El COMMENT debe apuntar al meta-test P1-HIST-AUDIT-5 para que un
    futuro operador entienda que esto es DEFENSA-EN-PROFUNDIDAD, no
    el gate primario."""
    sql = _read(_MIGRATION_PATH)
    assert "test_p1_hist_audit_5_lesson_event_whitelist" in sql, (
        "COMMENT del constraint debe referenciar "
        "`test_p1_hist_audit_5_lesson_event_whitelist` (el gate semántico "
        "que vive a nivel de CI). Sin este cross-link, un operador puede "
        "asumir que esta CHECK es el único gate."
    )
