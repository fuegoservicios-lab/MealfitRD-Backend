"""[P2-4 · 2026-05-10] Regression guard: `chunk_deferrals.created_at`
documenta su semántica (buffer time, no flush time) y el flush counta
el fallback cuando `buffered_at` falta.

Bug raíz (audit 2026-05-10):
    La migración previa `20260503231237_rename_chunk_deferrals_deferred_at_to_created_at`
    renombró la columna sin documentar la semántica. Los call sites del
    flush en `cron_tasks._flush_pending_deferrals` usan
    `COALESCE(rec.buffered_at, NOW())` — el valor de la columna refleja:
      - tiempo del defer event original cuando `buffered_at` está en el
        record JSONL (caso normal: los `_record_*` siempre lo setean), o
      - tiempo del flush (NOW()) cuando el record legacy carece de la key
        (caso degradado aceptable, pero invisible al operador).

    El GC (`_gc_orphan_chunk_telemetry`) filtra por `created_at < ...`
    asumiendo semántica de "tiempo del evento". Bajo data drift legacy,
    la edad reportada es 0 → GC preserva más de lo correcto.

Fix dual:
    A) Migración `p2_4_chunk_deferrals_created_at_comment.sql` añade
       `COMMENT ON COLUMN public.chunk_deferrals.created_at IS '...'`
       explicando la semántica + el fallback.
    B) `_flush_pending_deferrals` incrementa `stats["fallback_to_now"]`
       cada vez que un record sin `buffered_at` flushea, y emite WARN
       al final si el counter > 0 — el operador detecta callsite que
       olvidó setear la key.

Cobertura de este test (parser-based, no DB):
    1. La migración existe y declara el COMMENT con marker P2-4.
    2. El COMMENT cita `buffered_at` (key in-memory) y `_flush_pending_deferrals`
       (función responsable).
    3. La función de flush incrementa `stats["fallback_to_now"]` cuando
       `buffered_at` falta.
    4. El log final emite WARN con `[P2-4/FLUSH-DEFERRALS]` cuando
       el counter > 0.

Out of scope:
    - Smoke runtime contra DB real (verificar via `\d+ chunk_deferrals` ya
      hecho durante la aplicación de la migración).
    - Rename de la columna a `buffered_at`: pospuesto por costo (5+ call
      sites + cron GC + test sibling). El COMMENT desbloquea sin tocar
      contrato.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATION_PATH = _REPO_ROOT / "migrations" / "p2_4_chunk_deferrals_created_at_comment.sql"
_CRON_TASKS_PATH = _REPO_ROOT / "backend" / "cron_tasks.py"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo requerido no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Migración SSOT con COMMENT
# ---------------------------------------------------------------------------
def test_migration_file_exists():
    assert _MIGRATION_PATH.exists(), (
        f"Migración SSOT debe vivir en {_MIGRATION_PATH}."
    )


def test_migration_declares_comment_on_column():
    sql = _read(_MIGRATION_PATH)
    assert re.search(
        r"COMMENT\s+ON\s+COLUMN\s+public\.chunk_deferrals\.created_at",
        sql, re.IGNORECASE,
    ), (
        "Migración debe declarar `COMMENT ON COLUMN public.chunk_deferrals.created_at`."
    )


def test_comment_cites_buffered_at_semantic():
    """El COMMENT debe mencionar `buffered_at` para que el operador
    entienda la dualidad in-memory vs columna física."""
    sql = _read(_MIGRATION_PATH)
    assert "buffered_at" in sql, (
        "COMMENT debe citar `buffered_at` — sin esto, el operador no "
        "sabe que la columna recibe el timestamp in-memory."
    )


def test_comment_cites_flush_function():
    sql = _read(_MIGRATION_PATH)
    assert "_flush_pending_deferrals" in sql, (
        "COMMENT debe nombrar `_flush_pending_deferrals` para que un "
        "operador que inspecciona el schema sepa qué código gobierna "
        "la semántica de la columna."
    )


def test_comment_has_marker():
    sql = _read(_MIGRATION_PATH)
    assert "P2-4" in sql, (
        "COMMENT debe incluir el marker P2-4 para trazabilidad cross-"
        "archivo (memoria, app.py:_LAST_KNOWN_PFIX)."
    )


# ---------------------------------------------------------------------------
# 2. Counter del fallback en el flush
# ---------------------------------------------------------------------------
def test_flush_increments_fallback_counter():
    """Cuando `rec.get('buffered_at')` es falsy, debe incrementarse
    `stats['fallback_to_now']`."""
    src = _read(_CRON_TASKS_PATH)
    # Encontrar el bloque del flush (entre `_flush_pending_deferrals` def y
    # el siguiente def).
    func_match = re.search(
        r"def\s+_flush_pending_deferrals\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    assert func_match is not None, "_flush_pending_deferrals no encontrado."
    body = func_match.group(0)
    assert 'stats["fallback_to_now"]' in body or "stats['fallback_to_now']" in body, (
        "`_flush_pending_deferrals` debe incrementar `stats[\"fallback_to_now\"]` "
        "cuando un record sin `buffered_at` flushea. Sin el counter, el "
        "fallback queda invisible al operador."
    )


def test_flush_warns_when_fallback_counter_nonzero():
    """Al final del flush, si `stats['fallback_to_now'] > 0`, debe
    emitirse `logger.warning` con prefijo `[P2-4/FLUSH-DEFERRALS]`."""
    src = _read(_CRON_TASKS_PATH)
    func_match = re.search(
        r"def\s+_flush_pending_deferrals\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    body = func_match.group(0)
    assert "[P2-4/FLUSH-DEFERRALS]" in body, (
        "`_flush_pending_deferrals` debe emitir log con prefijo "
        "`[P2-4/FLUSH-DEFERRALS]` cuando el counter de fallback > 0. "
        "Sin esto, el operador no detecta callsite roto."
    )
    # Y debe ser logger.warning (no debug ni info — buscamos visibilidad).
    assert re.search(
        r"logger\.warning\([^)]*\[P2-4/FLUSH-DEFERRALS\]",
        body, re.DOTALL,
    ), (
        "El log de fallback debe ser `logger.warning` (no debug/info) — "
        "para que aparezca en alertas operacionales."
    )
