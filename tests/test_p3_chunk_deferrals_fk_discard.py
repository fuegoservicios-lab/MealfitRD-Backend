"""
[P3-CHUNK-DEFERRALS-FK-DISCARD · 2026-05-18]

Audit Supabase MCP 2026-05-19 detectó burst de ~100 ERRORs/25s en logs Postgres:

    insert or update on table "chunk_deferrals"
    violates foreign key constraint "chunk_deferrals_meal_plan_id_fkey"

Causa raíz: `_flush_pending_deferrals` (cron_tasks.py:14937-14953) descartaba
records con `"violates not-null"` o `"invalid input syntax"` pero NO con
`"violates foreign key"`. Modo de fallo concreto:

  1. Usuario hace reset cuenta (P3-RESET-SINGLE-TXN borra `meal_plans` en txn).
  2. Buffer jsonl local del worker conserva records de deferrals huérfanos
     cuyo `meal_plan_id` ya no existe.
  3. Cron `_flush_pending_deferrals` (cada 5 min) reintenta el INSERT.
  4. FK violation lanza Exception. El guard NO la matchea → record se
     re-bufferea en `remaining_records` → loop infinito.

Fix: extender el guard para descartar FK violations también — el parent plan
nunca regresará, retry no se recupera.

Tests:
  - parser-based: anchor del marker + presencia de "violates foreign key"
    en la condición de descarte.
  - funcional: simular Exception con texto "violates foreign key constraint"
    y verificar que el contador `discarded_invalid` incrementa.
"""
import json
import re
from pathlib import Path
from unittest.mock import patch
import pytest


def _cron_tasks_source() -> str:
    return Path(__file__).resolve().parent.parent.joinpath("cron_tasks.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# parser-based
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """El marker textual debe estar en el archivo de prod para que renombres lo rompan."""
    src = _cron_tasks_source()
    assert "P3-CHUNK-DEFERRALS-FK-DISCARD" in src, (
        "Marker P3-CHUNK-DEFERRALS-FK-DISCARD ausente. Si refactorizaste el "
        "guard del flush, actualizá el marker antes del rename."
    )


def test_flush_discards_fk_violation():
    """La condición de descarte del flush debe matchear `violates foreign key`."""
    src = _cron_tasks_source()
    body = src[src.index("def _flush_pending_deferrals"):]
    body = body[: body.index("\ndef ")]
    assert '"violates foreign key" in _err_msg' in body, (
        "El guard del flush debe descartar FK violations — sin esto, records "
        "huérfanos post-reset-cuenta loopean infinitamente en el buffer y "
        "saturan logs Postgres (burst observado 2026-05-19 02:46 UTC)."
    )
    assert '"violates not-null" in _err_msg' in body, "Regresión: NOT NULL guard removido."
    assert '"invalid input syntax" in _err_msg' in body, "Regresión: syntax guard removido."


# ---------------------------------------------------------------------------
# funcional
# ---------------------------------------------------------------------------
def test_flush_fk_violation_discards_record(tmp_path):
    """Un Exception con texto de FK violation debe incrementar `discarded_invalid`
    y NO mantener el record en el buffer (no re-loop)."""
    import cron_tasks

    buffer_path = tmp_path / "deferrals_pending.jsonl"
    record = {
        "user_id": "11111111-1111-1111-1111-111111111111",
        "meal_plan_id": "22222222-2222-2222-2222-222222222222",
        "week_number": 2,
        "reason": "temporal_gate",
        "days_until_prev_end": 2,
        "buffered_at": "2026-05-18T00:00:00+00:00",
    }
    buffer_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    fk_exc = Exception(
        'insert or update on table "chunk_deferrals" violates foreign key '
        'constraint "chunk_deferrals_meal_plan_id_fkey"'
    )

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", str(buffer_path)):
        with patch.object(cron_tasks, "execute_sql_write", side_effect=fk_exc):
            stats = cron_tasks._flush_pending_deferrals()

    assert stats["discarded_invalid"] == 1, (
        f"FK violation debió descartar el record (discarded_invalid=1); stats={stats}"
    )
    assert stats["flushed"] == 0
    # El record huérfano NO debe seguir en el buffer.
    leftover = buffer_path.read_text(encoding="utf-8").strip() if buffer_path.exists() else ""
    assert leftover == "", (
        f"Buffer debió quedar vacío tras descartar FK violation; contiene: {leftover!r}"
    )


def test_flush_transient_error_keeps_record_for_retry(tmp_path):
    """Errores de conexión/timeout NO deben dispararse en el guard de descarte —
    el record se mantiene para retry. Regresión guard."""
    import cron_tasks

    buffer_path = tmp_path / "deferrals_pending.jsonl"
    record = {
        "user_id": "11111111-1111-1111-1111-111111111111",
        "meal_plan_id": "22222222-2222-2222-2222-222222222222",
        "week_number": 2,
        "reason": "temporal_gate",
        "days_until_prev_end": 2,
        "buffered_at": "2026-05-18T00:00:00+00:00",
    }
    buffer_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    transient_exc = Exception("connection refused: server closed the connection unexpectedly")

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", str(buffer_path)):
        with patch.object(cron_tasks, "execute_sql_write", side_effect=transient_exc):
            stats = cron_tasks._flush_pending_deferrals()

    assert stats["discarded_invalid"] == 0, (
        f"Error transitorio NO debe ser descartado; stats={stats}"
    )
    assert stats["flushed"] == 0
    leftover = buffer_path.read_text(encoding="utf-8").strip() if buffer_path.exists() else ""
    assert leftover != "", "Buffer debió mantener el record para retry de error transitorio."
