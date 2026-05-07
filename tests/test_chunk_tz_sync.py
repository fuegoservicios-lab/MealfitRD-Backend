"""
[P0-4] Tests para `_sync_chunk_queue_tz_offsets`.

Valida que:
  A. Drift por debajo del threshold se ignora (no UPDATE).
  B. Drift por encima del threshold reescribe tzOffset, tz_offset_minutes Y desplaza
     execute_after por delta_minutos = live_tz - snapshot_tz.
  C. target_user_id limita el SELECT a un usuario (fast-path desde
     update_user_health_profile en db_profiles.py).
  D. Sin live_tz (NULL en user_profile) la fila se ignora silenciosamente.
"""
import sys
from unittest.mock import patch, MagicMock

sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())


def _row(chunk_id, snapshot_tz, live_tz):
    return {
        "id": chunk_id,
        "user_id": "user-abc",
        "execute_after": "2026-05-01T17:00:00+00:00",
        "snapshot_tz": snapshot_tz,
        "live_tz": live_tz,
    }


def test_a_drift_below_threshold_is_ignored():
    """[P0-4] Drift de 5 min < threshold 15 → ningún UPDATE."""
    rows = [_row("chunk-1", snapshot_tz=-240, live_tz=-235)]  # drift = 5

    write_calls = []
    def fake_write(sql, params=None, returning=False):
        write_calls.append((sql, params))
        return None

    with patch("cron_tasks.execute_sql_query", return_value=rows), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        from cron_tasks import _sync_chunk_queue_tz_offsets
        updated = _sync_chunk_queue_tz_offsets()

    assert updated == 0
    assert write_calls == [], "Drift por debajo del threshold no debe disparar UPDATE"


def test_b_drift_above_threshold_writes_new_tz_and_shifts_execute_after():
    """[P0-4] Drift de 60 min ≥ threshold 15 → UPDATE con delta y nuevo tz."""
    rows = [_row("chunk-1", snapshot_tz=-240, live_tz=-300)]  # drift = 60, viaje al oeste

    write_calls = []
    def fake_write(sql, params=None, returning=False):
        write_calls.append((sql, params))
        return None

    with patch("cron_tasks.execute_sql_query", return_value=rows), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        from cron_tasks import _sync_chunk_queue_tz_offsets
        updated = _sync_chunk_queue_tz_offsets()

    assert updated == 1
    assert len(write_calls) == 1
    sql, params = write_calls[0]
    assert "UPDATE plan_chunk_queue" in sql
    assert "tzOffset" in sql
    assert "tz_offset_minutes" in sql
    assert "execute_after" in sql
    # params: (live_tz, live_tz, delta_minutes, chunk_id)
    assert params[0] == -300, f"Primer param debe ser live_tz=-300, fue {params[0]}"
    assert params[1] == -300
    delta = params[2]
    assert delta == -60, f"delta_minutes debe ser live - snapshot = -300 - (-240) = -60, fue {delta}"
    assert params[3] == "chunk-1"


def test_c_target_user_id_limits_select():
    """[P0-4] target_user_id añade 'AND q.user_id = %s' al SELECT (fast-path)."""
    captured = {"sql": None, "params": None}

    def fake_query(sql, params=None, fetch_all=False, fetch_one=False):
        captured["sql"] = sql
        captured["params"] = params
        return []

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write"):
        from cron_tasks import _sync_chunk_queue_tz_offsets
        _sync_chunk_queue_tz_offsets(target_user_id="user-xyz")

    assert "AND q.user_id = %s" in captured["sql"], \
        "target_user_id debe restringir el SELECT a ese usuario"
    assert captured["params"] == ("user-xyz",)


def test_d_null_live_tz_skips_row():
    """[P0-4] Filas con live_tz=NULL (perfil sin tz_offset_minutes) se saltan sin UPDATE."""
    rows = [_row("chunk-1", snapshot_tz=-240, live_tz=None)]

    write_calls = []
    with patch("cron_tasks.execute_sql_query", return_value=rows), \
         patch("cron_tasks.execute_sql_write", side_effect=lambda *a, **k: write_calls.append(a)):
        from cron_tasks import _sync_chunk_queue_tz_offsets
        updated = _sync_chunk_queue_tz_offsets()

    assert updated == 0
    assert write_calls == []


def test_e_null_snapshot_tz_persists_live_without_shift():
    """[P0-4] Filas con snapshot_tz=NULL pero live_tz definido: persistir live_tz al
    snapshot SIN desplazar execute_after (delta desconocido). Caso edge real visto
    en producción (chunk encolado por path antiguo sin tz_offset)."""
    rows = [_row("chunk-1", snapshot_tz=None, live_tz=240)]

    write_calls = []
    def fake_write(sql, params=None, returning=False):
        write_calls.append((sql, params))
        return None

    with patch("cron_tasks.execute_sql_query", return_value=rows), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        from cron_tasks import _sync_chunk_queue_tz_offsets
        updated = _sync_chunk_queue_tz_offsets()

    # No cuenta como "drift fixed" en el contador (no hay shift)
    assert updated == 0, "Backfill de snapshot_tz NULL no se cuenta como drift fix"
    # Debe haber UPDATE de tzOffset / tz_offset_minutes pero SIN execute_after
    assert len(write_calls) == 1
    sql, params = write_calls[0]
    assert "tzOffset" in sql and "tz_offset_minutes" in sql
    assert "execute_after" not in sql, "No debe shiftear execute_after cuando snapshot_tz era NULL"
    assert params[0] == 240
    assert params[1] == 240
    assert params[2] == "chunk-1"
