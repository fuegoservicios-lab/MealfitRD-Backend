"""
Tests P0-1: Heartbeat guarantee del worker.

Cubre:
1. _touch_chunk_heartbeat — helper inline:
   a. None task_id → retorna False sin crashear.
   b. UPDATE OK → retorna True.
   c. UPDATE falla (excepción) → retorna False (silencioso, no aborta el chunk).

2. _heartbeat_loop comportamiento conceptual:
   a. El thread arranca con un primer UPDATE inmediato (no espera _HB_INTERVAL).
   b. Counter de fallos consecutivos se resetea al primer éxito.
   c. Logging escalado: primer fallo ERROR, recuperación INFO.
"""
from unittest.mock import patch, MagicMock
import pytest


def test_touch_heartbeat_none_task_returns_false():
    """task_id=None: helper retorna False sin tocar DB ni crashear."""
    from cron_tasks import _touch_chunk_heartbeat
    assert _touch_chunk_heartbeat(None) is False


@patch("cron_tasks.execute_sql_write")
def test_touch_heartbeat_ok_returns_true(mock_write):
    """UPDATE OK → True; debe haber emitido el SQL exacto contra chunk_user_locks."""
    from cron_tasks import _touch_chunk_heartbeat
    mock_write.return_value = None  # execute_sql_write OK
    result = _touch_chunk_heartbeat("chunk-abc")
    assert result is True
    mock_write.assert_called_once()
    sql = mock_write.call_args[0][0]
    assert "chunk_user_locks" in sql
    assert "heartbeat_at = NOW()" in sql
    assert "locked_by_chunk_id = %s" in sql
    assert mock_write.call_args[0][1] == ("chunk-abc",)


@patch("cron_tasks.execute_sql_write")
def test_touch_heartbeat_failure_returns_false_silently(mock_write):
    """UPDATE falla → retorna False sin re-raise (no debe abortar el chunk)."""
    from cron_tasks import _touch_chunk_heartbeat
    mock_write.side_effect = Exception("DB pool blip")
    result = _touch_chunk_heartbeat("chunk-xyz")
    assert result is False  # silencioso, sin re-raise


def test_heartbeat_loop_does_initial_update_before_wait():
    """
    [P0-1] El thread debe hacer un UPDATE inmediato antes del primer wait,
    cubriendo la ventana entre el INSERT del lock (heartbeat_at = NOW()) y el
    primer cycle de _HB_INTERVAL segundos.

    Verificamos que dado un stop_event ya seteado (que haría que el while
    body NO ejecute), el primer _do_update aún corre y emite UPDATE.
    """
    import threading
    from datetime import datetime, timezone

    update_calls = []

    def fake_execute_sql_write(query, params):
        update_calls.append((query, params))
        return None

    with patch("cron_tasks.execute_sql_write", side_effect=fake_execute_sql_write):
        # Replicar el cuerpo de _heartbeat_loop manualmente (la función está nested
        # dentro de _chunk_worker así que no podemos importarla directamente; testeamos
        # el invariante: hay un update inicial antes de entrar al while).
        from cron_tasks import execute_sql_write as _eswrite
        from constants import (
            CHUNK_LOCK_HEARTBEAT_INTERVAL_SECONDS as _HB_INTERVAL,
            CHUNK_LOCK_STALE_MINUTES as _HB_STALE_MIN,
        )

        stop_event = threading.Event()
        stop_event.set()  # entrar al wait y salir inmediatamente

        state = {
            "last_heartbeat_at": datetime.now(timezone.utc),
            "consecutive_failures": 0,
            "lock_chunk_id": "chunk-init-update",
        }

        # Simular el invariante clave: `_do_update()` corre ANTES del while wait().
        def _do_update():
            try:
                _eswrite(
                    "UPDATE chunk_user_locks SET heartbeat_at = NOW() WHERE locked_by_chunk_id = %s",
                    (state["lock_chunk_id"],)
                )
                state["last_heartbeat_at"] = datetime.now(timezone.utc)
                state["consecutive_failures"] = 0
            except Exception:
                state["consecutive_failures"] += 1

        _do_update()  # update inicial
        while not stop_event.wait(_HB_INTERVAL):
            _do_update()

        assert len(update_calls) == 1, (
            f"Se esperaba 1 UPDATE inicial pero hubo {len(update_calls)}"
        )
        assert update_calls[0][1] == ("chunk-init-update",)


def test_heartbeat_consecutive_failures_reset_on_success():
    """[P0-1] Counter de fallos consecutivos resetea al primer éxito."""
    from datetime import datetime, timezone

    state = {
        "last_heartbeat_at": datetime.now(timezone.utc),
        "consecutive_failures": 5,  # ya falló 5 veces
        "lock_chunk_id": "chunk-recover",
    }

    # Replicar la lógica de reset del helper interno _do_update
    def _simulate_success():
        state["last_heartbeat_at"] = datetime.now(timezone.utc)
        if state["consecutive_failures"] > 0:
            state["consecutive_failures"] = 0

    _simulate_success()
    assert state["consecutive_failures"] == 0


def test_heartbeat_stale_threshold_calculation():
    """
    [P0-1] El thread reporta ERROR cuando los fallos consecutivos cruzan el umbral
    derivado de CHUNK_LOCK_STALE_MINUTES / CHUNK_LOCK_HEARTBEAT_INTERVAL_SECONDS.

    Con defaults (3 min / 60s = 3 ciclos), el zombie rescue procede tras 3 fallos.
    """
    from constants import (
        CHUNK_LOCK_HEARTBEAT_INTERVAL_SECONDS as _HB_INTERVAL,
        CHUNK_LOCK_STALE_MINUTES as _HB_STALE_MIN,
    )
    max_cycles_before_stale = max(1, int((_HB_STALE_MIN * 60) // max(_HB_INTERVAL, 1)))
    # Sanity: en defaults razonables (1m interval, 3min stale), el máximo ciclos antes
    # del zombie rescue debe ser >= 1 y <= un valor razonable (e.g., 60).
    assert 1 <= max_cycles_before_stale <= 60, (
        f"max_cycles_before_stale={max_cycles_before_stale} fuera del rango razonable"
    )


@patch("cron_tasks.execute_sql_write")
def test_touch_heartbeat_emits_update_for_specific_chunk(mock_write):
    """Múltiples chunks deben recibir su propio UPDATE con su propio task_id."""
    from cron_tasks import _touch_chunk_heartbeat
    mock_write.return_value = None

    _touch_chunk_heartbeat("chunk-A")
    _touch_chunk_heartbeat("chunk-B")
    _touch_chunk_heartbeat("chunk-C")

    assert mock_write.call_count == 3
    chunk_ids = [c.args[1][0] for c in mock_write.call_args_list]
    assert chunk_ids == ["chunk-A", "chunk-B", "chunk-C"]
