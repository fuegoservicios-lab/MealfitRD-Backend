"""[P1-4] Tests para `_cas_update_chunk_status`.

Antes, varios paths terminales (failed_pantry_violation, last-resort failed,
user_lock_pending) hacían `UPDATE … WHERE id = %s` sin verificar que el chunk
seguía siendo "nuestro". Si el zombie rescue había incrementado `attempts` y
un worker B ya había re-claim el chunk, nuestra UPDATE clobbearía el estado
del worker B (e.g., marcando 'failed' un chunk que B está procesando).

El helper `_cas_update_chunk_status` aplica el patrón CAS:
    UPDATE plan_chunk_queue
    SET status = <new>, ...
    WHERE id = %s AND attempts = %s AND status = %s

Si el guard falla, el caller sabe que fue desplazado y debe abortar limpiamente.
"""
from unittest.mock import patch


def test_cas_update_succeeds_when_attempts_match():
    """Caso normal: attempts y status coinciden → UPDATE afecta 1 fila."""
    from cron_tasks import _cas_update_chunk_status

    captured = {}

    def fake_write(sql, params, returning=False):
        captured["sql"] = sql
        captured["params"] = params
        captured["returning"] = returning
        return [{"id": "task-1"}]

    with patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        ok = _cas_update_chunk_status("task-1", expected_attempts=2, new_status="failed")

    assert ok is True
    assert captured["returning"] is True
    sql = captured["sql"]
    assert "WHERE id = %s AND attempts = %s AND status = %s" in sql
    assert "RETURNING id" in sql
    assert captured["params"] == ("failed", "task-1", 2, "processing")


def test_cas_update_returns_false_when_attempts_mismatch():
    """Zombie rescue desplazó el chunk → 0 filas → False, no clobbear."""
    from cron_tasks import _cas_update_chunk_status

    with patch("cron_tasks.execute_sql_write", return_value=[]):
        ok = _cas_update_chunk_status("task-1", expected_attempts=2, new_status="failed")

    assert ok is False


def test_cas_update_returns_false_on_db_exception():
    """Si la UPDATE lanza, devolvemos False — no clobbear estado bajo ninguna circunstancia."""
    from cron_tasks import _cas_update_chunk_status

    with patch("cron_tasks.execute_sql_write",
               side_effect=RuntimeError("connection lost")):
        ok = _cas_update_chunk_status("task-1", expected_attempts=2, new_status="failed")

    assert ok is False


def test_cas_update_with_extra_set_clauses():
    """Soporta SET clauses adicionales (e.g., escalated_at, quality_tier)."""
    from cron_tasks import _cas_update_chunk_status

    captured = {}

    def fake_write(sql, params, returning=False):
        captured["sql"] = sql
        return [{"id": "task-2"}]

    with patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        ok = _cas_update_chunk_status(
            "task-2", expected_attempts=1, new_status="cancelled",
            extra_set_clauses={"escalated_at": "NOW()"},
        )

    assert ok is True
    sql = captured["sql"]
    assert "escalated_at = NOW()" in sql
    assert "status = %s" in sql


def test_cas_update_custom_expected_status():
    """Permite verificar transiciones desde un status distinto a 'processing'."""
    from cron_tasks import _cas_update_chunk_status

    captured = {}

    def fake_write(sql, params, returning=False):
        captured["params"] = params
        return [{"id": "task-3"}]

    with patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        _cas_update_chunk_status(
            "task-3", expected_attempts=0, new_status="cancelled",
            expected_status="pending",
        )

    assert captured["params"][-1] == "pending"


def test_cas_update_does_not_swallow_returning_value_change():
    """Confirma que RETURNING id va presente en el SQL."""
    from cron_tasks import _cas_update_chunk_status

    captured = {}

    def fake_write(sql, params, returning=False):
        captured["sql"] = sql
        captured["returning"] = returning
        return [{"id": "task-4"}]

    with patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        _cas_update_chunk_status("task-4", expected_attempts=0, new_status="failed")

    assert captured["returning"] is True
    assert "RETURNING id" in captured["sql"]
