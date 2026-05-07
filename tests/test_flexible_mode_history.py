"""
[P1-3] Tests para `_activate_flexible_mode` con persistencia en _mode_history.

Antes el helper solo modificaba el snapshot in-memory + log estructurado. Ahora,
si recibe `meal_plan_id`, persiste un evento en `meal_plans.plan_data._mode_history`
para que el frontend pueda mostrar el badge "Plan en modo flexible — verifica tu nevera"
en vez de aceptar el plan degradado silenciosamente.

Valida:
  A. Con meal_plan_id, persiste evento estructurado en _mode_history y setea
     _current_mode/_current_mode_reason/_current_mode_advisory_only.
  B. Sin meal_plan_id, NO intenta persistir (mantiene comportamiento legacy).
  C. Si la persistencia falla, el snapshot in-memory aún se devuelve (no
     bloquea la generación del chunk).
  D. Eventos previos en _mode_history se preservan (append, no overwrite).
  E. Cap de 100 eventos: el helper conserva los últimos 100 si la lista crece.
"""
import sys
from unittest.mock import patch, MagicMock

sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())


def _base_snapshot():
    return {"form_data": {"foo": "bar"}}


def test_a_persists_mode_event_when_meal_plan_id_provided():
    """[P1-3] Con meal_plan_id, el evento queda en _mode_history y se setea _current_mode."""
    captured_mutators = []

    def fake_atomic(plan_id, mutator, lock_timeout_ms=None):
        # Simular el contrato real del helper: pasar plan_data inicial vacío
        pd = {}
        result = mutator(pd)
        captured_mutators.append((plan_id, result if result is not None else pd))
        return result if result is not None else pd

    with patch("db_plans.update_plan_data_atomic", side_effect=fake_atomic):
        from cron_tasks import _activate_flexible_mode
        snap = _activate_flexible_mode(
            _base_snapshot(),
            reason="stale_snapshot_force_flex",
            user_id="user-1",
            week_num=3,
            meal_plan_id="plan-xyz",
            advisory_only=True,
        )

    # In-memory: comportamiento legacy preservado
    assert snap["_pantry_flexible_mode"] is True
    assert snap["_pantry_advisory_only"] is True
    assert snap["_pantry_pause_resolution"] == "stale_snapshot_force_flex"

    # Persistencia: 1 llamada al helper atómico
    assert len(captured_mutators) == 1
    plan_id, mutated = captured_mutators[0]
    assert plan_id == "plan-xyz"
    assert mutated["_current_mode"] == "flexible"
    assert mutated["_current_mode_reason"] == "stale_snapshot_force_flex"
    assert mutated["_current_mode_advisory_only"] is True
    assert isinstance(mutated["_mode_history"], list)
    assert len(mutated["_mode_history"]) == 1
    event = mutated["_mode_history"][0]
    assert event["mode"] == "flexible"
    assert event["reason"] == "stale_snapshot_force_flex"
    assert event["advisory_only"] is True
    assert event["week_number"] == 3
    assert "ts" in event


def test_b_no_meal_plan_id_skips_persistence():
    """[P1-3] Sin meal_plan_id (callsites legacy), NO se intenta persistir."""
    with patch("db_plans.update_plan_data_atomic") as mock_atomic:
        from cron_tasks import _activate_flexible_mode
        snap = _activate_flexible_mode(
            _base_snapshot(),
            reason="degraded_flexible_meal",
            user_id="user-1",
            week_num=2,
        )

    assert snap["_pantry_flexible_mode"] is True
    assert mock_atomic.call_count == 0


def test_c_persist_failure_does_not_block_chunk_generation():
    """[P1-3] Si update_plan_data_atomic explota, el snapshot in-memory se devuelve
    igual — el chunk debe seguir generándose; lo que se pierde es el badge del frontend."""
    def fake_atomic(plan_id, mutator, lock_timeout_ms=None):
        raise RuntimeError("simulated DB lock timeout")

    with patch("db_plans.update_plan_data_atomic", side_effect=fake_atomic):
        from cron_tasks import _activate_flexible_mode
        # No debe propagar la excepción.
        snap = _activate_flexible_mode(
            _base_snapshot(),
            reason="zero_log_force_flex",
            user_id="user-1",
            week_num=2,
            meal_plan_id="plan-xyz",
            learning_flexible=True,
        )

    assert snap["_pantry_flexible_mode"] is True
    assert snap["_learning_flexible_mode"] is True


def test_d_appends_to_existing_mode_history():
    """[P1-3] Eventos previos en _mode_history se preservan (append, no overwrite)."""
    initial_history = [
        {"ts": "2026-04-01T00:00:00+00:00", "mode": "flexible", "reason": "old_event", "week_number": 1},
    ]

    def fake_atomic(plan_id, mutator, lock_timeout_ms=None):
        pd = {"_mode_history": list(initial_history)}
        return mutator(pd)

    captured = {}
    def capture_atomic(plan_id, mutator, lock_timeout_ms=None):
        result = fake_atomic(plan_id, mutator, lock_timeout_ms)
        captured["result"] = result
        return result

    with patch("db_plans.update_plan_data_atomic", side_effect=capture_atomic):
        from cron_tasks import _activate_flexible_mode
        _activate_flexible_mode(
            _base_snapshot(),
            reason="zero_log_force_flex",
            week_num=4,
            meal_plan_id="plan-xyz",
        )

    history = captured["result"]["_mode_history"]
    assert len(history) == 2, "Debe preservar el evento previo y añadir el nuevo"
    assert history[0]["reason"] == "old_event"  # preservado
    assert history[1]["reason"] == "zero_log_force_flex"  # nuevo
    assert history[1]["week_number"] == 4


def test_e_caps_history_at_100_events():
    """[P1-3] Si _mode_history crece sin control (bug en algún path que loopea),
    el helper conserva los últimos 100 para evitar bloat infinito de plan_data."""
    # 105 eventos previos: tras añadir uno nuevo, total=106, debe quedarse en 100.
    initial_history = [
        {"ts": f"2026-04-01T00:{i:02d}:00+00:00", "mode": "flexible", "reason": "old", "week_number": 1}
        for i in range(105)
    ]

    captured = {}
    def fake_atomic(plan_id, mutator, lock_timeout_ms=None):
        pd = {"_mode_history": list(initial_history)}
        result = mutator(pd)
        captured["result"] = result
        return result

    with patch("db_plans.update_plan_data_atomic", side_effect=fake_atomic):
        from cron_tasks import _activate_flexible_mode
        _activate_flexible_mode(
            _base_snapshot(),
            reason="latest_event",
            week_num=5,
            meal_plan_id="plan-xyz",
        )

    history = captured["result"]["_mode_history"]
    assert len(history) == 100, f"Cap debe ser 100, got {len(history)}"
    # El evento nuevo debe ser el último.
    assert history[-1]["reason"] == "latest_event"
    # Los 5 más viejos deben haberse descartado (LRU).
    # Quedan eventos viejos cuyo minuto va de :05 a :104 (los primeros 5 con minuto 0-4
    # se descartaron por la operación [-100:]).
    # Verificamos que el primero ya NO sea el del minuto 0.
    assert history[0]["ts"] != "2026-04-01T00:00:00+00:00", \
        "Los 5 eventos más viejos deben haberse descartado"
