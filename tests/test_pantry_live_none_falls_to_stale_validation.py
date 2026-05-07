"""
[P0-3] Test del gap silencioso donde `get_user_inventory_net` retornaba None
sin excepción y el código aceptaba el snapshot fallback marcándolo como `live`,
sin pasar por la re-validación de age + retry.

Ahora None se trata igual que excepción: se cae al flujo de fallback con
re-validación de snapshot age. Si el snapshot ya supera el TTL, dispara el live
retry adicional o pausa/escala según corresponda.
"""
import sys
from unittest.mock import patch, MagicMock

sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())


def _build_form_data():
    return {}


def _build_snapshot(captured_hours_ago: float | None = None):
    from datetime import datetime, timezone, timedelta
    snap = {"current_pantry_ingredients": [{"name": "Arroz", "quantity": 1000, "unit": "g"}]}
    if captured_hours_ago is not None:
        snap["_pantry_captured_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=captured_hours_ago)
        ).isoformat()
    return snap


def test_a_pantry_live_none_no_longer_silently_uses_snapshot():
    """[P0-3] Antes: pantry_live=None → snapshot silencioso con source='live'.
    Ahora: None se trata como falla → re-validación de age.
    """
    snap_form = _build_snapshot(captured_hours_ago=10)  # > TTL=6h

    # Mock chain: get_user_inventory_net retorna None la primera vez (live falla),
    # también la segunda (live retry tras detectar stale), y el backoff falla.
    fetch_calls = {"n": 0}
    def fake_get_inv(uid):
        fetch_calls["n"] += 1
        return None  # siempre None, sin excepción

    with patch("cron_tasks.get_user_inventory_net", side_effect=fake_get_inv), \
         patch("cron_tasks._record_inventory_live_failure", return_value=False), \
         patch("cron_tasks._record_inventory_live_success"), \
         patch("cron_tasks._get_user_tz_live", return_value=0), \
         patch("cron_tasks._fetch_inventory_with_backoff", return_value=(None, [], "stub")), \
         patch("cron_tasks._pause_chunk_for_stale_inventory") as mock_pause:
        # [P0-5] Renamed from `_refresh_pantry_and_get_inventory` to `_refresh_chunk_pantry`
        # in cron_tasks.py:1928. The signature is identical.
        from cron_tasks import _refresh_chunk_pantry
        form_data = _build_form_data()
        result = _refresh_chunk_pantry(
            user_id="user-1",
            form_data=form_data,
            snapshot_form_data=snap_form,
            task_id="task-1",
            week_number=2,
        )

    # El source NUNCA debe ser "live" si pantry_live retornó None.
    assert result.get("_fresh_pantry_source") != "live", (
        f"source='live' es mentira si pantry_live=None. Got source={result.get('_fresh_pantry_source')}"
    )
    # Debe haber llamado al least dos veces a get_user_inventory_net (live inicial + retry tras detectar stale).
    assert fetch_calls["n"] >= 2, "Debe re-intentar live cuando snapshot está stale"


def test_b_pantry_live_ok_preserves_live_source():
    """[P0-3] Cuando live retorna inventario válido, source debe ser 'live' y telemetría
    de snapshot age debe estar presente."""
    snap_form = _build_snapshot(captured_hours_ago=2.5)

    def fake_get_inv(uid):
        return [{"name": "Pollo", "quantity": 500, "unit": "g"}]

    with patch("cron_tasks.get_user_inventory_net", side_effect=fake_get_inv), \
         patch("cron_tasks._record_inventory_live_success") as mock_succ, \
         patch("cron_tasks._get_user_tz_live", return_value=0):
        # [P0-5] Renamed from `_refresh_pantry_and_get_inventory` to `_refresh_chunk_pantry`
        # in cron_tasks.py:1928. The signature is identical.
        from cron_tasks import _refresh_chunk_pantry
        form_data = _build_form_data()
        result = _refresh_chunk_pantry(
            user_id="user-2",
            form_data=form_data,
            snapshot_form_data=snap_form,
            task_id="task-2",
            week_number=1,
        )

    assert result.get("_fresh_pantry_source") == "live"
    assert mock_succ.call_count == 1
    # [P0-3/TELEMETRY] snapshot_age_hours debe estar set aún con live OK.
    age = result.get("_pantry_snapshot_age_hours")
    assert age is not None, "Debe registrar snapshot_age aunque live OK (telemetría)"
    assert 2.0 <= age <= 3.5, f"Age debe estar cerca de 2.5h, got {age}"
