"""[P0-5] TZ drift sync pre-pickup en process_plan_chunk_queue.

Antes del fix:
    El TZ sync vivía en dos sitios:
      (a) Cron dedicado `_sync_chunk_queue_tz_offsets`, cada 15 min default.
      (b) Gate `_check_chunk_learning_ready` AFTER pickup (con advisory lock).

    Ambos dejaban una ventana TOCTOU:
      - Cron 15min: el usuario cambia TZ a las 12:00, no hay sync hasta 12:15;
        chunks disparan con TZ vieja durante 15 min.
      - Gate AFTER pickup: el chunk ya está status='processing' con el snapshot
        cargado en memoria del worker. El gate update-ea la DB para OTROS chunks
        pero el actual sigue con su snapshot stale.

Después del fix:
    `process_plan_chunk_queue` (cada 1 min) llama `_sync_chunk_queue_tz_offsets()`
    ANTES de la SELECT FOR UPDATE de pickup. Cualquier chunk con drift se
    actualiza pre-pickup; el siguiente SELECT lee el snapshot ya fresco.

Tests verifican:
  - Orden de llamadas: TZ sync ANTES de pickup query.
  - Si TZ sync falla, pickup sigue (best-effort).
  - El cron dedicado sigue activo (red de seguridad).
"""
from unittest.mock import patch, MagicMock, call

import pytest


def test_pre_pickup_tz_sync_called_before_pickup_query():
    """Contrato de orden: dentro de process_plan_chunk_queue, la llamada a
    _sync_chunk_queue_tz_offsets DEBE ocurrir ANTES de la SELECT FOR UPDATE
    de pickup. Si se invierte el orden, los chunks pickeados leerían TZ
    stale antes de ser sincronizados.
    """
    import cron_tasks

    call_order = []

    def _record_tz_sync(target_user_id=None):
        call_order.append("tz_sync")
        return 0

    def _record_pickup(query, params=None, fetch_all=None, returning=None, **kw):
        if query and "FOR UPDATE SKIP LOCKED" in query and "plan_chunk_queue" in query:
            call_order.append("pickup")
        return []

    with patch.object(cron_tasks, "_sync_chunk_queue_tz_offsets", side_effect=_record_tz_sync), \
         patch.object(cron_tasks, "execute_sql_write", side_effect=_record_pickup), \
         patch.object(cron_tasks, "execute_sql_query", return_value=None), \
         patch.object(cron_tasks, "_process_pending_shopping_lists", return_value=None), \
         patch.object(cron_tasks, "_recover_pantry_paused_chunks", return_value=None):
        cron_tasks.process_plan_chunk_queue()

    assert "tz_sync" in call_order, (
        "process_plan_chunk_queue no invocó _sync_chunk_queue_tz_offsets. "
        "Sin esto, chunks con TZ stale serían pickeados sin resync previo."
    )
    assert "pickup" in call_order, (
        "process_plan_chunk_queue no llegó a la pickup query (¿abortó antes?). "
        "El test no puede verificar el orden si el pickup no corrió."
    )
    tz_idx = call_order.index("tz_sync")
    pickup_idx = call_order.index("pickup")
    assert tz_idx < pickup_idx, (
        f"Orden INVERTIDO: tz_sync@{tz_idx}, pickup@{pickup_idx}. "
        "El TZ sync DEBE ejecutarse ANTES del pickup para que el SELECT FOR "
        "UPDATE lea snapshots ya frescos. Si pickup corre primero, los chunks "
        "promocionados a 'processing' arrastran TZ stale en memoria."
    )


def test_pre_pickup_tz_sync_failure_does_not_block_pickup():
    """Best-effort: si _sync_chunk_queue_tz_offsets lanza excepción (DB blip,
    timeout, etc.), el worker NO debe abortar. El cron dedicado de 15 min y
    el gate post-pickup son redes de seguridad.
    """
    import cron_tasks

    call_order = []

    def _failing_tz_sync(target_user_id=None):
        call_order.append("tz_sync_failed")
        raise RuntimeError("simulated DB blip")

    def _record_pickup(query, params=None, fetch_all=None, returning=None, **kw):
        if query and "FOR UPDATE SKIP LOCKED" in query and "plan_chunk_queue" in query:
            call_order.append("pickup")
        return []

    with patch.object(cron_tasks, "_sync_chunk_queue_tz_offsets", side_effect=_failing_tz_sync), \
         patch.object(cron_tasks, "execute_sql_write", side_effect=_record_pickup), \
         patch.object(cron_tasks, "execute_sql_query", return_value=None), \
         patch.object(cron_tasks, "_process_pending_shopping_lists", return_value=None), \
         patch.object(cron_tasks, "_recover_pantry_paused_chunks", return_value=None):
        # No debe lanzar.
        cron_tasks.process_plan_chunk_queue()

    assert "tz_sync_failed" in call_order
    assert "pickup" in call_order, (
        "Cuando el pre-pickup TZ sync falla, el worker debe continuar al "
        "pickup (best-effort). El cron dedicado de 15min cubre el gap."
    )


def test_pre_pickup_tz_sync_runs_after_zombie_rescue():
    """El zombie rescue mueve chunks 'processing' (con heartbeat stale) a
    'pending'. Esos chunks recién resucitados deben pasar por TZ sync ANTES
    de ser pickeados de nuevo. Por eso el orden correcto es:
        zombie rescue → TZ sync → pickup.
    """
    import cron_tasks

    call_order = []

    def _record_sql_write(query, params=None, fetch_all=None, returning=None, **kw):
        # [test fix · P2-CHUNK-1] El zombie-rescue movió la ventana de 10min de
        # literal `INTERVAL '10 minutes'` a knob CHUNK_ZOMBIE_RESCUE_MINUTES via
        # `make_interval(mins => %s)`. Identificamos la query por sus marcadores
        # estables: UPDATE a plan_chunk_queue, filtra processing por updated_at
        # stale (make_interval) y excluye dead-lettered + lock con heartbeat fresco.
        if (
            query
            and "UPDATE plan_chunk_queue" in query
            and "status = 'processing'" in query
            and "updated_at < NOW() - make_interval(mins => %s)" in query
            and "dead_lettered_at IS NULL" in query
        ):
            call_order.append("zombie_rescue")
        elif query and "FOR UPDATE SKIP LOCKED" in query and "plan_chunk_queue" in query:
            call_order.append("pickup")
        return []

    def _record_tz_sync(target_user_id=None):
        call_order.append("tz_sync")
        return 0

    with patch.object(cron_tasks, "_sync_chunk_queue_tz_offsets", side_effect=_record_tz_sync), \
         patch.object(cron_tasks, "execute_sql_write", side_effect=_record_sql_write), \
         patch.object(cron_tasks, "execute_sql_query", return_value=None), \
         patch.object(cron_tasks, "_process_pending_shopping_lists", return_value=None), \
         patch.object(cron_tasks, "_recover_pantry_paused_chunks", return_value=None):
        cron_tasks.process_plan_chunk_queue()

    # Verificar: zombie_rescue → tz_sync → pickup.
    assert "zombie_rescue" in call_order, "Zombie rescue no corrió (¿se movió de orden?)."
    assert "tz_sync" in call_order
    assert "pickup" in call_order

    z = call_order.index("zombie_rescue")
    t = call_order.index("tz_sync")
    p = call_order.index("pickup")
    assert z < t < p, (
        f"Orden incorrecto: zombie@{z}, tz_sync@{t}, pickup@{p}. "
        "Debe ser zombie → tz_sync → pickup para que chunks resucitados "
        "tengan TZ fresca antes de ser pickeados."
    )


def test_dedicated_tz_sync_cron_remains_registered():
    """Defense-in-depth: el cron dedicado `sync_chunk_queue_tz_offsets`
    sigue registrado en el scheduler (cada 15 min). Con el pre-pickup
    sync corriendo cada 1 min, el dedicated es redundante en el caso normal,
    pero crítico cuando process_plan_chunk_queue está pausado (apscheduler
    issue, deploy en marcha, debug).
    """
    import cron_tasks
    import inspect
    src = inspect.getsource(cron_tasks.register_plan_chunk_scheduler)
    assert 'sync_chunk_queue_tz_offsets' in src, (
        "El cron dedicado de TZ sync fue removido del scheduler. Sin esto, "
        "si process_plan_chunk_queue se pausa, no hay TZ sync de respaldo."
    )
    assert '_sync_chunk_queue_tz_offsets' in src
    assert 'CHUNK_TZ_SYNC_INTERVAL_MINUTES' in src


def test_in_gate_tz_resync_remains_active():
    """Defense-in-depth: el TZ resync DENTRO de _check_chunk_learning_ready
    (con advisory lock purpose='tz_resync') sigue activo. Es la última red
    de seguridad para chunks que YA están processing — el pre-pickup sync
    no los toca (sólo afecta pending/stale).
    """
    import cron_tasks
    import inspect
    src = inspect.getsource(cron_tasks._check_chunk_learning_ready)
    assert 'CHUNK_TZ_DRIFT_THRESHOLD_MINUTES' in src, (
        "El threshold de drift fue removido del gate. Sin esto, chunks ya "
        "processing no detectarían drift que ocurrió post-pickup."
    )
    assert 'tz_resync' in src, (
        "El advisory lock purpose='tz_resync' fue removido del gate."
    )
