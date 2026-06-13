"""[P2-3] Tests para `_cleanup_orphan_chunks` como cron job dedicado.

Antes la lógica corría dentro de `process_plan_chunk_queue` (cada
CHUNK_SCHEDULER_INTERVAL_MINUTES = 1 min). Si la query de orphan-detection
era lenta o fallaba, bloqueaba el hot path del worker. Ahora es:

  - Función `_cleanup_orphan_chunks()` aislada.
  - Cron job dedicado `cleanup_orphan_chunks` con frecuencia configurable
    `CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES` (default 5 min).
  - El processor (`process_plan_chunk_queue`) ya NO ejecuta esta limpieza
    inline en cada tick.

Ejecutar:
    cd backend && python -m pytest tests/test_p2_3_orphan_cleanup_independent_cron.py -v
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ---------------------------------------------------------------------------
# 1. Función identifica chunks huérfanos y los cancela
# ---------------------------------------------------------------------------
# [P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado al CTE atómico de prod
# (P2-RACE-FIX · 2026-05-26): `_cleanup_orphan_chunks` ya NO hace SELECT +
# UPDATE en 2 pasos. Hace UN solo `execute_sql_write(CTE, returning=True)`
# que detecta + cancela los huérfanos en un statement atómico (FOR UPDATE
# SKIP LOCKED) y devuelve las filas canceladas vía RETURNING. Los orphans
# ahora vienen del RETURN del write, no de un execute_sql_query separado.
@patch("cron_tasks.release_chunk_reservations")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_cancels_chunks_whose_meal_plan_no_longer_exists(
    mock_query, mock_write, mock_release
):
    from cron_tasks import _cleanup_orphan_chunks

    # El CTE con RETURNING devuelve las filas canceladas (id, user_id).
    mock_write.return_value = [
        {"id": "chunk-1", "user_id": "user-A"},
        {"id": "chunk-2", "user_id": "user-B"},
    ]

    cancelled = _cleanup_orphan_chunks()

    assert cancelled == 2

    # Liberación de reservas: una por chunk huérfano.
    assert mock_release.call_count == 2
    mock_release.assert_any_call("user-A", "chunk-1")
    mock_release.assert_any_call("user-B", "chunk-2")

    # El statement de detección+cancelación atómico se ejecutó: un único
    # `execute_sql_write` con el CTE `UPDATE plan_chunk_queue SET status =
    # 'cancelled'` + RETURNING.
    update_calls = [
        c for c in mock_write.call_args_list
        if "UPDATE plan_chunk_queue" in c.args[0] and "cancelled" in c.args[0]
    ]
    assert len(update_calls) == 1
    # Debe usar el snapshot atómico FOR UPDATE SKIP LOCKED + RETURNING
    # (sin esto, reaparece el race "INSERT meal_plan entre SELECT y UPDATE").
    assert "FOR UPDATE SKIP LOCKED" in update_calls[0].args[0]
    assert "RETURNING" in update_calls[0].args[0]
    # Y NO debe quedar un SELECT separado por execute_sql_query (2-step legacy).
    mock_query.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Sin huérfanos: no UPDATE, no release
# ---------------------------------------------------------------------------
@patch("cron_tasks.release_chunk_reservations")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_no_orphans_no_update_no_release(mock_query, mock_write, mock_release):
    from cron_tasks import _cleanup_orphan_chunks

    # CTE no encontró huérfanos → RETURNING vacío.
    mock_write.return_value = []

    cancelled = _cleanup_orphan_chunks()

    assert cancelled == 0
    # Sin huérfanos, ninguna reserva se libera (la propiedad clave: no se
    # toca nada cuando no hay nada que cancelar).
    mock_release.assert_not_called()
    # [P1-NEON-DB-MIGRATION · 2026-06-12] El statement de detección atómico
    # SÍ se ejecuta (es el CTE que mira si hay huérfanos), pero como RETURNING
    # viene vacío, cancela 0 filas → efecto neto nulo. Ya NO existe el patrón
    # 2-step donde "no orphans" implicaba "no UPDATE" — ahora detección y
    # cancelación son el mismo statement. No se hace un SELECT separado.
    mock_query.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Filtro: SELECT solo trae estados "vivos"
# ---------------------------------------------------------------------------
@patch("cron_tasks.release_chunk_reservations")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_select_only_includes_live_statuses(mock_query, mock_write, mock_release):
    from cron_tasks import _cleanup_orphan_chunks

    mock_write.return_value = []
    _cleanup_orphan_chunks()

    # [P1-NEON-DB-MIGRATION · 2026-06-12] El filtro de estados vive ahora en el
    # CTE del `execute_sql_write` (el SELECT del CTE), no en un execute_sql_query
    # separado. Parseamos el SQL real del write.
    sql = mock_write.call_args.args[0]
    # Estados vivos que pueden ser huérfanos.
    assert "'pending'" in sql
    assert "'stale'" in sql
    assert "'processing'" in sql
    # NO incluye estados terminales ya cerrados como candidatos a huérfano.
    assert "'completed'" not in sql
    # 'cancelled' solo aparece en el `SET status = 'cancelled'` del UPDATE,
    # nunca como estado de entrada candidato a cancelar.
    assert "'cancelled'" not in sql or "SET status = 'cancelled'" in sql


# ---------------------------------------------------------------------------
# 4. Resilencia: error en release individual NO interrumpe el batch
# ---------------------------------------------------------------------------
@patch("cron_tasks.release_chunk_reservations")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_release_error_does_not_abort_batch(mock_query, mock_write, mock_release):
    from cron_tasks import _cleanup_orphan_chunks

    # El CTE ya canceló ambos chunks atómicamente (RETURNING los 2).
    mock_write.return_value = [
        {"id": "chunk-1", "user_id": "user-A"},
        {"id": "chunk-2", "user_id": "user-B"},
    ]
    # release del primer chunk falla; el segundo debe procesarse igualmente.
    # El cancel ya ocurrió en el CTE (pre-release), así que el conteo no se ve
    # afectado por un fallo de release best-effort.
    mock_release.side_effect = [RuntimeError("boom"), None]

    cancelled = _cleanup_orphan_chunks()

    assert cancelled == 2
    # Ambos chunks intentan liberar reservas pese al fallo del primero.
    assert mock_release.call_count == 2
    # El statement atómico de cancelación corrió una sola vez.
    update_calls = [
        c for c in mock_write.call_args_list
        if "UPDATE plan_chunk_queue" in c.args[0]
    ]
    assert len(update_calls) == 1, "El CTE de cancellation corre aunque release falle"


# ---------------------------------------------------------------------------
# 5. Resilencia: excepción en query principal retorna 0, no propaga
# ---------------------------------------------------------------------------
@patch("cron_tasks.release_chunk_reservations")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_query_exception_returns_zero_does_not_raise(
    mock_query, mock_write, mock_release
):
    from cron_tasks import _cleanup_orphan_chunks

    # [P1-NEON-DB-MIGRATION · 2026-06-12] La query de detección+cancelación es
    # ahora el `execute_sql_write` del CTE; un fallo de conexión sale de ahí.
    mock_write.side_effect = RuntimeError("connection refused")

    # No debe propagar excepción (es job de mantenimiento).
    cancelled = _cleanup_orphan_chunks()

    assert cancelled == 0
    # Si el statement atómico falló, no hay filas que liberar.
    mock_release.assert_not_called()


# ---------------------------------------------------------------------------
# 6. process_plan_chunk_queue ya NO ejecuta el cleanup inline
# ---------------------------------------------------------------------------
def test_process_plan_chunk_queue_does_not_run_orphan_cleanup_inline():
    """Lectura del archivo: el bloque `[GAP 3 FIX: Cleanup chunks huérfanos]`
    inline en `process_plan_chunk_queue` ya no debe existir; el cleanup vive
    en el cron dedicado `_cleanup_orphan_chunks`."""
    cron_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cron_tasks.py",
    )
    with open(cron_path, encoding="utf-8") as f:
        content = f.read()

    # El header inline antiguo no debe estar presente como bloque ejecutable.
    assert "[GAP 3 FIX: Cleanup chunks huérfanos]" not in content, (
        "El bloque inline antiguo debe eliminarse — el cleanup vive en cron dedicado."
    )
    # El comentario de migración SÍ debe estar.
    assert "_cleanup_orphan_chunks" in content


# ---------------------------------------------------------------------------
# 7. Cron job se registra con la frecuencia configurada
# ---------------------------------------------------------------------------
def test_cron_job_registered_with_configured_interval():
    """register_plan_chunk_scheduler debe registrar `cleanup_orphan_chunks` con
    interval = CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES."""
    from cron_tasks import register_plan_chunk_scheduler, _cleanup_orphan_chunks
    from constants import CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES

    scheduler = MagicMock()
    scheduler.get_job.return_value = None  # ningún job preexistente

    register_plan_chunk_scheduler(scheduler)

    # Buscar la llamada a add_job con id='cleanup_orphan_chunks'.
    cleanup_calls = [
        c for c in scheduler.add_job.call_args_list
        if c.kwargs.get("id") == "cleanup_orphan_chunks"
    ]
    assert len(cleanup_calls) == 1, (
        "Cron job 'cleanup_orphan_chunks' no fue registrado en register_plan_chunk_scheduler."
    )
    call = cleanup_calls[0]
    assert call.args[0] is _cleanup_orphan_chunks
    assert call.args[1] == "interval"
    assert call.kwargs["minutes"] == CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES
    assert call.kwargs["max_instances"] == 1
    assert call.kwargs["coalesce"] is True


def test_cron_job_skipped_if_already_registered():
    """Idempotencia: si `cleanup_orphan_chunks` ya existe, no se re-registra."""
    from cron_tasks import register_plan_chunk_scheduler

    scheduler = MagicMock()
    # Simular que cleanup_orphan_chunks ya existe pero los demás no.
    def get_job_side_effect(job_id):
        return MagicMock() if job_id == "cleanup_orphan_chunks" else None
    scheduler.get_job.side_effect = get_job_side_effect

    register_plan_chunk_scheduler(scheduler)

    cleanup_calls = [
        c for c in scheduler.add_job.call_args_list
        if c.kwargs.get("id") == "cleanup_orphan_chunks"
    ]
    assert cleanup_calls == [], (
        "Si el job ya existe, register_plan_chunk_scheduler NO debe re-registrarlo."
    )


# ---------------------------------------------------------------------------
# 8. Constante respeta env override y mínimo
# ---------------------------------------------------------------------------
def test_orphan_cleanup_interval_env_override():
    import importlib

    with patch.dict(os.environ, {"CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES": "10"}, clear=False):
        import constants as _c
        importlib.reload(_c)
        assert _c.CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES == 10

    # Restaurar default para no contaminar otros tests.
    with patch.dict(os.environ, {"CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES": "5"}, clear=False):
        import constants as _c
        importlib.reload(_c)


def test_orphan_cleanup_interval_minimum_is_one():
    """`max(1, ...)` previene intervalos 0 que dispararían loops infinitos."""
    import importlib

    with patch.dict(os.environ, {"CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES": "0"}, clear=False):
        import constants as _c
        importlib.reload(_c)
        assert _c.CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES == 1

    with patch.dict(os.environ, {"CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES": "5"}, clear=False):
        import constants as _c
        importlib.reload(_c)
