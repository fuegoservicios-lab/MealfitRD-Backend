"""[P0-DASH-CHIP-HONESTY · 2026-05-09] Tests para el chip honesto del Dashboard.

Tooltip-anchor: `P0-DASH-CHIP-HONESTY-START`, `P0-DASH-CHIP-HONESTY-FE`,
`P0-DASH-CHIP-HONESTY-PAUSE` (cron_tasks.py). El test verifica que
los anchors están vivos para drift detection (renombrar el bloque rompe el test).

Bug original (reportado en producción 2026-05-09):
    Plan con first_chunk completed (Sáb-Dom) + 2 rolling_refill chunks
    en pending_user_action (Lun-Jue). El primero pausado por el worker
    al pickup (cron_tasks.py:17669) cuando detectó pantry insuficiente,
    PERO `_pause_chunk_for_pantry_refresh` recibía `reason=None` (default
    legacy) y la rama `else` omitía persistir `_pantry_pause_reason` en
    el snapshot.

    Resultado:
      - `/blocked_reasons` (plans.py:3713-3728) caía a `_unknown` →
        usuario veía "Bloqueo sin clasificar" sin CTA.
      - Dashboard.jsx:2712-2774 derivaba el chip leyendo solo
        `plan_data.generation_status` ('generating_next') +
        `_user_action_required` (NULL) → pintaba "Lunes - en camino"
        con shimmer + spinner mientras la queue tenía chunks pausados.
        El usuario no se enteraba de que tenía que actualizar la nevera.

3 fixes encadenados:
  A. /chunk-status devuelve `paused_chunks` con reason resuelto +
     counters (in_flight_count, pending_user_action_count, etc.).
  B. Dashboard.jsx 3 ramas honestas (en camino / pausado / acción).
  C. _pause_chunk_for_pantry_refresh siempre persiste reason
     (default `"empty_pantry"` matches reason_to_text canónico).

Este archivo cubre el fix C (backend) + el contrato del endpoint /chunk-status.
"""
import copy
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# Fix C · _pause_chunk_for_pantry_refresh siempre escribe reason
# ============================================================

@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_pause_chunk_persists_reason_when_caller_omits_it(
    mock_query, mock_write, _mock_push
):
    """Caller invoca sin reason → snapshot persiste reason='empty_pantry'.

    Reproduce el call site cron_tasks.py:17669 que llamaba sin reason.
    Antes: snapshot terminaba con _pantry_pause_started_at, reminders, ttl,
    reminder_hours pero SIN _pantry_pause_reason → /blocked_reasons caía
    a `_unknown`. Ahora el default 'empty_pantry' persiste siempre.
    """
    import cron_tasks

    # Snapshot pre-pausa: vacío (chunk recién creado).
    mock_query.return_value = {"pipeline_snapshot": {}}
    captured: dict = {}

    def _capture(query, params):
        # El UPDATE recibe (snapshot_json, task_id) en ese orden.
        if "UPDATE plan_chunk_queue" in query and "pending_user_action" in query:
            captured["snapshot"] = json.loads(params[0])

    mock_write.side_effect = _capture

    cron_tasks._pause_chunk_for_pantry_refresh(
        task_id="chunk-no-reason",
        user_id="u-1",
        week_number=1,
        fresh_inventory=[],
        # NO pasamos reason — reproduce el caller del bug.
    )

    assert "snapshot" in captured, "El UPDATE debe ejecutarse"
    snap = captured["snapshot"]
    assert snap.get("_pantry_pause_reason") == "empty_pantry", (
        "Reason debe persistirse aunque el caller no lo pase. "
        f"snapshot={snap}"
    )
    assert "_pantry_pause_started_at" in snap
    assert "_pantry_pause_ttl_hours" in snap


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_pause_chunk_persists_explicit_reason(mock_query, mock_write, _mock_push):
    """Caller invoca con reason → ese reason se persiste tal cual."""
    import cron_tasks

    mock_query.return_value = {"pipeline_snapshot": {}}
    captured: dict = {}

    def _capture(query, params):
        if "UPDATE plan_chunk_queue" in query and "pending_user_action" in query:
            captured["snapshot"] = json.loads(params[0])

    mock_write.side_effect = _capture

    cron_tasks._pause_chunk_for_pantry_refresh(
        task_id="chunk-explicit",
        user_id="u-1",
        week_number=2,
        fresh_inventory=[],
        reason="empty_pantry_proactive",
    )

    snap = captured["snapshot"]
    assert snap.get("_pantry_pause_reason") == "empty_pantry_proactive"


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_pause_chunk_persistent_drift_uses_longer_ttl(mock_query, mock_write, _mock_push):
    """`reason='persistent_drift'` activa CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS
    en lugar de CHUNK_PANTRY_EMPTY_TTL_HOURS — diferenciado del path empty_pantry."""
    import cron_tasks
    from constants import (
        CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS,
        CHUNK_PANTRY_EMPTY_TTL_HOURS,
    )

    mock_query.return_value = {"pipeline_snapshot": {}}
    captured: dict = {}

    def _capture(query, params):
        if "UPDATE plan_chunk_queue" in query and "pending_user_action" in query:
            captured["snapshot"] = json.loads(params[0])

    mock_write.side_effect = _capture

    cron_tasks._pause_chunk_for_pantry_refresh(
        task_id="chunk-drift",
        user_id="u-1",
        week_number=2,
        fresh_inventory=[],
        reason="persistent_drift",
    )

    snap = captured["snapshot"]
    assert snap.get("_pantry_pause_reason") == "persistent_drift"
    assert snap.get("_pantry_pause_ttl_hours") == CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS
    assert CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS != CHUNK_PANTRY_EMPTY_TTL_HOURS, (
        "Las constantes deben diferir; si convergen, el test pierde poder discriminativo."
    )


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_pause_chunk_preserves_existing_started_at(mock_query, mock_write, _mock_push):
    """Si el snapshot ya tenía _pantry_pause_started_at (re-pausa), NO se
    sobrescribe — preserva la marca de inicio del primer pause."""
    import cron_tasks

    original_started = "2026-05-08T12:00:00+00:00"
    mock_query.return_value = {
        "pipeline_snapshot": {"_pantry_pause_started_at": original_started}
    }
    captured: dict = {}

    def _capture(query, params):
        if "UPDATE plan_chunk_queue" in query:
            captured["snapshot"] = json.loads(params[0])

    mock_write.side_effect = _capture

    cron_tasks._pause_chunk_for_pantry_refresh(
        task_id="chunk-repaused",
        user_id="u-1",
        week_number=1,
        fresh_inventory=[],
    )

    snap = captured["snapshot"]
    assert snap.get("_pantry_pause_started_at") == original_started, (
        "La marca original debe preservarse en re-pausas."
    )


# ============================================================
# Drift detection · tooltip-anchors vivos
# ============================================================

def test_pause_chunk_anchor_is_live():
    """El bloque de comentario con tooltip-anchor `P0-DASH-CHIP-HONESTY-PAUSE`
    debe seguir presente en cron_tasks._pause_chunk_for_pantry_refresh —
    si alguien renombra/elimina el anchor sin actualizar este test, falla.

    Cierra el patrón establecido en CLAUDE.md: cuando un test parsea source
    de prod con regex, incluir tooltip-anchor en el código fuente para que
    un futuro renombre falle el test antes de cambiar producción.
    """
    import cron_tasks
    import inspect

    src = inspect.getsource(cron_tasks._pause_chunk_for_pantry_refresh)
    assert "P0-DASH-CHIP-HONESTY-PAUSE" in src, (
        "Tooltip-anchor missing en _pause_chunk_for_pantry_refresh — "
        "renombre detectado o anchor borrado. Si este fix se reformula, "
        "actualizar también el anchor + este test."
    )


def test_chunk_status_endpoint_anchor_is_live():
    """Mismo guard para el bloque del endpoint /chunk-status (plans.py)."""
    from routers import plans
    import inspect

    src = inspect.getsource(plans.api_chunk_status)
    assert "P0-DASH-CHIP-HONESTY-START" in src, (
        "Tooltip-anchor missing en api_chunk_status — el bloque que devuelve "
        "paused_chunks + counters debe seguir presente."
    )


# ============================================================
# Contrato del endpoint /chunk-status: paused_chunks shape
# ============================================================

def test_chunk_status_payload_includes_paused_counters_and_list():
    """Whitebox del helper de armado del payload: el endpoint debe
    devolver `paused_chunks`, `pending_user_action_count`,
    `in_flight_count`, `failed_count`, `completed_count` en el response.
    Verificamos vía inspección del source que el dict de retorno tiene
    esas keys (drift detection cross-frontend ↔ backend)."""
    from routers import plans
    import inspect

    src = inspect.getsource(plans.api_chunk_status)
    for required_key in (
        '"in_flight_count"',
        '"pending_user_action_count"',
        '"failed_count"',
        '"completed_count"',
        '"paused_chunks"',
    ):
        assert required_key in src, (
            f"Key {required_key} missing en api_chunk_status response. "
            f"El frontend Dashboard.jsx la espera para diferenciar el chip."
        )
