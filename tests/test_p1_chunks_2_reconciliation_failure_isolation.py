"""
Tests P1-CHUNKS-2: aislación del fallo de reconciliación de reservas.

Bug original (audit P1-CHUNKS-2):
  `_reconcile_chunk_reservations` retornaba `bool` (True/False) y los dos
  call sites del worker dependían de un check `if not result:` para liberar
  reservas parciales y pausar el chunk. Discipline-by-convention.
  Si un futuro refactor introducía un tercer call site SIN ese check, o si
  alguien "simplificaba" llamando `_reconcile_chunk_reservations(...)` sin
  procesar el retorno, el chunk se marcaba 'completed' con
  reservation_status='partial' → siguiente chunk del mismo plan veía
  inventario sobreestimado tras los 5 min del bloqueo de pickup
  (`process_plan_chunk_queue` ~9844-9847) → overbooking silente.

Fix:
  1. `_reconcile_chunk_reservations` levanta excepción dedicada
     `ReservationReconciliationFailed` en lugar de retornar `False`.
     Subclase directa de `Exception` (no de `RuntimeError` ni `ValueError`)
     para evitar que excepts genéricos upstream la atrapen sin querer.
  2. Helper centralizado `_handle_reservation_reconciliation_exhausted`
     concentra el cleanup en orden: release → pause → push → system_alerts.
  3. Worker (cron_tasks._chunk_worker) usa `try/except
     ReservationReconciliationFailed` en ambos call sites; el helper hace
     todo el cleanup y el worker hace `return` para abortar antes de marcar
     'completed'.
  4. system_alerts dedupada por `alert_key` permite a SRE alertar si % de
     chunks con conciliación agotada sube en producción (indicador de CAS
     contention sostenida o DB latency).
"""
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import cron_tasks
from cron_tasks import (
    ReservationReconciliationFailed,
    _handle_reservation_reconciliation_exhausted,
)


# ---------------------------------------------------------------------------
# 1. Excepción exportada y forma correcta.
# ---------------------------------------------------------------------------
def test_excepcion_es_subclase_directa_de_exception():
    """Subclase directa de `Exception`: un upstream con
    `except RuntimeError:` no debe atraparla por accidente. El audit
    P1-CHUNKS-2 fue explícito en que la excepción no debe quedar enmascarada
    por excepts genéricos."""
    assert ReservationReconciliationFailed.__bases__ == (Exception,)


def test_excepcion_carga_metadata_util_para_diagnostico():
    """Un caller que atrape la excepción debe poder reconstruir contexto
    sin acudir a logs: user_id, chunk_id, attempts, last_error."""
    exc = ReservationReconciliationFailed(
        user_id="u-1", chunk_id="c-42", attempts=3,
        last_error="RuntimeError('CAS conflict')",
    )
    assert exc.user_id == "u-1"
    assert exc.chunk_id == "c-42"
    assert exc.attempts == 3
    assert "CAS conflict" in (exc.last_error or "")
    # str() debe ser informativo para logs.
    msg = str(exc)
    assert "u-1" in msg and "c-42" in msg and "3" in msg


# ---------------------------------------------------------------------------
# 2. `_reconcile_chunk_reservations`: nuevo contrato (raise vs return False).
# ---------------------------------------------------------------------------
def test_reconcile_raises_on_exhaustion_no_returns_false():
    """[P1-CHUNKS-2] Antes retornaba False; ahora levanta excepción. Cualquier
    caller que asuma bool ahora falla en boundary, evitando el modo silente
    de overbooking."""
    with patch.object(cron_tasks, "reserve_plan_ingredients", return_value=0), \
         patch.object(cron_tasks, "execute_sql_write"), \
         patch("time.sleep"):
        days = [{"meals": [{"ingredients": ["100g pollo", "200g arroz"]}]}]
        with pytest.raises(ReservationReconciliationFailed) as exc_info:
            cron_tasks._reconcile_chunk_reservations(
                "user-x", "chunk-y", days, max_retries=2,
            )
    # Metadata debe propagarse al handler:
    assert exc_info.value.user_id == "user-x"
    assert exc_info.value.chunk_id == "chunk-y"
    assert exc_info.value.attempts == 2


def test_reconcile_returns_true_on_success_unchanged():
    """Regresión: el path de éxito sigue retornando True (el caller no debe
    procesar la excepción ni hacer cleanup en este caso)."""
    with patch.object(cron_tasks, "reserve_plan_ingredients", return_value=10), \
         patch.object(cron_tasks, "execute_sql_write"):
        days = [{"meals": [{"ingredients": ["100g pollo", "200g arroz", "50g cebolla"]}]}]
        result = cron_tasks._reconcile_chunk_reservations(
            "user-x", "chunk-y", days, max_retries=3,
        )
    assert result is True


def test_reconcile_no_op_chunk_returns_true_no_raise():
    """Edge case: chunk sin ingredientes parseables. `_expected=0` →
    short-circuit a True. El caller no debe ver excepción."""
    with patch.object(cron_tasks, "reserve_plan_ingredients", return_value=0), \
         patch.object(cron_tasks, "execute_sql_write"):
        days = [{"meals": [{"name": "Plato sin ingredientes parseables"}]}]
        result = cron_tasks._reconcile_chunk_reservations(
            "user-x", "chunk-y", days, max_retries=1,
        )
    assert result is True


# ---------------------------------------------------------------------------
# 3. Helper centralizado `_handle_reservation_reconciliation_exhausted`.
# ---------------------------------------------------------------------------
def _build_exception() -> ReservationReconciliationFailed:
    return ReservationReconciliationFailed(
        user_id="user-1", chunk_id="chunk-42", attempts=3,
        last_error="RuntimeError('CAS conflict')",
    )


def test_handler_invokes_release_pause_push_alert():
    """Smoke test: el helper invoca las 4 acciones de cleanup en el orden
    documentado. release ANTES de pause para minimizar ventana de
    inconsistencia (audit P1-CHUNKS-2)."""
    with patch.object(cron_tasks, "release_chunk_reservations") as mock_release, \
         patch.object(cron_tasks, "_pause_chunk_for_pantry_refresh") as mock_pause, \
         patch.object(cron_tasks, "_dispatch_pantry_nudge") as mock_push, \
         patch.object(cron_tasks, "execute_sql_write") as mock_write:
        _handle_reservation_reconciliation_exhausted(
            exc=_build_exception(),
            week_number=2,
            meal_plan_id="plan-99",
            fresh_inventory=["pollo 500g"],
        )

    # 1. release
    mock_release.assert_called_once_with("user-1", "chunk-42")
    # 2. pause con reason canónico
    mock_pause.assert_called_once()
    pause_kwargs = mock_pause.call_args.kwargs
    assert pause_kwargs.get("reason") == "inventory_reconciliation_failed", (
        f"reason esperado='inventory_reconciliation_failed', recibido={pause_kwargs.get('reason')}"
    )
    assert pause_kwargs.get("fresh_inventory") == ["pollo 500g"]
    # 3. push al usuario
    mock_push.assert_called_once()
    push_kwargs = mock_push.call_args.kwargs
    # [P2-PANTRY-NUDGE-THROTTLE] el canal recibe el user_id posicional
    _push_call = mock_push.call_args
    assert (_push_call.kwargs.get("user_id") or (_push_call.args[0] if _push_call.args else None)) == "user-1"
    # 4. system_alerts INSERT
    sql_calls = [c for c in mock_write.call_args_list if "system_alerts" in str(c.args)]
    assert len(sql_calls) >= 1, "El helper debe persistir alerta en system_alerts."
    alert_call = sql_calls[0]
    sql_text = alert_call.args[0]
    sql_params = alert_call.args[1]
    # alert_key dedupada: user+plan+week.
    alert_key = sql_params[0]
    assert "reservation_reconciliation_exhausted" in alert_key
    assert "plan-99" in alert_key
    assert ":2" in alert_key  # week_number
    assert "ON CONFLICT" in sql_text  # idempotencia


def test_handler_continues_when_release_throws():
    """Si release_chunk_reservations falla (DB blip), el helper NO debe
    abortar el resto del cleanup. Pausar y notificar siempre vale la pena.
    El cron `_recover_orphan_chunk_reservations` cubre el residual."""
    with patch.object(cron_tasks, "release_chunk_reservations",
                      side_effect=RuntimeError("DB blip")) as mock_release, \
         patch.object(cron_tasks, "_pause_chunk_for_pantry_refresh") as mock_pause, \
         patch.object(cron_tasks, "_dispatch_pantry_nudge") as mock_push, \
         patch.object(cron_tasks, "execute_sql_write"):
        # NO debe propagar la excepción.
        _handle_reservation_reconciliation_exhausted(
            exc=_build_exception(),
            week_number=2,
            meal_plan_id="plan-99",
            fresh_inventory=[],
        )

    mock_release.assert_called_once()
    # Pausa y push siguen ejecutándose.
    mock_pause.assert_called_once()
    mock_push.assert_called_once()


def test_handler_continues_when_pause_throws():
    """Si pause falla, push y system_alerts deben seguir ejecutándose."""
    with patch.object(cron_tasks, "release_chunk_reservations"), \
         patch.object(cron_tasks, "_pause_chunk_for_pantry_refresh",
                      side_effect=RuntimeError("schema error")) as mock_pause, \
         patch.object(cron_tasks, "_dispatch_pantry_nudge") as mock_push, \
         patch.object(cron_tasks, "execute_sql_write") as mock_write:
        _handle_reservation_reconciliation_exhausted(
            exc=_build_exception(),
            week_number=2,
            meal_plan_id="plan-99",
            fresh_inventory=[],
        )

    mock_pause.assert_called_once()
    mock_push.assert_called_once()
    # system_alerts sigue activo.
    alert_calls = [c for c in mock_write.call_args_list if "system_alerts" in str(c.args)]
    assert len(alert_calls) >= 1


def test_handler_continues_when_push_throws():
    """Push fallido (Firebase saturado, permisos removidos) NO debe abortar
    el cleanup. system_alerts sigue ejecutándose para que SRE tenga
    visibilidad del incidente."""
    with patch.object(cron_tasks, "release_chunk_reservations"), \
         patch.object(cron_tasks, "_pause_chunk_for_pantry_refresh"), \
         patch.object(cron_tasks, "_dispatch_pantry_nudge",
                      side_effect=RuntimeError("Firebase down")) as mock_push, \
         patch.object(cron_tasks, "execute_sql_write") as mock_write:
        _handle_reservation_reconciliation_exhausted(
            exc=_build_exception(),
            week_number=2,
            meal_plan_id="plan-99",
            fresh_inventory=[],
        )

    mock_push.assert_called_once()
    alert_calls = [c for c in mock_write.call_args_list if "system_alerts" in str(c.args)]
    assert len(alert_calls) >= 1


def test_handler_no_emite_alerta_sin_meal_plan_id():
    """Si `meal_plan_id` es None (caso patológico — el chunk debería tener
    plan_id pero por algún bug viene vacío), saltamos la INSERT de
    system_alerts (alert_key sin plan_id sería ambigua, generaría
    duplicados que no se podrían dedupar). Cleanup operacional sigue:
    release + pause + push."""
    with patch.object(cron_tasks, "release_chunk_reservations") as mock_release, \
         patch.object(cron_tasks, "_pause_chunk_for_pantry_refresh") as mock_pause, \
         patch.object(cron_tasks, "_dispatch_pantry_nudge") as mock_push, \
         patch.object(cron_tasks, "execute_sql_write") as mock_write:
        _handle_reservation_reconciliation_exhausted(
            exc=_build_exception(),
            week_number=2,
            meal_plan_id=None,  # ← case crítico
            fresh_inventory=[],
        )

    mock_release.assert_called_once()
    mock_pause.assert_called_once()
    mock_push.assert_called_once()
    alert_calls = [c for c in mock_write.call_args_list if "system_alerts" in str(c.args)]
    assert len(alert_calls) == 0, "Sin meal_plan_id no debe emitirse system_alerts."


def test_handler_continues_when_alert_throws():
    """Si la INSERT de system_alerts falla (tabla missing, permisos rotos,
    DB lentitud), el helper retorna silenciosamente — el cleanup operacional
    ya se aplicó. La alerta SRE queda solo en logs."""
    def _fake_write(query, *args, **kwargs):
        if "system_alerts" in str(query):
            raise RuntimeError("permission denied")

    with patch.object(cron_tasks, "release_chunk_reservations") as mock_release, \
         patch.object(cron_tasks, "_pause_chunk_for_pantry_refresh") as mock_pause, \
         patch.object(cron_tasks, "_dispatch_pantry_nudge") as mock_push, \
         patch.object(cron_tasks, "execute_sql_write", side_effect=_fake_write):
        # NO debe propagar.
        _handle_reservation_reconciliation_exhausted(
            exc=_build_exception(),
            week_number=2,
            meal_plan_id="plan-99",
            fresh_inventory=[],
        )

    mock_release.assert_called_once()
    mock_pause.assert_called_once()
    mock_push.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Bug original reproducido: sin try/except, una llamada bare a reconcile
#    propaga la excepción y el caller NO marca el chunk como completed.
# ---------------------------------------------------------------------------
def test_escenario_bug_original_caller_sin_try_except_propaga():
    """
    PRE-FIX (P1-2 bool): un caller que olvidaba `if not result:` dejaba el
    chunk listo para marcarse 'completed' silenciosamente — el bug de
    overbooking. El sistema continuaba sin saber del fallo.

    POST-FIX (P1-CHUNKS-2 raise): un caller que olvida `try/except` recibe
    la excepción y propaga. El chunk no se marca completed (porque la
    función envolvente termina abruptamente). Si arriba hay un broad
    `except Exception`, lo loguea — pero la excepción dedicada deja rastro
    grep-able y el `_recover_orphan_chunk_reservations` cubre las reservas
    fantasma. El modo silencioso desaparece.
    """
    with patch.object(cron_tasks, "reserve_plan_ingredients", return_value=0), \
         patch.object(cron_tasks, "execute_sql_write"), \
         patch("time.sleep"):
        days = [{"meals": [{"ingredients": ["100g pollo", "200g arroz"]}]}]
        # Caller "olvidadizo": no procesa el retorno ni atrapa.
        # Antes esto pasaba silencioso (devolvía False y se ignoraba);
        # ahora levanta — el caller debe arreglar su lógica.
        with pytest.raises(ReservationReconciliationFailed):
            cron_tasks._reconcile_chunk_reservations(
                "user-1", "chunk-1", days, max_retries=2,
            )
