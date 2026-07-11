"""[P1-2] / [P1-CHUNKS-2] Reconciliación de reservas agotada → pausa, no overbooking.

Antes del fix original (P1-2):
    `_reconcile_chunk_reservations` retornaba `None` y los callers no
    distinguían éxito de fracaso → el worker marcaba 'completed' con
    reservation_status='partial' → siguiente chunk del mismo plan veía
    inventario sobreestimado → overbooking silente.

P1-2 (anterior):
    `_reconcile_chunk_reservations` migró a `bool`:
      - `True`  → reconciliación OK (≥ 50% reservado), reservation_status='ok'.
      - `False` → reconciliación AGOTADA, reservation_status persiste 'partial'.
    Callers debían chequear `if not result:` para liberar+pausar.
    Funcionaba pero dependía de discipline-by-convention; un futuro caller
    que olvidara revisar el bool reabría el bug.

P1-CHUNKS-2 (actual):
    `_reconcile_chunk_reservations` migró a EXCEPCIÓN dedicada
    (`ReservationReconciliationFailed`). Callers están obligados por la firma
    a manejar con `try/except` o propagar explícitamente. El cleanup
    (release + pause + push + system_alerts) está centralizado en
    `_handle_reservation_reconciliation_exhausted` para evitar divergencias
    entre los call sites del worker.
"""
import inspect
from unittest.mock import patch

import pytest

import cron_tasks


def test_reconcile_returns_true_on_success():
    """Caso feliz: reconciliación logra >= 50% reservado en el primer intento.
    Debe retornar True para que el caller continúe normalmente a T2."""
    with patch.object(cron_tasks, "reserve_plan_ingredients", return_value=10), \
         patch.object(cron_tasks, "execute_sql_write") as mock_write:
        days = [{"meals": [{"ingredients": ["100g pollo", "200g arroz", "50g cebolla"]}]}]
        result = cron_tasks._reconcile_chunk_reservations("user-1", "chunk-1", days, max_retries=3)
    assert result is True, (
        "_reconcile_chunk_reservations debe retornar True cuando reservó >= 50%. "
        "Sin esto, el caller no puede distinguir éxito de fracaso."
    )
    # Verifica que marcó reservation_status='ok'.
    mock_write.assert_called()
    sql_calls = [c.args[0] for c in mock_write.call_args_list]
    assert any("'ok'" in s for s in sql_calls), (
        "Reconciliación exitosa debe escribir reservation_status='ok' en plan_chunk_queue."
    )


def test_reconcile_raises_when_exhausted():
    """[P1-CHUNKS-2] Caso crítico: reconciliación falla los 3 intentos. Debe
    levantar `ReservationReconciliationFailed` con metadata útil. Antes
    retornaba `False` y dependía de que el caller lo chequeara — ahora la
    firma fuerza `try/except`."""
    with patch.object(cron_tasks, "reserve_plan_ingredients", return_value=0), \
         patch.object(cron_tasks, "execute_sql_write"), \
         patch("time.sleep"):
        days = [{"meals": [{"ingredients": ["100g pollo", "200g arroz", "50g cebolla"]}]}]
        with pytest.raises(cron_tasks.ReservationReconciliationFailed) as exc_info:
            cron_tasks._reconcile_chunk_reservations(
                "user-1", "chunk-1", days, max_retries=3,
            )
    # Metadata útil para el handler centralizado.
    assert exc_info.value.user_id == "user-1"
    assert exc_info.value.chunk_id == "chunk-1"
    assert exc_info.value.attempts == 3
    # last_error es None porque la causa fue ratio insuficiente, no excepción.
    assert exc_info.value.last_error is None


def test_reconcile_raises_when_all_attempts_throw():
    """[P1-CHUNKS-2] Si todas las invocaciones a `reserve_plan_ingredients`
    lanzan excepción (DB unreachable, schema error), reconciliación se
    considera agotada y levanta `ReservationReconciliationFailed` con
    `last_error` populado para diagnóstico."""
    def _always_raise(*a, **kw):
        raise RuntimeError("simulated DB blip")
    with patch.object(cron_tasks, "reserve_plan_ingredients", side_effect=_always_raise), \
         patch.object(cron_tasks, "execute_sql_write"), \
         patch("time.sleep"):
        days = [{"meals": [{"ingredients": ["100g pollo"]}]}]
        with pytest.raises(cron_tasks.ReservationReconciliationFailed) as exc_info:
            cron_tasks._reconcile_chunk_reservations(
                "user-1", "chunk-1", days, max_retries=2,
            )
    assert exc_info.value.attempts == 2
    assert "simulated DB blip" in (exc_info.value.last_error or "")


def test_reconcile_returns_true_when_partial_then_succeeds():
    """Si el primer intento es parcial pero el segundo logra ≥ 50%, retorna
    True. Esto es el caso normal de CAS retries (transient conflict)."""
    call_count = {"n": 0}

    def _flaky_reserve(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return 0  # primer intento falla
        return 5  # segundo intento OK

    with patch.object(cron_tasks, "reserve_plan_ingredients", side_effect=_flaky_reserve), \
         patch.object(cron_tasks, "execute_sql_write"), \
         patch("time.sleep"):
        days = [{"meals": [{"ingredients": ["100g pollo", "200g arroz"]}]}]
        result = cron_tasks._reconcile_chunk_reservations("user-1", "chunk-1", days, max_retries=3)
    assert result is True
    assert call_count["n"] == 2, "Debió haber retried una vez después del fallo inicial."


def test_worker_calls_handler_on_exhausted_reconcile():
    """[P1-CHUNKS-2] Regression guard del comportamiento en el worker:
    cuando `_reconcile_chunk_reservations` levanta
    `ReservationReconciliationFailed`, el worker DEBE atrapar la excepción
    e invocar `_handle_reservation_reconciliation_exhausted` antes de hacer
    return. Si este guard se rompe (i.e., alguien hace 'simplificar' y vuelve
    a ignorar el manejo), el chunk volvería a marcarse 'completed' con
    reservas parciales y reabriría el bug de overbooking."""
    src = inspect.getsource(cron_tasks)
    assert "except ReservationReconciliationFailed" in src, (
        "El worker no maneja `ReservationReconciliationFailed` — alguien "
        "removió el try/except y reabrió el bug de overbooking."
    )
    assert "_handle_reservation_reconciliation_exhausted" in src, (
        "El helper centralizado de cleanup no se invoca; los call sites "
        "podrían haber regresado a la copia inline divergente."
    )


def test_handler_does_release_pause_push_alert_in_order():
    """[P1-CHUNKS-2] Orden correcto en el helper: release PRIMERO, luego
    pause, push, system_alerts. Si pausamos antes de liberar y release falla,
    quedan reservas fantasma + chunk pausado. El helper debe ejecutar release
    primero para minimizar la ventana de inconsistencia.

    También verificamos que reason='inventory_reconciliation_failed' aparezca
    en el helper, no inline en el worker — eso garantiza que la lógica está
    centralizada (audit P1-CHUNKS-2 quería un único punto de mantenimiento)."""
    src = inspect.getsource(cron_tasks._handle_reservation_reconciliation_exhausted)
    # Buscamos el INVOCATIONS reales (parens) para los 3 helpers, y el INSERT
    # SQL para system_alerts — evita matchear menciones en el docstring.
    release_idx = src.find("release_chunk_reservations(")
    pause_idx = src.find("_pause_chunk_for_pantry_refresh(")
    # [P2-PANTRY-NUDGE-THROTTLE · 2026-07-11] el push de la clase nevera va por el
    # canal único con cooldown — mismo orden del handler, nuevo nombre.
    push_idx = src.find("_dispatch_pantry_nudge(")
    alert_idx = src.find("INSERT INTO system_alerts")
    assert release_idx != -1, "release_chunk_reservations no aparece en el helper."
    assert pause_idx != -1, "_pause_chunk_for_pantry_refresh no aparece en el helper."
    assert push_idx != -1, "_dispatch_pantry_nudge no aparece en el helper."
    assert alert_idx != -1, "INSERT INTO system_alerts no aparece en el helper."
    assert release_idx < pause_idx < push_idx < alert_idx, (
        "Orden incorrecto: release → pause → push → alert. "
        f"Posiciones: release={release_idx}, pause={pause_idx}, "
        f"push={push_idx}, alert={alert_idx}."
    )
    # Reason canónico vive en el helper, no inline.
    assert 'reason="inventory_reconciliation_failed"' in src, (
        "El reason 'inventory_reconciliation_failed' debe vivir en el helper "
        "centralizado, no inline en el worker."
    )


def test_reconcile_function_docstring_documents_exception_contract():
    """[P1-CHUNKS-2] La docstring de `_reconcile_chunk_reservations` debe
    documentar explícitamente la nueva excepción para que un caller futuro
    sepa que debe usar try/except (no chequeo bool)."""
    src = inspect.getsource(cron_tasks._reconcile_chunk_reservations)
    assert "ReservationReconciliationFailed" in src, (
        "Docstring no menciona la excepción dedicada. "
        "Sin esto, callers nuevos podrían tratarla como `None` y reabrir el bug."
    )
    assert "P1-CHUNKS-2" in src, (
        "Docstring debe mencionar P1-CHUNKS-2 para trazabilidad del cambio "
        "de contrato bool → exception."
    )


def test_exception_class_exposed_at_module_level():
    """[P1-CHUNKS-2] La clase `ReservationReconciliationFailed` debe estar
    accesible como atributo público de `cron_tasks`. Sin esto, tests externos
    e integraciones tendrían que importarla desde un path interno frágil."""
    assert hasattr(cron_tasks, "ReservationReconciliationFailed")
    assert issubclass(cron_tasks.ReservationReconciliationFailed, Exception)
    # Debe ser subclase DIRECTA de Exception, no de RuntimeError/ValueError —
    # un broad `except RuntimeError:` upstream no debería atraparla.
    assert cron_tasks.ReservationReconciliationFailed.__bases__ == (Exception,)
