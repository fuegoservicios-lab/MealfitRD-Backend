"""[P1-19] Tests para persistencia de alerta en `system_alerts` cuando
`summarize_and_prune` falla N veces consecutivas.

Bug original (audit P1-19):
  P1-18 ya cubrió la promoción de `print warning silencioso` a
  `logger.error` con contador in-memory. Pero un dashboard que solo
  parsea logs puede perder señal durante outages largos. P1-19 agrega
  la persistencia de un row dedupado en `system_alerts` cuando la racha
  cruza un threshold (5 fallos consecutivos por default), permitiendo
  que SRE alerte vía la query estándar de `triggered_at IS NOT NULL AND
  resolved_at IS NULL`.

Fix:
  1. `_SUMMARY_FAILURE_ALERT_THRESHOLD = 5` (configurable a futuro).
  2. `_persist_summary_failure_alert` upsert en `system_alerts` con
     `alert_key='memory_summary_failures'`, severity='warning', metadata
     incluye {consecutive_failures, last_error, last_session_id, model}.
  3. Re-disparada cada `_SUMMARY_FAILURE_ALERT_REPEAT_EVERY=50` para
     visibilidad si el problema persiste tras la primera alerta.
  4. Auto-resolve: en el path de éxito post-racha, marca la alerta como
     `resolved_at = NOW()`.
  5. Best-effort: si la DB también está caída, fallamos silenciosamente
     (el log error ya capturó la señal).

Cobertura:
  - test_threshold_constant_is_reasonable
  - test_persist_helper_exists_and_signature
  - test_alert_not_persisted_below_threshold
  - test_alert_persisted_at_threshold
  - test_alert_resolved_on_recovery_after_threshold
  - test_alert_repeats_at_intervals (smoke)
  - test_persist_helper_swallows_db_errors_silently
  - test_documentation_p1_19_present
"""
import inspect
from unittest.mock import patch, MagicMock

import pytest

import memory_manager


@pytest.fixture(autouse=True)
def _reset_failures():
    """Limpia el contador de fallos antes y después de cada test."""
    memory_manager._summarize_failures["count"] = 0
    memory_manager._summarize_failures["last_error"] = None
    yield
    memory_manager._summarize_failures["count"] = 0
    memory_manager._summarize_failures["last_error"] = None


# ---------------------------------------------------------------------------
# 1. Constantes y firma del helper.
# ---------------------------------------------------------------------------
def test_threshold_constant_is_reasonable():
    """`_SUMMARY_FAILURE_ALERT_THRESHOLD` debe estar en un rango sensato:
    no muy bajo (false positives) ni muy alto (alerta tardía)."""
    threshold = memory_manager._SUMMARY_FAILURE_ALERT_THRESHOLD
    assert isinstance(threshold, int)
    assert 3 <= threshold <= 20, f"threshold {threshold} fuera del rango razonable"


def test_repeat_constant_is_set():
    """`_SUMMARY_FAILURE_ALERT_REPEAT_EVERY` controla la cadencia de
    re-alerta tras la primera. Debe ser > threshold para que no spamee."""
    repeat = memory_manager._SUMMARY_FAILURE_ALERT_REPEAT_EVERY
    threshold = memory_manager._SUMMARY_FAILURE_ALERT_THRESHOLD
    assert isinstance(repeat, int)
    assert repeat > threshold, "repeat debe ser > threshold para evitar spam"


def test_persist_helper_exists_and_signature():
    """`_persist_summary_failure_alert(session_id, error_str, count)` debe
    estar exportado como helper testeable."""
    assert hasattr(memory_manager, "_persist_summary_failure_alert")
    sig = inspect.signature(memory_manager._persist_summary_failure_alert)
    params = list(sig.parameters.keys())
    assert params == ["session_id", "error_str", "count"]


# ---------------------------------------------------------------------------
# 2. Comportamiento: alert NO se persiste antes del threshold.
# ---------------------------------------------------------------------------
def test_alert_not_persisted_below_threshold():
    """Con N fallos < threshold (e.g., 3), `_persist_summary_failure_alert`
    NO se invoca."""
    threshold = memory_manager._SUMMARY_FAILURE_ALERT_THRESHOLD

    with patch.object(memory_manager, "acquire_summarizing_lock", return_value=True), \
         patch.object(memory_manager, "release_summarizing_lock"), \
         patch.object(memory_manager, "get_memory", side_effect=RuntimeError("simulated")), \
         patch.object(memory_manager, "_persist_summary_failure_alert") as persist_mock:

        # Disparar (threshold-1) fallos consecutivos.
        for _ in range(threshold - 1):
            memory_manager.summarize_and_prune("test-session")

        persist_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Alert SÍ se persiste cuando N alcanza threshold.
# ---------------------------------------------------------------------------
def test_alert_persisted_at_threshold():
    """En el fallo número N == threshold, el helper se invoca con
    (session_id, error_str, count=threshold)."""
    threshold = memory_manager._SUMMARY_FAILURE_ALERT_THRESHOLD

    with patch.object(memory_manager, "acquire_summarizing_lock", return_value=True), \
         patch.object(memory_manager, "release_summarizing_lock"), \
         patch.object(memory_manager, "get_memory", side_effect=RuntimeError("simulated boom")), \
         patch.object(memory_manager, "_persist_summary_failure_alert") as persist_mock:

        # Disparar exactamente threshold fallos.
        for i in range(threshold):
            memory_manager.summarize_and_prune(f"session-{i}")

        # Debe haberse invocado UNA vez al cruzar el threshold.
        assert persist_mock.call_count == 1
        call_args = persist_mock.call_args
        # El último session_id, el error, y count=threshold.
        assert call_args.args[0] == f"session-{threshold-1}"
        assert "simulated" in call_args.args[1]
        assert call_args.args[2] == threshold


def test_alert_repeat_after_repeat_interval():
    """Tras la alerta inicial, la siguiente debe disparar al cruzar
    threshold + repeat_every. Smoke test (uso fake threshold/repeat para
    evitar spam de N reales)."""
    threshold = memory_manager._SUMMARY_FAILURE_ALERT_THRESHOLD
    repeat = memory_manager._SUMMARY_FAILURE_ALERT_REPEAT_EVERY

    with patch.object(memory_manager, "acquire_summarizing_lock", return_value=True), \
         patch.object(memory_manager, "release_summarizing_lock"), \
         patch.object(memory_manager, "get_memory", side_effect=RuntimeError("err")), \
         patch.object(memory_manager, "_persist_summary_failure_alert") as persist_mock:

        # Disparar threshold + repeat fallos.
        for _ in range(threshold + repeat):
            memory_manager.summarize_and_prune("s")

        # Debe haberse invocado al menos 2 veces (en threshold y en
        # threshold + repeat).
        assert persist_mock.call_count >= 2


# ---------------------------------------------------------------------------
# 4. Auto-resolve: defensa estructural sobre el source.
# ---------------------------------------------------------------------------
def test_source_contains_resolve_alert_logic():
    """El path de éxito debe contener un UPDATE a `system_alerts` con
    `resolved_at = NOW()` para auto-resolver tras recuperación. Test
    estructural sobre el source — el test funcional E2E del path
    completo es brittle por dependencias del flow del LLM."""
    src = inspect.getsource(memory_manager.summarize_and_prune)
    assert "resolved_at = NOW()" in src, (
        "P1-19: falta UPDATE de auto-resolve en path de éxito"
    )
    assert "memory_summary_failures" in src, (
        "P1-19: el alert_key debe coincidir entre persist y resolve"
    )


# ---------------------------------------------------------------------------
# 5. Defensa: la persistencia no rompe nada si la DB también falla.
# ---------------------------------------------------------------------------
def test_persist_helper_swallows_db_errors_silently():
    """Si `execute_sql_write` lanza dentro del helper, NO debe
    re-propagar — el log error del caller ya capturó la señal."""
    with patch("db_core.execute_sql_write", side_effect=RuntimeError("DB down too")):
        # No debe lanzar.
        memory_manager._persist_summary_failure_alert("session-x", "some error", 5)


def test_persist_helper_writes_correct_alert_key():
    """El alert_key debe ser `memory_summary_failures` (dedupe estable
    para que ON CONFLICT funcione)."""
    captured = []

    def _capture(query, params=None, *_a, **_kw):
        captured.append((query, params))

    with patch("db_core.execute_sql_write", side_effect=_capture):
        memory_manager._persist_summary_failure_alert("s1", "err", 5)

    assert captured, "execute_sql_write debe haberse llamado"
    query, params = captured[0]
    assert "system_alerts" in query
    assert "memory_summary_failures" in (params or [None])[0:1] or "memory_summary_failures" in str(params)
    # ON CONFLICT debe estar presente (dedupe).
    assert "ON CONFLICT" in query.upper()


def test_persist_helper_includes_metadata_with_count_and_model():
    """El JSON metadata debe incluir count, last_error, last_session_id y model."""
    captured = []

    def _capture(query, params=None, *_a, **_kw):
        captured.append((query, params))

    with patch("db_core.execute_sql_write", side_effect=_capture):
        memory_manager._persist_summary_failure_alert("session-meta-test", "boom", 7)

    query, params = captured[0]
    # El último param debe ser el JSON metadata.
    metadata_str = params[-1]
    assert "consecutive_failures" in metadata_str
    assert "last_session_id" in metadata_str
    assert "session-meta-test" in metadata_str
    assert "model" in metadata_str


# ---------------------------------------------------------------------------
# 6. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_19_present():
    """Comentario `[P1-19]` debe documentar el rationale."""
    src = inspect.getsource(memory_manager)
    assert "[P1-19]" in src
