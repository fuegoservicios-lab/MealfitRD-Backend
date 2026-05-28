"""[P1-CRON-BUNDLE · 2026-05-26] Tests parser-based + funcionales para el
bundle P1 del audit prod-readiness 2026-05-26 (post-P0-CHAT-DELETE-IDOR).

5 sub-P1 cerrados con un helper SSOT y un cron nuevo:

  - **P1-1 (P1-KV-SWEEP)**: cron `_sweep_stale_app_kv_store_prefixes` con
    catálogo declarativo `_KV_SWEEP_PREFIXES` (4 entries: `title_gen_
    inflight:`, `pending_pipeline:`, `rag_`, `reflection_`).
  - **P1-2 (P1-CRON-CONSECUTIVE-FAIL)**: helper SSOT
    `_track_cron_consecutive_failure` aplicado en 3 crons.
  - **P1-3 (P1-KNOB-CLAMPS)**: 3 knobs migrados a `_env_int(...,
    validator=...)`.
  - **P1-4 (P1-CRON-TIMEOUT)**: helper `_call_with_timeout` + aplicación
    en `_proactive_refresh_pending_pantry_snapshots`.
  - **P1-5 (P1-ROLLING-ABANDONED)**: nueva branch del sweep
    `_alert_stranded_partial_plans`.

NOTA: los tests parser-based escanean source de prod con regex. Si renombras
un símbolo (`_track_cron_consecutive_failure`, `_KV_SWEEP_PREFIXES`, etc.)
el test falla — esto es por diseño: el rename debe documentarse y el test
actualizarse en el mismo commit (anchor de la decisión).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"
_CONSTANTS = _BACKEND_ROOT / "constants.py"

# Asegurar que backend/ esté en sys.path para importar `knobs` directamente.
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# [P1-CRON-BUNDLE · 2026-05-26] Stub local de `apscheduler` para CI sin la dep
# instalada. `conftest.py` ya stubea langgraph + langchain_google_genai pero no
# apscheduler. Importar el módulo real cuando esté disponible (CI prod); fallback
# a MagicMock si no. Análogo al pattern usado por conftest.py para langgraph.
try:
    import apscheduler  # noqa: F401
    import apscheduler.triggers.cron  # noqa: F401
except Exception:
    sys.modules.setdefault("apscheduler", MagicMock())
    sys.modules.setdefault("apscheduler.triggers", MagicMock())
    sys.modules.setdefault("apscheduler.triggers.cron", MagicMock())


def _read_cron() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


def _read_constants() -> str:
    return _CONSTANTS.read_text(encoding="utf-8")


def _extract_function(src: str, name: str) -> str:
    """Devuelve el cuerpo de la función `name` desde `def <name>(` hasta el
    siguiente `def ` toplevel o EOF."""
    pattern = re.compile(
        rf"^def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:\n"
        rf"(?:[ \t]+.*\n|[ \t]*\n)+",
        re.MULTILINE,
    )
    m = pattern.search(src)
    assert m, f"No se encontró def `{name}` en cron_tasks.py."
    return m.group(0)


# ===========================================================================
# Marker anchor cross-link (cumple `test_p2_hist_audit_14_marker_test_link.py`)
# ===========================================================================

def test_marker_anchor_present_in_cron_tasks():
    src = _read_cron()
    assert "P1-CRON-BUNDLE" in src or "P1-KV-SWEEP" in src, (
        "Falta anchor del bundle P1 en `cron_tasks.py`. Sin anchor el "
        "test cross-link `test_p2_hist_audit_14_marker_test_link.py` "
        "fallará — el marker `P1-CRON-BUNDLE · 2026-05-26` debe matchear "
        "al menos un anchor en source de prod."
    )


# ===========================================================================
# P1-1 · P1-KV-SWEEP — cron _sweep_stale_app_kv_store_prefixes
# ===========================================================================

def test_kv_sweep_prefixes_catalog_exists():
    """`_KV_SWEEP_PREFIXES` debe declarar al menos los 4 prefixes observados."""
    src = _read_cron()
    assert "_KV_SWEEP_PREFIXES" in src, "Falta catálogo `_KV_SWEEP_PREFIXES`."
    # Los 4 prefixes deben estar declarados literales en el catálogo.
    for prefix in ("title_gen_inflight:", "pending_pipeline:", "rag_", "reflection_"):
        assert f'"prefix": "{prefix}"' in src or f"'prefix': '{prefix}'" in src, (
            f"Catálogo `_KV_SWEEP_PREFIXES` no declara prefix `{prefix}`. "
            f"Audit 2026-05-26 observó keys huérfanas con este prefix."
        )


def test_kv_sweep_function_exists_and_uses_helper():
    """`_sweep_stale_app_kv_store_prefixes` debe existir + iterar el catálogo
    + invocar `_track_cron_consecutive_failure`."""
    src = _read_cron()
    body = _extract_function(src, "_sweep_stale_app_kv_store_prefixes")
    assert "_KV_SWEEP_PREFIXES" in body, (
        "El sweep debe iterar `_KV_SWEEP_PREFIXES` — sin la iteración no GC nada."
    )
    assert "DELETE FROM app_kv_store" in body, (
        "El sweep debe ejecutar DELETE sobre app_kv_store."
    )
    assert "_track_cron_consecutive_failure" in body, (
        "El sweep debe usar `_track_cron_consecutive_failure` para escalar "
        "fallos consecutivos (P1-CRON-CONSECUTIVE-FAIL)."
    )


def test_kv_sweep_registered_in_scheduler():
    """El cron debe estar registrado en `register_plan_chunk_scheduler`."""
    src = _read_cron()
    body = _extract_function(src, "register_plan_chunk_scheduler")
    assert "sweep_stale_app_kv_store_prefixes" in body, (
        "El job `sweep_stale_app_kv_store_prefixes` debe estar registrado en "
        "`register_plan_chunk_scheduler`; sin registrar, el cron nunca corre."
    )
    assert "MEALFIT_KV_SWEEP_INTERVAL_MIN" in body, (
        "El intervalo del cron debe leerse del knob `MEALFIT_KV_SWEEP_INTERVAL_MIN`."
    )


# ===========================================================================
# P1-2 · P1-CRON-CONSECUTIVE-FAIL — helper SSOT + 3 callsites
# ===========================================================================

def test_helper_track_cron_consecutive_failure_exists():
    src = _read_cron()
    m = re.search(
        r"def\s+_track_cron_consecutive_failure\s*\("
        r"\s*cron_name\s*:\s*str\s*,"
        r"\s*alert_key\s*:\s*str\s*,"
        r"\s*alert_type\s*:\s*str\s*,"
        r"\s*alert_title\s*:\s*str\s*,"
        r"\s*is_failure\s*:\s*bool\s*,",
        src,
    )
    assert m is not None, (
        "La signature de `_track_cron_consecutive_failure` no coincide con "
        "el contrato declarado (cron_name, alert_key, alert_type, alert_title, "
        "is_failure). Si la renombraste, actualiza este test + memoria del P1-CRON-BUNDLE."
    )


@pytest.mark.parametrize("cron_name,alert_key", [
    ("drain_pending_facts_queue", "drain_pending_facts_queue_failures_burst"),
    ("resolve_stale_scheduler_alerts", "resolve_stale_scheduler_alerts_failures_burst"),
    ("recover_failed_chunks_for_long_plans", "recover_failed_chunks_for_long_plans_failures_burst"),
    ("sweep_stale_app_kv_store_prefixes", "sweep_stale_app_kv_store_prefixes_failures_burst"),
])
def test_3_crons_use_track_helper(cron_name, alert_key):
    """Los 3 crons P1-2 + el nuevo P1-1 deben invocar el helper con su alert_key."""
    src = _read_cron()
    func_name = f"_{cron_name}"
    body = _extract_function(src, func_name)
    assert "_track_cron_consecutive_failure" in body, (
        f"`{func_name}` debe invocar `_track_cron_consecutive_failure`. "
        f"Sin tracking, fallos consecutivos quedan silenciosos (P1-2 reabre)."
    )
    assert alert_key in body, (
        f"`{func_name}` debe usar alert_key=`{alert_key}` (convención "
        f"`<cron>_failures_burst`). Si lo cambiaste, actualiza la fila "
        f"correspondiente en `docs/system_alerts_resolution_table.md`."
    )


# ===========================================================================
# P1-3 · P1-KNOB-CLAMPS — 3 knobs migrados a validator
# ===========================================================================

def test_knob_coh_alert_min_plans_has_clamp():
    src = _read_cron()
    # Buscamos `_env_int("MEALFIT_COH_ALERT_MIN_PLANS", 5, validator=...)`.
    m = re.search(
        r'_env_int\(\s*\n?\s*"MEALFIT_COH_ALERT_MIN_PLANS"\s*,\s*5\s*,\s*\n?\s*validator\s*=',
        src,
    )
    assert m is not None, (
        "`MEALFIT_COH_ALERT_MIN_PLANS` debe usar `_env_int(..., validator=...)` "
        "con clamp [1, 10_000]. Pre-P1-KNOB-CLAMPS el knob aceptaba `0` y `-5` "
        "silenciosamente."
    )


def test_knob_failed_deductions_threshold_has_clamp():
    src = _read_cron()
    m = re.search(
        r'_env_int\(\s*\n?\s*"MEALFIT_FAILED_DEDUCTIONS_ALERT_THRESHOLD"\s*,\s*25\s*,\s*\n?\s*validator\s*=',
        src,
    )
    assert m is not None, (
        "`MEALFIT_FAILED_DEDUCTIONS_ALERT_THRESHOLD` debe usar `_env_int(..., "
        "validator=lambda v: 1 <= v <= 100_000)`."
    )


def test_chunk_recovery_batch_limit_uses_env_int_validator():
    src = _read_constants()
    m = re.search(
        r'CHUNK_RECOVERY_BATCH_LIMIT\s*=\s*_env_int\(\s*\n?\s*"CHUNK_RECOVERY_BATCH_LIMIT"\s*,\s*20\s*,\s*validator\s*=',
        src,
    )
    assert m is not None, (
        "`CHUNK_RECOVERY_BATCH_LIMIT` debe migrar de `max(1, int(os.environ.get(...)))` "
        "a `_env_int(..., validator=lambda v: 1 <= v <= 1000)`. Pre-fix tenía "
        "floor pero NO ceiling — operador con `CHUNK_RECOVERY_BATCH_LIMIT=10000` "
        "causaba OOM/timeout del cron de recovery."
    )


# ===========================================================================
# P1-4 · P1-CRON-TIMEOUT — helper + per-user timeout en proactive refresh
# ===========================================================================

def test_call_with_timeout_helper_exists():
    src = _read_cron()
    m = re.search(
        r"def\s+_call_with_timeout\s*\(\s*fn\s*,\s*timeout_s\s*:\s*float\s*,\s*\*args\s*,\s*\*\*kwargs\s*\)",
        src,
    )
    assert m is not None, (
        "Falta helper `_call_with_timeout(fn, timeout_s, *args, **kwargs)`. "
        "Sin él, el cron proactive refresh queda sin defensa contra un user "
        "atascado (P1-4 reabre)."
    )


def test_proactive_refresh_uses_per_user_timeout():
    src = _read_cron()
    body = _extract_function(src, "_proactive_refresh_pending_pantry_snapshots")
    assert "_call_with_timeout" in body, (
        "`_proactive_refresh_pending_pantry_snapshots` debe envolver "
        "`get_user_inventory_net` en `_call_with_timeout` para que un user "
        "lento NO bloquee el resto del cron."
    )
    assert "MEALFIT_PROACTIVE_REFRESH_PER_USER_TIMEOUT_S" in body, (
        "El timeout per-user debe leerse del knob "
        "`MEALFIT_PROACTIVE_REFRESH_PER_USER_TIMEOUT_S` (default 30, clamp [5, 300])."
    )
    assert "MEALFIT_PROACTIVE_REFRESH_TOTAL_BUDGET_S" in body, (
        "Budget total del tick debe leerse del knob "
        "`MEALFIT_PROACTIVE_REFRESH_TOTAL_BUDGET_S` (default 600, clamp [60, 1800])."
    )


def test_call_with_timeout_raises_on_slow_callable():
    """Funcional: si `fn` excede `timeout_s`, debe levantar TimeoutError."""
    from cron_tasks import _call_with_timeout
    import time

    def slow_fn():
        time.sleep(2.0)
        return "completed"

    with pytest.raises(TimeoutError):
        _call_with_timeout(slow_fn, timeout_s=0.5)


def test_call_with_timeout_returns_value_when_fast():
    from cron_tasks import _call_with_timeout

    def fast_fn(x, y):
        return x + y

    assert _call_with_timeout(fast_fn, 5.0, 3, y=4) == 7


def test_call_with_timeout_propagates_exceptions():
    from cron_tasks import _call_with_timeout

    def boom():
        raise ValueError("inner error")

    with pytest.raises(ValueError, match="inner error"):
        _call_with_timeout(boom, 5.0)


# ===========================================================================
# P1-5 · P1-ROLLING-ABANDONED — segunda branch del sweep stranded
# ===========================================================================

def test_stranded_partial_has_rolling_abandoned_branch():
    src = _read_cron()
    body = _extract_function(src, "_alert_stranded_partial_plans")
    assert "P1-ROLLING-ABANDONED" in body, (
        "Falta anchor `P1-ROLLING-ABANDONED` en `_alert_stranded_partial_plans`. "
        "La extensión del sweep debe estar claramente marcada."
    )
    assert "plan_rolling_abandoned:" in body, (
        "El sweep debe emitir alert_key con prefix `plan_rolling_abandoned:`."
    )
    assert "MEALFIT_ROLLING_ABANDONED_AGE_HOURS" in body, (
        "Knob `MEALFIT_ROLLING_ABANDONED_AGE_HOURS` (default 168, clamp [24, 720]) "
        "debe controlar el umbral de edad. Sin knob, no hay tuning operacional."
    )


def test_rolling_abandoned_query_filters_correct_state():
    """La query SQL debe filtrar `partial+days>0+age>N` Y `chunks all in
    terminal state (pending_user_action / cancelled / dead_lettered)` Y
    SÍ existe al menos un chunk pending_user_action."""
    src = _read_cron()
    body = _extract_function(src, "_alert_stranded_partial_plans")
    # Verificar predicates clave del SQL (todos en el SELECT abandoned_rows).
    expected_predicates = [
        # Status partial + días > 0 (distingue de P2-STRANDED-PARTIAL que cubre days=[]).
        "'generation_status' = 'partial'",
        # Día count > 0.
        "> 0",
        # NOT EXISTS chunks activos.
        "NOT EXISTS",
        "'pending', 'processing'",
        # EXISTS pending_user_action.
        "EXISTS",
        "'pending_user_action'",
    ]
    for snippet in expected_predicates:
        assert snippet in body, (
            f"La query SQL del P1-ROLLING-ABANDONED debe contener `{snippet}`. "
            f"Sin ese predicate, el sweep no distingue planes vivos de abandonados."
        )


def test_rolling_abandoned_uses_auto_explicit_resolution():
    """Modelo Auto-explicit: sin abandoned, UPDATE resolved_at = NOW()."""
    src = _read_cron()
    body = _extract_function(src, "_alert_stranded_partial_plans")
    # Debe haber un sweep que resuelva alerts viejas si no hay rows.
    m = re.search(
        r"UPDATE system_alerts\s+SET resolved_at = NOW\(\).*?"
        r"alert_key LIKE.*?plan_rolling_abandoned",
        body,
        re.DOTALL,
    )
    assert m is not None, (
        "El sweep debe incluir `UPDATE system_alerts SET resolved_at = NOW() "
        "WHERE alert_key LIKE 'plan_rolling_abandoned:%' AND resolved_at IS NULL` "
        "para implementar el modelo Auto-explicit (alerts viejas cierran cuando "
        "ya no hay planes abandoned)."
    )


# ===========================================================================
# Funcional: helper consecutive-failure (con mocks)
# ===========================================================================

def test_track_helper_inserts_failure_count_and_alert_at_threshold():
    """Funcional: cuando `is_failure=True` Y count cruza el threshold, emite
    alert. Mocks de `execute_sql_query` / `execute_sql_write`."""
    from cron_tasks import _track_cron_consecutive_failure

    # Simular state KV con prev_count=1; tras esta llamada count=2 >= threshold(2)
    # → debe emitir alert.
    fake_query = MagicMock(return_value={"value": {"count": 1}})
    fake_write = MagicMock()

    with patch("cron_tasks.execute_sql_query", fake_query), \
         patch("cron_tasks.execute_sql_write", fake_write):
        _track_cron_consecutive_failure(
            "my_cron",
            "my_cron_failures_burst",
            "my_cron_failures_burst",
            "Cron my_cron con fallos consecutivos",
            is_failure=True,
            threshold=2,
            last_error="boom",
        )

    # 2 writes: 1 UPSERT al KV (count=2) + 1 INSERT al system_alerts.
    assert fake_write.call_count == 2
    sql_first = fake_write.call_args_list[0].args[0]
    sql_second = fake_write.call_args_list[1].args[0]
    assert "app_kv_store" in sql_first
    assert "system_alerts" in sql_second
    assert "my_cron_failures_burst" in fake_write.call_args_list[1].args[1]


def test_track_helper_resets_count_on_success():
    """Funcional: `is_failure=False` con prev_count>0 → reset KV +
    UPDATE alert resolved_at = NOW()."""
    from cron_tasks import _track_cron_consecutive_failure

    fake_query = MagicMock(return_value={"value": {"count": 3}})
    fake_write = MagicMock()

    with patch("cron_tasks.execute_sql_query", fake_query), \
         patch("cron_tasks.execute_sql_write", fake_write):
        _track_cron_consecutive_failure(
            "my_cron",
            "my_cron_failures_burst",
            "my_cron_failures_burst",
            "Cron my_cron con fallos consecutivos",
            is_failure=False,
            threshold=2,
        )

    # 2 writes: 1 UPSERT KV (count=0) + 1 UPDATE system_alerts resolved_at.
    assert fake_write.call_count == 2
    sql_first = fake_write.call_args_list[0].args[0]
    params_first = fake_write.call_args_list[0].args[1]
    sql_second = fake_write.call_args_list[1].args[0]
    assert "app_kv_store" in sql_first
    # params_first es tupla (key, json_str). El JSON debe contener "count": 0.
    assert '"count": 0' in params_first[1]
    assert "system_alerts" in sql_second
    assert "resolved_at = NOW()" in sql_second


def test_track_helper_noop_on_first_success():
    """Funcional: `is_failure=False` con prev_count=0 → no escribe nada
    (no hay alert que cerrar)."""
    from cron_tasks import _track_cron_consecutive_failure

    fake_query = MagicMock(return_value={"value": {"count": 0}})
    fake_write = MagicMock()

    with patch("cron_tasks.execute_sql_query", fake_query), \
         patch("cron_tasks.execute_sql_write", fake_write):
        _track_cron_consecutive_failure(
            "my_cron",
            "my_cron_failures_burst",
            "my_cron_failures_burst",
            "Cron my_cron con fallos consecutivos",
            is_failure=False,
            threshold=2,
        )

    # 0 writes — nada que hacer (caso happy path estable).
    assert fake_write.call_count == 0


def test_track_helper_silences_internal_errors():
    """Funcional: si execute_sql_query lanza, el helper NO debe propagar."""
    from cron_tasks import _track_cron_consecutive_failure

    def boom(*args, **kwargs):
        raise RuntimeError("DB down")

    with patch("cron_tasks.execute_sql_query", side_effect=boom), \
         patch("cron_tasks.execute_sql_write", side_effect=boom):
        # No raises:
        _track_cron_consecutive_failure(
            "my_cron",
            "my_cron_failures_burst",
            "my_cron_failures_burst",
            "Cron my_cron",
            is_failure=True,
        )


# ===========================================================================
# Funcional: _env_int validator (P1-3)
# ===========================================================================

def test_env_int_validator_rejects_out_of_range(monkeypatch):
    """Funcional: `_env_int` con validator debe caer al default si el env
    var está fuera de rango. (Verifica que P1-3 está bien wireado.)"""
    from knobs import _env_int

    # Out-of-range bajo:
    monkeypatch.setenv("TEST_KNOB_P1_BUNDLE", "0")
    val = _env_int("TEST_KNOB_P1_BUNDLE", 5, validator=lambda v: 1 <= v <= 100)
    assert val == 5, "Validator debió rechazar 0 y caer al default 5."

    # Out-of-range alto:
    monkeypatch.setenv("TEST_KNOB_P1_BUNDLE", "100000")
    val = _env_int("TEST_KNOB_P1_BUNDLE", 5, validator=lambda v: 1 <= v <= 100)
    assert val == 5, "Validator debió rechazar 100000 y caer al default 5."

    # In-range:
    monkeypatch.setenv("TEST_KNOB_P1_BUNDLE", "42")
    val = _env_int("TEST_KNOB_P1_BUNDLE", 5, validator=lambda v: 1 <= v <= 100)
    assert val == 42
