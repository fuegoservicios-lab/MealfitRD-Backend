"""[P1-27] Tests para que el fallback DB del `LLMCircuitBreaker` sea atómico
contra concurrencia multi-worker.

Bug original (audit P1-27):
  Cuando Redis está caído, `LLMCircuitBreaker.record_failure` /
  `record_success` (y sus equivalentes async) caían a un patrón inseguro
  de read-modify-write contra `app_kv_store`:

    1. SELECT value FROM app_kv_store WHERE key = ?
    2. state['failures'] += 1; if >= threshold: state['is_open'] = True
    3. INSERT ... ON CONFLICT DO UPDATE SET value = (re-serialized state)

  `self._lock` (threading.Lock) protege la atomicidad SOLO dentro del
  mismo proceso. Bajo Gunicorn `--workers N`, cada worker tiene su
  propia instancia con su propio lock — dos workers que registran fallos
  concurrentes leen `failures=2` cada uno y escriben `failures=3`
  (lost-update). El threshold se cruza con DELAY proporcional al número
  de workers, dejando seguir requests contra un proveedor saturado.

  El docstring previo lo confesaba: "Fallback DB (best-effort, no
  atómico pero funcional)" — funcional en single-worker, ROTO en prod
  multi-worker.

Fix:
  Reemplazar el read-modify-write por SQL UPSERT atómica que hace el
  INCR del lado del servidor (Postgres serializa ON CONFLICT DO UPDATE
  a nivel de fila):

    INSERT INTO app_kv_store (key, value)
    VALUES (%s, jsonb_build_object('failures', 1, ...))
    ON CONFLICT (key) DO UPDATE SET
        value = jsonb_build_object(
            'failures',
                COALESCE((app_kv_store.value->>'failures')::int, 0) + 1,
            'last_failure', %s::float,
            'is_open',
                (COALESCE((app_kv_store.value->>'failures')::int, 0) + 1)
                    >= %s::int
        ),
        updated_at = NOW()

  Helpers nuevos:
    - `_atomic_record_failure_db` / `_aatomic_record_failure_db`
    - `_atomic_reset_db` / `_aatomic_reset_db`

  Reset también atomizado: UPSERT idempotente con value zero (sin
  SELECT previo).

Cobertura:
  - test_atomic_record_failure_helper_exists
  - test_atomic_reset_helper_exists
  - test_record_failure_uses_atomic_helper_when_redis_down
  - test_record_success_uses_atomic_helper_when_redis_down
  - test_arecord_failure_uses_atomic_helper_async
  - test_arecord_success_uses_atomic_helper_async
  - test_atomic_record_failure_sql_uses_jsonb_build_object_with_increment
  - test_atomic_record_failure_sql_passes_threshold
  - test_atomic_reset_sql_is_idempotent_zero_state
  - test_atomic_record_failure_sql_passes_db_key
  - test_documentation_p1_27_present
"""
import asyncio
import inspect
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

import graph_orchestrator
from graph_orchestrator import LLMCircuitBreaker


# ---------------------------------------------------------------------------
# 1. Helpers expuestos.
# ---------------------------------------------------------------------------
def test_atomic_record_failure_helper_exists():
    """`_atomic_record_failure_db` (sync) y `_aatomic_record_failure_db`
    (async) deben estar definidos en LLMCircuitBreaker."""
    cb = LLMCircuitBreaker(failure_threshold=3, reset_timeout=30)
    assert hasattr(cb, "_atomic_record_failure_db")
    assert callable(cb._atomic_record_failure_db)
    assert hasattr(cb, "_aatomic_record_failure_db")
    assert asyncio.iscoroutinefunction(cb._aatomic_record_failure_db)


def test_atomic_reset_helper_exists():
    """`_atomic_reset_db` (sync) y `_aatomic_reset_db` (async) deben existir."""
    cb = LLMCircuitBreaker(failure_threshold=3, reset_timeout=30)
    assert hasattr(cb, "_atomic_reset_db")
    assert callable(cb._atomic_reset_db)
    assert hasattr(cb, "_aatomic_reset_db")
    assert asyncio.iscoroutinefunction(cb._aatomic_reset_db)


# ---------------------------------------------------------------------------
# 2. Sync paths invocan los helpers atómicos cuando Redis no disponible.
# ---------------------------------------------------------------------------
def test_record_failure_uses_atomic_helper_when_redis_down():
    """Sin redis_client, `record_failure` debe invocar
    `_atomic_record_failure_db` (NO el patrón legacy SELECT+UPDATE)."""
    cb = LLMCircuitBreaker(failure_threshold=3, reset_timeout=30)
    with patch("graph_orchestrator.redis_client", None), \
         patch.object(cb, "_atomic_record_failure_db") as mock_atomic, \
         patch.object(cb, "_get_db_state") as mock_get, \
         patch.object(cb, "_save_db_state") as mock_save:
        cb.record_failure()

    mock_atomic.assert_called_once()
    # El patrón legacy NO debe usarse — verificamos que ni _get_db_state
    # ni _save_db_state se invocaron.
    mock_get.assert_not_called()
    mock_save.assert_not_called()


def test_record_success_uses_atomic_helper_when_redis_down():
    """Sin redis_client y con flag local unhealthy, `record_success` debe
    invocar `_atomic_reset_db`."""
    cb = LLMCircuitBreaker(failure_threshold=3, reset_timeout=30)
    # Forzar local_healthy=False para que el debounce no skip-ee la rama DB.
    cb._local_healthy = False
    cb._last_db_check = 0
    with patch("graph_orchestrator.redis_client", None), \
         patch.object(cb, "_atomic_reset_db") as mock_atomic, \
         patch.object(cb, "_get_db_state") as mock_get, \
         patch.object(cb, "_save_db_state") as mock_save:
        cb.record_success()

    mock_atomic.assert_called_once()
    mock_get.assert_not_called()
    mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Async paths invocan los helpers atómicos.
# ---------------------------------------------------------------------------
def test_arecord_failure_uses_atomic_helper_async():
    """`arecord_failure` debe invocar `_aatomic_record_failure_db` cuando
    redis_async_client no disponible."""
    cb = LLMCircuitBreaker(failure_threshold=3, reset_timeout=30)
    with patch("graph_orchestrator.redis_async_client", None), \
         patch.object(cb, "_aatomic_record_failure_db", new_callable=AsyncMock) as mock_atomic, \
         patch.object(cb, "_aget_db_state", new_callable=AsyncMock) as mock_get, \
         patch.object(cb, "_asave_db_state", new_callable=AsyncMock) as mock_save:
        asyncio.run(cb.arecord_failure())

    mock_atomic.assert_called_once()
    mock_get.assert_not_called()
    mock_save.assert_not_called()


def test_arecord_success_uses_atomic_helper_async():
    """`arecord_success` debe invocar `_aatomic_reset_db`."""
    cb = LLMCircuitBreaker(failure_threshold=3, reset_timeout=30)
    cb._local_healthy = False
    cb._last_db_check = 0
    with patch("graph_orchestrator.redis_async_client", None), \
         patch.object(cb, "_aatomic_reset_db", new_callable=AsyncMock) as mock_atomic, \
         patch.object(cb, "_aget_db_state", new_callable=AsyncMock) as mock_get, \
         patch.object(cb, "_asave_db_state", new_callable=AsyncMock) as mock_save:
        asyncio.run(cb.arecord_success())

    mock_atomic.assert_called_once()
    mock_get.assert_not_called()
    mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# 4. SQL emitida por los helpers es la atómica esperada.
# ---------------------------------------------------------------------------
def test_atomic_record_failure_sql_uses_jsonb_build_object_with_increment():
    """El SQL del helper debe usar `jsonb_build_object` y un INCR
    server-side (`COALESCE(... + 1`) en lugar de pasar un JSON
    pre-calculado en Python."""
    cb = LLMCircuitBreaker(failure_threshold=5, reset_timeout=30)
    captured = []

    def fake_write(sql, params=None, **kw):
        captured.append((sql, params))
        return True

    with patch("graph_orchestrator.execute_sql_write", side_effect=fake_write):
        cb._atomic_record_failure_db()

    assert len(captured) == 1
    sql, params = captured[0]
    assert "jsonb_build_object" in sql, (
        "P1-27: la SQL atómica debe usar jsonb_build_object para construir "
        "el value server-side."
    )
    assert "COALESCE" in sql and "+ 1" in sql, (
        "P1-27: el INCR debe ocurrir server-side via COALESCE(... + 1) — "
        "sin esto, el patrón sigue siendo no-atómico."
    )
    # Y NO debe haber un SELECT previo en la transacción del helper.
    assert "SELECT" not in sql.upper().split("INSERT")[0], (
        "P1-27: no debe haber SELECT previo al UPSERT en el mismo helper."
    )


def test_atomic_record_failure_sql_passes_threshold():
    """El threshold debe propagarse a la SQL como parámetro (NO hardcoded
    en Python ni inline en la SQL)."""
    cb = LLMCircuitBreaker(failure_threshold=7, reset_timeout=30)
    captured = []

    def fake_write(sql, params=None, **kw):
        captured.append((sql, params))
        return True

    with patch("graph_orchestrator.execute_sql_write", side_effect=fake_write):
        cb._atomic_record_failure_db()

    sql, params = captured[0]
    # El threshold (7) debe aparecer en los params.
    assert 7 in params, (
        f"P1-27: threshold=7 esperado en params, vio: {params}"
    )


def test_atomic_record_failure_sql_passes_db_key():
    """El alert key (`self._db_kv_key`) debe ser el primer parámetro."""
    cb = LLMCircuitBreaker(failure_threshold=3, reset_timeout=30,
                           model_name="custom-model")
    captured = []

    def fake_write(sql, params=None, **kw):
        captured.append((sql, params))
        return True

    with patch("graph_orchestrator.execute_sql_write", side_effect=fake_write):
        cb._atomic_record_failure_db()

    sql, params = captured[0]
    assert params[0] == "llm_circuit_breaker:custom-model", (
        f"P1-27: db_kv_key esperado como primer param, vio: {params[0]!r}"
    )


def test_atomic_reset_sql_is_idempotent_zero_state():
    """El reset debe escribir el JSON con failures=0/is_open=false sin
    leer el estado previo. Idempotente: dos llamadas seguidas dejan el
    mismo end-state."""
    cb = LLMCircuitBreaker(failure_threshold=3, reset_timeout=30)
    captured = []

    def fake_write(sql, params=None, **kw):
        captured.append((sql, params))
        return True

    with patch("graph_orchestrator.execute_sql_write", side_effect=fake_write):
        cb._atomic_reset_db()
        cb._atomic_reset_db()

    assert len(captured) == 2
    for sql, params in captured:
        assert '"failures": 0' in sql, (
            "P1-27: reset SQL debe contener `failures: 0` literal."
        )
        assert '"is_open": false' in sql, (
            "P1-27: reset SQL debe contener `is_open: false` literal."
        )
        # NO debe leer el state previo.
        assert "SELECT" not in sql.upper().split("INSERT")[0]


def test_atomic_record_failure_sql_includes_on_conflict_do_update():
    """La SQL debe usar `ON CONFLICT (key) DO UPDATE` — esa es la cláusula
    que Postgres serializa a nivel de fila, garantizando atomicidad
    cross-worker."""
    cb = LLMCircuitBreaker(failure_threshold=3, reset_timeout=30)
    captured = []

    with patch("graph_orchestrator.execute_sql_write",
               side_effect=lambda sql, params=None, **kw: captured.append((sql, params)) or True):
        cb._atomic_record_failure_db()

    sql = captured[0][0]
    assert "ON CONFLICT" in sql.upper()
    assert "DO UPDATE" in sql.upper()


# ---------------------------------------------------------------------------
# 5. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_27_present():
    """Comentario `[P1-27]` debe documentar el fix de atomicidad."""
    full_src = inspect.getsource(graph_orchestrator)
    assert "[P1-27]" in full_src


def test_documentation_mentions_lost_update_or_multi_worker():
    """El comentario debe explicar el rationale: lost-update bajo
    multi-worker. Sin esto un futuro lector podría reintroducir el
    patrón legacy pensando que el threading.Lock basta."""
    helper_src = inspect.getsource(LLMCircuitBreaker._atomic_record_failure_db)
    full_src = inspect.getsource(graph_orchestrator)
    p127_idx = full_src.find("[P1-27]")
    window = full_src[p127_idx : p127_idx + 3000]
    needles = ["lost-update", "lost update", "multi-worker", "multi worker",
               "atomicidad", "INCR server", "server-side", "race"]
    found = any(n in window.lower() for n in (n.lower() for n in needles))
    assert found, (
        f"P1-27: el comentario debe explicar lost-update / multi-worker / "
        f"atomicidad. Window: {window[:300]!r}"
    )
