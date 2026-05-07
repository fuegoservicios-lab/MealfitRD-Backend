"""[P1-B] Tests del fallback cuando el thread daemon de heartbeat no arranca.

Bug original: si `_heartbeat_thread.is_alive()` retornaba False tras `start()`
(límite de threads del proceso, OOM transient), el código solo loguaba ERROR y
continuaba al LLM call. El zombie rescue mataba el chunk tras CHUNK_LOCK_STALE_MINUTES,
perdiendo tokens y trabajo.

Fix: `_handle_heartbeat_start_failure` ejecuta abort seguro:
  1. Incrementa `_chunk_heartbeat_start_failures` (telemetría in-memory).
  2. Libera reservas (atómico, P1-A).
  3. UPDATE plan_chunk_queue: status='pending', attempts++, execute_after=NOW()+RETRY.
  4. DELETE chunk_user_locks.

Cubre:
  1. Counter se incrementa con timestamp y chunk_id.
  2. Llama a release_chunk_reservations con args correctos.
  3. UPDATE a plan_chunk_queue con SQL esperado y delay correcto.
  4. DELETE de chunk_user_locks por task_id.
  5. Resilencia: si release_chunk_reservations falla, sigue al UPDATE.
  6. Resilencia: si UPDATE falla, sigue al DELETE.
  7. Resilencia: si DELETE falla, no propaga.
  8. Múltiples invocaciones acumulan el counter.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_b_heartbeat_start_failure.py -v
"""
import sys
import os
import types
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


# Stubs estándar.
if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_kw: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_kw: None)
if "langchain_google_genai" not in sys.modules:
    _install_stub(
        "langchain_google_genai",
        GoogleGenerativeAIEmbeddings=object,
        ChatGoogleGenerativeAI=object,
    )
_install_stub(
    "db_core",
    execute_sql_query=lambda *_a, **_kw: None,
    execute_sql_write=lambda *_a, **_kw: None,
    connection_pool=None,
)
_install_stub(
    "db_inventory",
    deduct_consumed_meal_from_inventory=lambda *_a, **_kw: None,
    get_inventory_activity_since=lambda *_a, **_kw: [],
    get_raw_user_inventory=lambda *_a, **_kw: [],
    get_user_inventory_net=lambda *_a, **_kw: [],
    release_chunk_reservations=lambda *_a, **_kw: 0,
    # [test fix] db_inventory.reserve_plan_ingredients returns int (count of items reserved).
    # Stubbing with a dict caused TypeError: '>=' not supported between dict and int in
    # cron_tasks.py:5239/16519. Mirror production: count ingredients with len>=3 in days[2].
    reserve_plan_ingredients=lambda *_a, **_kw: sum(
        1 for d in (_a[2] if len(_a) >= 3 else (_kw.get("days") or []))
        for m in ((d or {}).get("meals") or [])
        for i in (m.get("ingredients") or [])
        if i and len(str(i).strip()) >= 3
    ),
)
_install_stub(
    "db",
    get_latest_meal_plan_with_id=lambda *_a, **_kw: None,
    get_user_likes=lambda *_a, **_kw: [],
    get_active_rejections=lambda *_a, **_kw: [],
    get_recent_plans=lambda *_a, **_kw: [],
)
_install_stub(
    "db_facts",
    get_all_user_facts=lambda *_a, **_kw: [],
    get_consumed_meals_since=lambda *_a, **_kw: [],
    get_user_facts_by_metadata=lambda *_a, **_kw: [],
)
_install_stub("pydantic", BaseModel=object, Field=lambda default=None, **_kw: default)
_install_stub("schemas", HealthProfileSchema=object, ExpandedRecipeModel=object)
_install_stub("graph_orchestrator", run_plan_pipeline=lambda *_a, **_kw: {})
_install_stub("memory_manager", build_memory_context=lambda *_a, **_kw: "")
_install_stub("services", _save_plan_and_track_background=lambda *_a, **_kw: None)
_install_stub("agent", analyze_preferences_agent=lambda *_a, **_kw: {})


def _stub_parse_quantity(text, *_a, **_kw):
    return (1.0, "ud", str(text or ""))


try:
    import shopping_calculator  # noqa: F401
except ImportError:
    _install_stub(
        "shopping_calculator",
        get_shopping_list_delta=lambda *_a, **_kw: [],
        _parse_quantity=_stub_parse_quantity,
    )
apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg


from unittest.mock import patch
import cron_tasks


def _reset_counter():
    """Resetea el counter in-memory entre tests para no contaminar."""
    cron_tasks._chunk_heartbeat_start_failures["count"] = 0
    cron_tasks._chunk_heartbeat_start_failures["last_failure_at"] = None
    cron_tasks._chunk_heartbeat_start_failures["last_chunk_id"] = None


# ---------------------------------------------------------------------------
# 1. Counter se incrementa con timestamp + chunk_id
# ---------------------------------------------------------------------------
def test_counter_increments_with_metadata():
    _reset_counter()
    before_ts = datetime.now(timezone.utc)

    with patch("cron_tasks.release_chunk_reservations"), \
         patch("cron_tasks.execute_sql_write"):
        cron_tasks._handle_heartbeat_start_failure("task-aaa", "user-1")

    state = cron_tasks._chunk_heartbeat_start_failures
    assert state["count"] == 1
    assert state["last_chunk_id"] == "task-aaa"
    assert state["last_failure_at"] is not None
    assert state["last_failure_at"] >= before_ts
    _reset_counter()


# ---------------------------------------------------------------------------
# 2. Llama release_chunk_reservations con args correctos
# ---------------------------------------------------------------------------
def test_release_called_with_user_and_chunk_id():
    _reset_counter()
    release_calls = []

    def fake_release(user_id, chunk_id):
        release_calls.append((user_id, chunk_id))
        return 0

    with patch("cron_tasks.release_chunk_reservations", side_effect=fake_release), \
         patch("cron_tasks.execute_sql_write"):
        cron_tasks._handle_heartbeat_start_failure("task-bbb", "user-2")

    assert release_calls == [("user-2", "task-bbb")]
    _reset_counter()


# ---------------------------------------------------------------------------
# 3. UPDATE a plan_chunk_queue con SQL y delay correctos
# ---------------------------------------------------------------------------
def test_update_plan_chunk_queue_sql_and_delay():
    _reset_counter()
    writes = []

    def fake_write(query, params=None):
        writes.append((query, params))

    with patch("cron_tasks.release_chunk_reservations"), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        cron_tasks._handle_heartbeat_start_failure("task-ccc", "user-3")

    update_calls = [w for w in writes if "UPDATE plan_chunk_queue" in w[0]]
    assert len(update_calls) == 1
    update_sql, update_params = update_calls[0]
    # Verificar componentes clave del SQL.
    assert "status = 'pending'" in update_sql
    assert "attempts = COALESCE(attempts, 0) + 1" in update_sql
    assert "make_interval(mins => %s)" in update_sql
    # Params: (retry_minutes, task_id).
    assert update_params == (cron_tasks.CHUNK_HEARTBEAT_START_FAIL_RETRY_MINUTES, "task-ccc")
    _reset_counter()


# ---------------------------------------------------------------------------
# 4. DELETE chunk_user_locks por task_id
# ---------------------------------------------------------------------------
def test_delete_chunk_user_locks_by_task_id():
    _reset_counter()
    writes = []

    with patch("cron_tasks.release_chunk_reservations"), \
         patch("cron_tasks.execute_sql_write", side_effect=lambda q, p=None: writes.append((q, p))):
        cron_tasks._handle_heartbeat_start_failure("task-ddd", "user-4")

    delete_calls = [w for w in writes if "DELETE FROM chunk_user_locks" in w[0]]
    assert len(delete_calls) == 1
    delete_sql, delete_params = delete_calls[0]
    assert "locked_by_chunk_id = %s" in delete_sql
    assert delete_params == ("task-ddd",)
    _reset_counter()


# ---------------------------------------------------------------------------
# 5. Resilencia: release falla → sigue al UPDATE y DELETE
# ---------------------------------------------------------------------------
def test_release_failure_does_not_abort_defer_or_unlock():
    _reset_counter()
    writes = []

    def boom_release(*_a, **_kw):
        raise RuntimeError("transient db blip")

    with patch("cron_tasks.release_chunk_reservations", side_effect=boom_release), \
         patch("cron_tasks.execute_sql_write", side_effect=lambda q, p=None: writes.append((q, p))):
        cron_tasks._handle_heartbeat_start_failure("task-eee", "user-5")

    # El UPDATE y DELETE deben haber corrido pese al fallo de release.
    assert any("UPDATE plan_chunk_queue" in w[0] for w in writes)
    assert any("DELETE FROM chunk_user_locks" in w[0] for w in writes)
    _reset_counter()


# ---------------------------------------------------------------------------
# 6. Resilencia: UPDATE falla → sigue al DELETE
# ---------------------------------------------------------------------------
def test_update_failure_does_not_abort_unlock():
    _reset_counter()
    writes = []

    def fake_write(query, params=None):
        writes.append((query, params))
        if "UPDATE plan_chunk_queue" in query:
            raise RuntimeError("DB lock contention")

    with patch("cron_tasks.release_chunk_reservations"), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        # No debe propagar la excepción.
        cron_tasks._handle_heartbeat_start_failure("task-fff", "user-6")

    # El DELETE debe seguir intentándose pese a que el UPDATE falló.
    delete_calls = [w for w in writes if "DELETE FROM chunk_user_locks" in w[0]]
    assert len(delete_calls) == 1
    _reset_counter()


# ---------------------------------------------------------------------------
# 7. Resilencia: DELETE falla → no propaga
# ---------------------------------------------------------------------------
def test_delete_failure_does_not_propagate():
    _reset_counter()

    def fake_write(query, params=None):
        if "DELETE" in query:
            raise RuntimeError("permission denied")

    with patch("cron_tasks.release_chunk_reservations"), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        # Si propagara, el caller (process_chunk_task) crashearía y dejaría el chunk
        # en estado peor.
        cron_tasks._handle_heartbeat_start_failure("task-ggg", "user-7")
    _reset_counter()


# ---------------------------------------------------------------------------
# 8. Múltiples invocaciones acumulan el counter
# ---------------------------------------------------------------------------
def test_multiple_invocations_accumulate_counter():
    _reset_counter()

    with patch("cron_tasks.release_chunk_reservations"), \
         patch("cron_tasks.execute_sql_write"):
        cron_tasks._handle_heartbeat_start_failure("task-1", "user-A")
        cron_tasks._handle_heartbeat_start_failure("task-2", "user-B")
        cron_tasks._handle_heartbeat_start_failure("task-3", "user-C")

    assert cron_tasks._chunk_heartbeat_start_failures["count"] == 3
    # last_chunk_id apunta al último.
    assert cron_tasks._chunk_heartbeat_start_failures["last_chunk_id"] == "task-3"
    _reset_counter()


# ---------------------------------------------------------------------------
# 9. Shape test: el call site en process_chunk_task usa el helper, no inline
# ---------------------------------------------------------------------------
def test_process_chunk_task_uses_helper_after_alive_check():
    """Verifica que el call site post-`is_alive()` invoca el helper y retorna.

    Esto protege contra regresiones donde alguien re-introduce la lógica inline
    olvidándose de llamar al helper.
    """
    import os as _os
    src_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "cron_tasks.py",
    )
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Buscar el bloque post-is_alive en process_chunk_task.
    idx = source.find("if not _heartbeat_thread.is_alive():")
    assert idx > -1, "no se encontró el guard `if not _heartbeat_thread.is_alive():`"
    nearby = source[idx:idx + 600]
    assert "_handle_heartbeat_start_failure(task_id, user_id)" in nearby, (
        "el guard `if not _heartbeat_thread.is_alive()` debe invocar "
        "_handle_heartbeat_start_failure y luego return — sin esto el chunk "
        "continuaría al LLM call sin protección."
    )
    # Después de la llamada, debe haber un return inmediato.
    assert "return" in nearby[nearby.find("_handle_heartbeat_start_failure"):], (
        "tras invocar el helper, el flujo debe abortar con `return` para no caer al LLM call."
    )
