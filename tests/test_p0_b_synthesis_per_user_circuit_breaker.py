"""[P0-B] Per-user circuit breaker para chunks que caen en síntesis low-confidence.

El alert agregado `_alert_high_synthesized_lesson_ratio` (cron_tasks.py:9642)
detecta degradación system-wide pero NO actúa: solo registra una row en
`system_alerts` cada `CHUNK_LESSON_SYNTH_ALERT_COOLDOWN_HOURS` (default 24h).
Mientras tanto los chunks afectados siguen generándose con learning low-confidence,
y el chunk N+1 hereda la degradación del N — la promesa del aprendizaje continuo
se rompe silenciosamente para el usuario.

Este test suite cubre:
  1. `_per_user_synthesis_ratio_exceeded`:
     - Devuelve dict con synth/total/ratio/exceeded
     - Respeta MIN_SAMPLES (no exceeded con n=2 aunque ratio sea alto)
     - Respeta THRESHOLD (no exceeded con ratio bajo)
     - Fail-open en caso de DB error (no bloquear al usuario por blip)
  2. `_pause_chunk_for_synthesis_overload`:
     - UPDATE plan_chunk_queue.status='pending_user_action' con metadata
     - INSERT system_alerts con alert_key específico al chunk + affected_user_ids
     - Push notification dispatched
     - Cooldown: si ya está pausado, NO re-pausar (return False)
     - Returns True cuando pausa exitosa
"""

import sys
import os
import types
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


# Stubs mínimos para que `import cron_tasks` no rompa cuando se corre aislado.
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
    release_chunk_reservations=lambda *_a, **_kw: None,
    reserve_plan_ingredients=lambda *_a, **_kw: 0,
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
try:
    import shopping_calculator  # noqa: F401
except ImportError:
    _install_stub(
        "shopping_calculator",
        get_shopping_list_delta=lambda *_a, **_kw: [],
        _parse_quantity=lambda *_a, **_kw: (1.0, "ud", str(_a[0] if _a else "")),
    )
apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg


from unittest.mock import patch, MagicMock
import cron_tasks


# ============================================================================
# Helper: query routers
# ============================================================================
def _query_router(*, telemetry_synth=0, queue_total=0, existing_pause=None):
    """Fake `execute_sql_query` que responde según el query.

    `existing_pause` simula la fila pre-existente que detiene el cooldown.
    """
    def fake_query(query, params=None, fetch_one=False, fetch_all=False):
        q = (query or "").strip()
        if "FROM chunk_lesson_telemetry" in q and "FROM plan_chunk_queue" in q:
            return {"synth": telemetry_synth, "total": queue_total}
        if "FROM plan_chunk_queue" in q and "_pause_reason" in q:
            return existing_pause
        if "FROM system_alerts" in q:
            return None
        return None
    return fake_query


# ============================================================================
# 1. _per_user_synthesis_ratio_exceeded
# ============================================================================
def test_ratio_helper_returns_dict_shape():
    """Debe devolver dict con keys synth/total/ratio/exceeded."""
    fake = _query_router(telemetry_synth=2, queue_total=10)
    with patch("cron_tasks.execute_sql_query", side_effect=fake):
        res = cron_tasks._per_user_synthesis_ratio_exceeded("u-1")
    assert set(res.keys()) == {"synth", "total", "ratio", "exceeded"}
    assert res["synth"] == 2
    assert res["total"] == 10
    assert abs(res["ratio"] - 0.2) < 1e-9


def test_ratio_helper_does_not_exceed_below_min_samples():
    """3/3 = 100% pero total<MIN_SAMPLES(=4) → exceeded=False."""
    fake = _query_router(telemetry_synth=3, queue_total=3)
    with patch("cron_tasks.execute_sql_query", side_effect=fake):
        res = cron_tasks._per_user_synthesis_ratio_exceeded("u-low-samples")
    assert res["exceeded"] is False, (
        f"con n=3 < MIN_SAMPLES=4, no debe disparar aunque ratio=100%. Got: {res}"
    )


def test_ratio_helper_does_not_exceed_below_threshold():
    """1/10 = 10% < 30% threshold → exceeded=False."""
    fake = _query_router(telemetry_synth=1, queue_total=10)
    with patch("cron_tasks.execute_sql_query", side_effect=fake):
        res = cron_tasks._per_user_synthesis_ratio_exceeded("u-low-ratio")
    assert res["exceeded"] is False
    assert abs(res["ratio"] - 0.1) < 1e-9


def test_ratio_helper_exceeds_when_above_threshold_and_samples():
    """5/10 = 50% >= 30% AND n>=4 → exceeded=True."""
    fake = _query_router(telemetry_synth=5, queue_total=10)
    with patch("cron_tasks.execute_sql_query", side_effect=fake):
        res = cron_tasks._per_user_synthesis_ratio_exceeded("u-overloaded")
    assert res["exceeded"] is True
    assert res["synth"] == 5
    assert res["total"] == 10


def test_ratio_helper_exactly_at_threshold_triggers():
    """Default threshold=0.30; ratio exactamente 0.30 debe disparar (>=)."""
    fake = _query_router(telemetry_synth=3, queue_total=10)
    with patch("cron_tasks.execute_sql_query", side_effect=fake):
        res = cron_tasks._per_user_synthesis_ratio_exceeded("u-edge")
    assert res["exceeded"] is True


def test_ratio_helper_fails_open_on_db_error():
    """Si la query falla, devolvemos exceeded=False (preferir degradación a deadlock)."""
    def boom(*_a, **_kw):
        raise RuntimeError("DB down")

    with patch("cron_tasks.execute_sql_query", side_effect=boom):
        res = cron_tasks._per_user_synthesis_ratio_exceeded("u-db-blip")

    assert res["exceeded"] is False
    assert res["synth"] == 0
    assert res["total"] == 0
    assert res["ratio"] == 0.0


def test_ratio_helper_handles_zero_total():
    """Usuario sin chunks aún → ratio=0, exceeded=False (no division by zero)."""
    fake = _query_router(telemetry_synth=0, queue_total=0)
    with patch("cron_tasks.execute_sql_query", side_effect=fake):
        res = cron_tasks._per_user_synthesis_ratio_exceeded("u-fresh")
    assert res["ratio"] == 0.0
    assert res["exceeded"] is False


# ============================================================================
# 2. _pause_chunk_for_synthesis_overload
# ============================================================================
def _ratio_info(synth=5, total=10, ratio=0.5):
    return {"synth": synth, "total": total, "ratio": ratio, "exceeded": True}


def _capture_writes():
    """Returns (writes_list, fake_write_fn). writes appended as (query, params)."""
    writes = []
    def fake_write(query, params=None):
        writes.append((query, params))
        return None
    return writes, fake_write


def test_pause_helper_updates_plan_chunk_queue_status():
    """Pause exitosa → UPDATE plan_chunk_queue.status='pending_user_action'."""
    fake_query = _query_router(existing_pause=None)
    writes, fake_write = _capture_writes()

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema", lambda: None):
        result = cron_tasks._pause_chunk_for_synthesis_overload(
            task_id="task-1",
            snap={"form_data": {}, "totalDays": 7},
            user_id="u-1",
            meal_plan_id="plan-1",
            week_number=2,
            ratio_info=_ratio_info(),
            source="last_chunk_learning_synth",
        )

    assert result is True

    # Hay 1 UPDATE plan_chunk_queue con status pending_user_action
    pause_writes = [
        w for w in writes
        if "UPDATE plan_chunk_queue" in (w[0] or "")
        and "pending_user_action" in (w[0] or "")
    ]
    assert len(pause_writes) == 1, f"writes: {[w[0][:80] for w in writes]}"
    # Snapshot serializado contiene _pause_reason='synthesis_ratio_exceeded'
    snap_json = pause_writes[0][1][0]
    snap = json.loads(snap_json)
    assert snap["_pause_reason"] == "synthesis_ratio_exceeded"
    assert snap["_p0b_synth_count"] == 5
    assert snap["_p0b_total_count"] == 10
    assert snap["_p0b_source"] == "last_chunk_learning_synth"
    assert "_p0b_paused_at" in snap


def test_pause_helper_inserts_system_alert_with_affected_user():
    """Pause → INSERT system_alerts con affected_user_ids=[user_id]."""
    fake_query = _query_router(existing_pause=None)
    writes, fake_write = _capture_writes()

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema", lambda: None):
        cron_tasks._pause_chunk_for_synthesis_overload(
            task_id="task-1",
            snap={"form_data": {}},
            user_id="u-affected",
            meal_plan_id="plan-1",
            week_number=2,
            ratio_info=_ratio_info(synth=4, total=8, ratio=0.5),
            source="last_chunk_learning_synth",
        )

    alert_writes = [
        w for w in writes if "INSERT INTO system_alerts" in (w[0] or "")
    ]
    assert len(alert_writes) == 1
    params = alert_writes[0][1]
    # alert_key tiene el formato chunk_synthesis_overload:user:plan:week
    assert "chunk_synthesis_overload:u-affected:plan-1:2" == params[0]
    # affected_user_ids es JSON con [user_id]
    affected = json.loads(params[-1])
    assert affected == ["u-affected"]


def test_pause_helper_dispatches_push_notification():
    """Pause → arranca thread de push (firma user_id, title, body, url)."""
    fake_query = _query_router(existing_pause=None)
    _writes, fake_write = _capture_writes()

    push_calls = []

    def fake_push(**kwargs):
        push_calls.append(kwargs)

    # utils_push se importa lazy dentro del helper. Insertamos un stub al
    # vuelo para capturar la llamada. El helper arranca un Thread daemon=True
    # con target=fake_push y kwargs={user_id,...}; le damos un join breve.
    fake_utils_push = types.ModuleType("utils_push")
    fake_utils_push.send_push_notification = fake_push

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema", lambda: None), \
         patch.dict(sys.modules, {"utils_push": fake_utils_push}):
        cron_tasks._pause_chunk_for_synthesis_overload(
            task_id="task-1",
            snap={"form_data": {}},
            user_id="u-push",
            meal_plan_id="plan-1",
            week_number=2,
            ratio_info=_ratio_info(),
            source="last_chunk_learning_synth",
        )

    # Damos un instante a los daemon threads para correr.
    import threading, time
    deadline = time.time() + 1.0
    while time.time() < deadline and not push_calls:
        time.sleep(0.01)

    assert len(push_calls) == 1, f"esperaba push, push_calls={push_calls}"
    assert push_calls[0]["user_id"] == "u-push"
    assert "revisión" in push_calls[0]["title"].lower() or "revision" in push_calls[0]["title"].lower()


def test_pause_helper_cooldown_skips_repause():
    """Si ya hay una pausa P0-B activa dentro del cooldown → return False."""
    # Existing row en plan_chunk_queue con _pause_reason='synthesis_ratio_exceeded'.
    fake_query = _query_router(
        existing_pause={"updated_at": "2026-05-02T08:00:00+00:00"},
    )
    writes, fake_write = _capture_writes()

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema", lambda: None):
        result = cron_tasks._pause_chunk_for_synthesis_overload(
            task_id="task-already-paused",
            snap={"form_data": {}},
            user_id="u-1",
            meal_plan_id="plan-1",
            week_number=2,
            ratio_info=_ratio_info(),
            source="last_chunk_learning_synth",
        )

    assert result is False, "cooldown activo debe skip pausa"
    # No debe haber UPDATE plan_chunk_queue ni INSERT system_alerts
    assert not any("UPDATE plan_chunk_queue" in (w[0] or "") for w in writes)
    assert not any("INSERT INTO system_alerts" in (w[0] or "") for w in writes)


def test_pause_helper_returns_false_when_pause_update_fails():
    """Si el UPDATE plan_chunk_queue falla, return False (no deadlockear)."""
    fake_query = _query_router(existing_pause=None)
    write_calls = []

    def fake_write(query, params=None):
        write_calls.append(query)
        if "UPDATE plan_chunk_queue" in query and "pending_user_action" in query:
            raise RuntimeError("FK constraint")
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema", lambda: None):
        result = cron_tasks._pause_chunk_for_synthesis_overload(
            task_id="task-fail",
            snap={"form_data": {}},
            user_id="u-1",
            meal_plan_id="plan-1",
            week_number=2,
            ratio_info=_ratio_info(),
            source="last_chunk_learning_synth",
        )

    assert result is False, (
        "si el UPDATE falla, return False para que el caller continúe en "
        "lugar de deadlockear el chunk"
    )


def test_pause_helper_continues_when_alert_insert_fails():
    """Si INSERT system_alerts falla pero UPDATE pasó, return True igual.
    La pausa es la acción crítica; el alert es observabilidad best-effort."""
    fake_query = _query_router(existing_pause=None)
    writes = []

    def fake_write(query, params=None):
        writes.append(query)
        if "INSERT INTO system_alerts" in query:
            raise RuntimeError("alert table missing")
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema", lambda: None):
        result = cron_tasks._pause_chunk_for_synthesis_overload(
            task_id="task-1",
            snap={"form_data": {}},
            user_id="u-1",
            meal_plan_id="plan-1",
            week_number=2,
            ratio_info=_ratio_info(),
            source="last_chunk_learning_synth",
        )

    # UPDATE corrió primero y exitoso → return True
    assert result is True
    # UPDATE plan_chunk_queue está en writes
    assert any("UPDATE plan_chunk_queue" in q for q in writes)


def test_pause_helper_serializes_metadata_with_ratio():
    """system_alerts.metadata debe contener synth/total/ratio/source para SRE."""
    fake_query = _query_router(existing_pause=None)
    writes, fake_write = _capture_writes()

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema", lambda: None):
        cron_tasks._pause_chunk_for_synthesis_overload(
            task_id="task-1",
            snap={"form_data": {}},
            user_id="u-1",
            meal_plan_id="plan-1",
            week_number=2,
            ratio_info=_ratio_info(synth=7, total=10, ratio=0.7),
            source="recent_lessons_regen",
        )

    alert_writes = [
        w for w in writes if "INSERT INTO system_alerts" in (w[0] or "")
    ]
    assert len(alert_writes) == 1
    metadata_json = alert_writes[0][1][3]
    metadata = json.loads(metadata_json)
    assert metadata["synth"] == 7
    assert metadata["total"] == 10
    assert metadata["ratio"] == 0.7
    assert metadata["source"] == "recent_lessons_regen"
    assert metadata["task_id"] == "task-1"
