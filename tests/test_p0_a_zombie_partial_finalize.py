"""[P0-A/ZOMBIE-PARTIAL] Tests para `_finalize_zombie_partial_plans`.

Cubre:
  1. SELECT incluye los filtros de zombie (status partial, edad mínima, sin chunks vivos).
  2. Plan con days>0 y todos los chunks terminados → UPDATE a 'complete_partial'.
  3. Plan con days==0 y todos los chunks terminados → UPDATE a 'failed'.
  4. Ningún candidato → no UPDATE, retorna 0.
  5. Multiple candidatos → un UPDATE por cada uno.
  6. Error en SELECT → retorna 0 sin crash.
  7. Error en UPDATE de un plan no aborta los demás.

Ejecutar:
    cd backend && python -m pytest tests/test_p0_a_zombie_partial_finalize.py -v
"""
import sys
import os
import types

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


# ---------------------------------------------------------------------------
# 1. SELECT incluye los filtros esperados
# ---------------------------------------------------------------------------
def test_select_query_filters_zombie_conditions():
    """El SELECT debe filtrar status partial, edad mínima, y excluir chunks vivos."""
    captured = {}

    def fake_query(query, params=None, fetch_one=False, fetch_all=False):
        captured["query"] = query
        captured["params"] = params
        return []

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write"):
        cron_tasks._finalize_zombie_partial_plans()

    q = captured["query"]
    # Filtro de status zombie.
    assert "'partial'" in q and "'generating_next'" in q
    # Filtro de edad mínima.
    assert "make_interval(hours" in q
    # Anti-join contra chunks vivos.
    assert "NOT EXISTS" in q
    assert "plan_chunk_queue" in q
    # Estados que aún pueden commitear.
    for live_status in ("pending", "processing", "stale", "pending_user_action"):
        assert f"'{live_status}'" in q
    # Failed sin dead-letter sigue siendo recuperable.
    assert "dead_lettered_at IS NULL" in q
    # Params deben ser (min_age_hours, batch_limit) en ese orden.
    params = captured["params"]
    assert params is not None
    assert params[0] == cron_tasks.CHUNK_ZOMBIE_PARTIAL_MIN_AGE_HOURS
    assert params[1] == cron_tasks.CHUNK_ZOMBIE_PARTIAL_BATCH_LIMIT


# ---------------------------------------------------------------------------
# 2. Plan con días materializados → complete_partial
# ---------------------------------------------------------------------------
def test_plan_with_days_marked_complete_partial():
    """Plan con days_count>0 → status='complete_partial', reason='partial_data'."""
    candidate = {
        "plan_id": "plan-aaa",
        "user_id": "user-1",
        "days_count": 7,
        "total_days_requested": 15,
    }
    writes = []

    def fake_query(query, params=None, fetch_one=False, fetch_all=False):
        return [candidate]

    def fake_write(query, params=None):
        writes.append((query, params))
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        finalized = cron_tasks._finalize_zombie_partial_plans()

    assert finalized == 1
    assert len(writes) == 1
    update_query, update_params = writes[0]
    assert "UPDATE meal_plans" in update_query
    assert "generation_status" in update_query
    assert "_partial_finalized_at" in update_query
    assert "_partial_finalized_reason" in update_query
    # Guard idempotente: solo actualiza si sigue en partial/generating_next.
    assert "'partial'" in update_query and "'generating_next'" in update_query
    # Params: (new_status, reason, plan_id).
    assert update_params[0] == "complete_partial"
    assert update_params[1] == "all_chunks_terminated_partial_data"
    assert update_params[2] == "plan-aaa"


# ---------------------------------------------------------------------------
# 3. Plan sin días materializados → failed
# ---------------------------------------------------------------------------
def test_plan_with_zero_days_marked_failed():
    """Plan con days_count==0 → status='failed', reason='no_days'."""
    candidate = {
        "plan_id": "plan-bbb",
        "user_id": "user-2",
        "days_count": 0,
        "total_days_requested": 7,
    }
    writes = []

    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write", side_effect=lambda q, p=None, **kw: writes.append((q, p))):
        finalized = cron_tasks._finalize_zombie_partial_plans()

    assert finalized == 1
    _q, params = writes[0]
    assert params[0] == "failed"
    assert params[1] == "all_chunks_terminated_no_days"
    assert params[2] == "plan-bbb"


# ---------------------------------------------------------------------------
# 4. Sin candidatos → no UPDATE
# ---------------------------------------------------------------------------
def test_no_candidates_short_circuits():
    writes = []
    with patch("cron_tasks.execute_sql_query", return_value=[]), \
         patch("cron_tasks.execute_sql_write", side_effect=lambda *a, **k: writes.append(a)):
        finalized = cron_tasks._finalize_zombie_partial_plans()

    assert finalized == 0
    assert writes == []


# ---------------------------------------------------------------------------
# 5. Múltiples candidatos → un UPDATE por cada uno
# ---------------------------------------------------------------------------
def test_multiple_candidates_each_updated():
    candidates = [
        {"plan_id": "p1", "user_id": "u1", "days_count": 3, "total_days_requested": 7},
        {"plan_id": "p2", "user_id": "u2", "days_count": 0, "total_days_requested": 7},
        {"plan_id": "p3", "user_id": "u3", "days_count": 14, "total_days_requested": 30},
    ]
    writes = []

    with patch("cron_tasks.execute_sql_query", return_value=candidates), \
         patch("cron_tasks.execute_sql_write", side_effect=lambda q, p=None, **kw: writes.append((q, p))):
        finalized = cron_tasks._finalize_zombie_partial_plans()

    assert finalized == 3
    assert len(writes) == 3
    # Status correcto por caso.
    statuses = [w[1][0] for w in writes]
    assert statuses == ["complete_partial", "failed", "complete_partial"]
    plan_ids = [w[1][2] for w in writes]
    assert plan_ids == ["p1", "p2", "p3"]


# ---------------------------------------------------------------------------
# 6. Error en SELECT → retorna 0 sin crash
# ---------------------------------------------------------------------------
def test_select_failure_returns_zero():
    def boom(*_a, **_kw):
        raise RuntimeError("DB unavailable")

    with patch("cron_tasks.execute_sql_query", side_effect=boom), \
         patch("cron_tasks.execute_sql_write"):
        finalized = cron_tasks._finalize_zombie_partial_plans()

    assert finalized == 0


# ---------------------------------------------------------------------------
# 7. Error en UPDATE de un plan no aborta los demás
# ---------------------------------------------------------------------------
def test_update_error_isolated_per_plan():
    candidates = [
        {"plan_id": "good-1", "user_id": "u1", "days_count": 5, "total_days_requested": 7},
        {"plan_id": "bad", "user_id": "u2", "days_count": 5, "total_days_requested": 7},
        {"plan_id": "good-2", "user_id": "u3", "days_count": 5, "total_days_requested": 7},
    ]
    writes_attempted = []

    def fake_write(query, params=None):
        writes_attempted.append(params[2] if params else None)
        if params and params[2] == "bad":
            raise RuntimeError("transient lock conflict")
        return None

    with patch("cron_tasks.execute_sql_query", return_value=candidates), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        finalized = cron_tasks._finalize_zombie_partial_plans()

    # Se intentaron los 3 UPDATEs (no se aborta tras el fallo).
    assert writes_attempted == ["good-1", "bad", "good-2"]
    # Solo 2 se finalizaron exitosamente (el "bad" lanzó).
    assert finalized == 2


# ---------------------------------------------------------------------------
# 8. Cron está registrado en el scheduler
# ---------------------------------------------------------------------------
def test_cron_registered_in_scheduler():
    """`register_plan_chunk_scheduler` debe añadir el job 'finalize_zombie_partial_plans'."""

    class FakeScheduler:
        def __init__(self):
            self.jobs = {}

        def get_job(self, job_id):
            return self.jobs.get(job_id)

        def add_job(self, func, trigger=None, *, id=None, **_kwargs):
            self.jobs[id] = {"func": func, "trigger": trigger}

    scheduler = FakeScheduler()
    # `CronTrigger` está stubeado como `object` en este test (no acepta args), pero el
    # cron `nightly_refresh_long_plan_snapshots` lo invoca con kwargs. Lo parcheamos
    # localmente con un dummy que sí acepta argumentos para que la registración llegue
    # a nuestro job.
    with patch("cron_tasks.CronTrigger", lambda *_a, **_kw: "cron_trigger_stub"):
        cron_tasks.register_plan_chunk_scheduler(scheduler)

    assert "finalize_zombie_partial_plans" in scheduler.jobs
    job = scheduler.jobs["finalize_zombie_partial_plans"]
    assert job["func"] is cron_tasks._finalize_zombie_partial_plans
    assert job["trigger"] == "interval"
