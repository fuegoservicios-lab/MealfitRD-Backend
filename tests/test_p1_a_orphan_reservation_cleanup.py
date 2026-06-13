"""[P1-A] Tests del cron `_recover_orphan_chunk_reservations`.

Defensa en profundidad sobre `release_chunk_reservations` atómico (P1-A): si por
cualquier razón quedaron reservas asociadas a chunks ya terminados (legacy state
pre-fix, caller que olvidó invocar release, chunks borrados de plan_chunk_queue),
este cron las detecta y las libera.

Cubre:
  1. SELECT vacío → no llama nada.
  2. Chunks pending/processing → NO se limpian.
  3. Chunks completed/cancelled recientes (<MIN_TERMINAL_AGE) → NO se limpian (race-safe).
  4. Chunks completed/cancelled viejos → se limpian.
  5. Chunks que ya no existen en plan_chunk_queue → se limpian (huérfanos definitivos).
  6. Mezcla: solo se limpian los que aplican.
  7. Error en SELECT → retorna 0 sin crash.
  8. Error en release_chunk_reservations de un chunk → no aborta los demás.
  9. Cron registrado en el scheduler.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_a_orphan_reservation_cleanup.py -v
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


# Stubs base (mismo patrón que otros tests P0/P1).
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
# cron_tasks importa también `_env_int`/`_env_float`/`_env_bool` desde
# graph_orchestrator (auto-registro de knobs, P1-A · 2026-05-08) — el stub
# debe proveerlos o el `import cron_tasks` falla en colección standalone.
_install_stub(
    "graph_orchestrator",
    run_plan_pipeline=lambda *_a, **_kw: {},
    _env_int=lambda _name, default=0, *_a, **_kw: default,
    _env_float=lambda _name, default=0.0, *_a, **_kw: default,
    _env_bool=lambda _name, default=False, *_a, **_kw: default,
)
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
# CronTrigger debe ser instanciable con kwargs: `register_plan_chunk_scheduler`
# hace imports locales `from apscheduler.triggers.cron import CronTrigger as
# _CronTrigger` que bypassean el patch de `cron_tasks.CronTrigger`, y un stub
# `object` revienta con `object() takes no arguments`.
apscheduler_cron_pkg = _install_stub(
    "apscheduler.triggers.cron",
    CronTrigger=lambda *_a, **_kw: "cron_trigger_stub",
)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg


from unittest.mock import patch
import cron_tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_inv_row(row_id, user_id, chunks_meals: dict):
    """Construye una fila de user_inventory simulada.

    chunks_meals: {chunk_id: {meal_token: qty}} → genera reservation_details con keys
    `chunk:<chunk_id>:meal:<meal_token>`.
    """
    details = {}
    for chunk_id, meals in chunks_meals.items():
        for meal_token, qty in meals.items():
            details[f"chunk:{chunk_id}:meal:{meal_token}"] = qty
    return {
        "id": row_id,
        "user_id": user_id,
        "reservation_details": details,
    }


def _query_responder(inv_rows=None, plan_chunk_rows=None):
    """Devuelve un fake `execute_sql_query` que responde según el query."""
    def fake_query(query, params=None, fetch_one=False, fetch_all=False):
        q = (query or "").strip()
        if "FROM user_inventory" in q:
            return inv_rows
        if "FROM plan_chunk_queue" in q:
            return plan_chunk_rows
        return None
    return fake_query


# ---------------------------------------------------------------------------
# 1. SELECT vacío → no llama release
# ---------------------------------------------------------------------------
def test_no_inventory_rows_returns_zero():
    release_calls = []

    with patch("cron_tasks.execute_sql_query", side_effect=_query_responder(inv_rows=[])), \
         patch("db_inventory.release_chunk_reservations", side_effect=lambda *a, **k: release_calls.append(a)):
        released = cron_tasks._recover_orphan_chunk_reservations()

    assert released == 0
    assert release_calls == []


# ---------------------------------------------------------------------------
# 2. Chunks pending/processing → NO se limpian
# ---------------------------------------------------------------------------
def test_active_chunks_not_cleaned():
    inv_rows = [_make_inv_row("inv-1", "user-1", {"chunk-active": {"pollo": 5.0}})]
    plan_rows = [{"id": "chunk-active", "status": "processing", "is_old_enough": True}]
    release_calls = []

    with patch("cron_tasks.execute_sql_query", side_effect=_query_responder(inv_rows, plan_rows)), \
         patch("db_inventory.release_chunk_reservations", side_effect=lambda *a, **k: release_calls.append(a) or 0):
        released = cron_tasks._recover_orphan_chunk_reservations()

    assert released == 0
    assert release_calls == [], "chunk en estado activo no debe limpiarse"


# ---------------------------------------------------------------------------
# 3. Chunks terminados pero recientes → NO se limpian (race-safe)
# ---------------------------------------------------------------------------
def test_recent_terminal_chunks_not_cleaned():
    inv_rows = [_make_inv_row("inv-1", "user-1", {"chunk-recent": {"pollo": 5.0}})]
    plan_rows = [{"id": "chunk-recent", "status": "completed", "is_old_enough": False}]
    release_calls = []

    with patch("cron_tasks.execute_sql_query", side_effect=_query_responder(inv_rows, plan_rows)), \
         patch("db_inventory.release_chunk_reservations", side_effect=lambda *a, **k: release_calls.append(a) or 0):
        released = cron_tasks._recover_orphan_chunk_reservations()

    assert released == 0
    assert release_calls == [], "chunk terminal pero reciente no debe limpiarse (anti-race)"


# ---------------------------------------------------------------------------
# 4. Chunks terminados viejos → se limpian
# ---------------------------------------------------------------------------
def test_old_terminal_chunks_cleaned():
    inv_rows = [_make_inv_row("inv-1", "user-1", {"chunk-old": {"pollo": 5.0}})]
    plan_rows = [{"id": "chunk-old", "status": "completed", "is_old_enough": True}]
    release_calls = []

    def fake_release(user_id, chunk_id):
        release_calls.append((user_id, chunk_id))
        return 1  # 1 key liberada

    with patch("cron_tasks.execute_sql_query", side_effect=_query_responder(inv_rows, plan_rows)), \
         patch("db_inventory.release_chunk_reservations", side_effect=fake_release):
        released = cron_tasks._recover_orphan_chunk_reservations()

    assert released == 1
    assert release_calls == [("user-1", "chunk-old")]


# ---------------------------------------------------------------------------
# 5. Chunks que ya no existen en plan_chunk_queue → se limpian (huérfanos definitivos)
# ---------------------------------------------------------------------------
def test_nonexistent_chunks_cleaned():
    inv_rows = [_make_inv_row("inv-1", "user-1", {"chunk-deleted": {"pollo": 5.0}})]
    plan_rows = []  # chunk-deleted no existe en plan_chunk_queue
    release_calls = []

    def fake_release(user_id, chunk_id):
        release_calls.append((user_id, chunk_id))
        return 1

    with patch("cron_tasks.execute_sql_query", side_effect=_query_responder(inv_rows, plan_rows)), \
         patch("db_inventory.release_chunk_reservations", side_effect=fake_release):
        released = cron_tasks._recover_orphan_chunk_reservations()

    assert released == 1
    assert release_calls == [("user-1", "chunk-deleted")]


# ---------------------------------------------------------------------------
# 6. Mezcla: solo limpia los aplicables
# ---------------------------------------------------------------------------
def test_mixed_chunks_only_cleanable_processed():
    inv_rows = [
        _make_inv_row("inv-1", "user-1", {
            "chunk-old-completed": {"pollo": 3.0},
            "chunk-recent-completed": {"arroz": 2.0},
            "chunk-active": {"cebolla": 1.0},
        }),
        _make_inv_row("inv-2", "user-2", {"chunk-deleted": {"queso": 4.0}}),
    ]
    plan_rows = [
        {"id": "chunk-old-completed", "status": "completed", "is_old_enough": True},
        {"id": "chunk-recent-completed", "status": "completed", "is_old_enough": False},
        {"id": "chunk-active", "status": "processing", "is_old_enough": True},
        # chunk-deleted no aparece (DELETE)
    ]
    release_calls = []

    def fake_release(user_id, chunk_id):
        release_calls.append((user_id, chunk_id))
        return 1

    with patch("cron_tasks.execute_sql_query", side_effect=_query_responder(inv_rows, plan_rows)), \
         patch("db_inventory.release_chunk_reservations", side_effect=fake_release):
        released = cron_tasks._recover_orphan_chunk_reservations()

    cleaned_chunks = {c[1] for c in release_calls}
    # Solo chunk-old-completed (terminal viejo) y chunk-deleted (huérfano).
    assert cleaned_chunks == {"chunk-old-completed", "chunk-deleted"}
    assert released == 2  # 1 por cada chunk limpiado


# ---------------------------------------------------------------------------
# 7. Error en SELECT inicial → retorna 0
# ---------------------------------------------------------------------------
def test_select_failure_returns_zero():
    def boom(*_a, **_kw):
        raise RuntimeError("DB unavailable")

    with patch("cron_tasks.execute_sql_query", side_effect=boom):
        released = cron_tasks._recover_orphan_chunk_reservations()

    assert released == 0


# ---------------------------------------------------------------------------
# 8. Error en release_chunk_reservations de un chunk → no aborta los demás
# ---------------------------------------------------------------------------
def test_release_error_isolated_per_chunk():
    inv_rows = [
        _make_inv_row("inv-1", "user-A", {"chunk-good-1": {"pollo": 1.0}}),
        _make_inv_row("inv-2", "user-B", {"chunk-bad": {"arroz": 1.0}}),
        _make_inv_row("inv-3", "user-C", {"chunk-good-2": {"queso": 1.0}}),
    ]
    plan_rows = [
        {"id": "chunk-good-1", "status": "completed", "is_old_enough": True},
        {"id": "chunk-bad", "status": "completed", "is_old_enough": True},
        {"id": "chunk-good-2", "status": "completed", "is_old_enough": True},
    ]
    attempted = []

    def fake_release(user_id, chunk_id):
        attempted.append((user_id, chunk_id))
        if chunk_id == "chunk-bad":
            raise RuntimeError("transient lock")
        return 1

    with patch("cron_tasks.execute_sql_query", side_effect=_query_responder(inv_rows, plan_rows)), \
         patch("db_inventory.release_chunk_reservations", side_effect=fake_release):
        released = cron_tasks._recover_orphan_chunk_reservations()

    # Los 3 fueron intentados.
    attempted_chunks = {c[1] for c in attempted}
    assert attempted_chunks == {"chunk-good-1", "chunk-bad", "chunk-good-2"}
    # Solo 2 exitosos.
    assert released == 2


# ---------------------------------------------------------------------------
# 9. Reservation_details como JSON string (caso real Supabase)
# ---------------------------------------------------------------------------
def test_reservation_details_as_json_string_parsed():
    import json
    inv_rows = [{
        "id": "inv-1",
        "user_id": "user-1",
        "reservation_details": json.dumps({"chunk:chunk-old:meal:pollo": 3.0}),
    }]
    plan_rows = [{"id": "chunk-old", "status": "completed", "is_old_enough": True}]
    release_calls = []

    with patch("cron_tasks.execute_sql_query", side_effect=_query_responder(inv_rows, plan_rows)), \
         patch("db_inventory.release_chunk_reservations", side_effect=lambda *a, **k: release_calls.append(a) or 1):
        released = cron_tasks._recover_orphan_chunk_reservations()

    assert released == 1
    assert release_calls == [("user-1", "chunk-old")]


# ---------------------------------------------------------------------------
# 10. Cron está registrado en el scheduler
# ---------------------------------------------------------------------------
def test_cron_registered_in_scheduler():
    class FakeScheduler:
        def __init__(self):
            self.jobs = {}

        def get_job(self, job_id):
            return self.jobs.get(job_id)

        def add_job(self, func, trigger=None, *, id=None, **_kwargs):
            self.jobs[id] = {"func": func, "trigger": trigger}

    scheduler = FakeScheduler()
    with patch("cron_tasks.CronTrigger", lambda *_a, **_kw: "cron_trigger_stub"):
        cron_tasks.register_plan_chunk_scheduler(scheduler)

    assert "recover_orphan_chunk_reservations" in scheduler.jobs
    job = scheduler.jobs["recover_orphan_chunk_reservations"]
    # [P2-CRON-CORRELATION · 2026-05-28] `_add_job_jittered` envuelve cada
    # cron func con un scope de correlation_id vía functools.wraps — el
    # callable registrado puede ser el wrapper. `wraps` expone el original en
    # `__wrapped__`; la propiedad verificada sigue siendo "el job ejecuta
    # _recover_orphan_chunk_reservations".
    registered_fn = job["func"]
    unwrapped = getattr(registered_fn, "__wrapped__", registered_fn)
    assert unwrapped is cron_tasks._recover_orphan_chunk_reservations
    assert job["trigger"] == "interval"
