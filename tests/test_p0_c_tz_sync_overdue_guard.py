"""[P0-C] Tests del guard de overdue en `_sync_chunk_queue_tz_offsets`.

Bug original: la fórmula `execute_after = execute_after + make_interval(mins => delta)`
preservaba la intención original (disparar a la misma hora local que se planeó), pero si
`execute_after` ya estaba vencido y `delta_minutes > 0` (TZ al este), el shift movía el
disparo MÁS al futuro en vez de dispararlo ya. Ejemplo: usuario en CDT (-300) actualiza
perfil a EDT (-240), su chunk vencido hace 2h se desplazaba +60min hacia adelante en vez
de disparar inmediatamente.

Fix: cuando `execute_after <= NOW()` y `delta > 0`, forzar `execute_after = NOW()` (cap
de overdue).

Casos cubiertos:
  1. Vencido + delta positivo  → `execute_after = NOW()` (FIX).
  2. Vencido + delta negativo  → shift normal (no degrada el caso ya OK).
  3. Futuro  + delta positivo  → shift normal.
  4. Futuro  + delta negativo  → shift normal.
  5. execute_after sin tzinfo  → tratado como UTC.
  6. execute_after = None      → shift normal (no detectable como vencido).
  7. drift bajo threshold      → no UPDATE (sanity).

Ejecutar:
    cd backend && python -m pytest tests/test_p0_c_tz_sync_overdue_guard.py -v
"""
import sys
import os
import types
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


# Stubs mínimos (mismo patrón que test_p0_a_zombie_partial_finalize.py).
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
# Helpers
# ---------------------------------------------------------------------------
def _make_row(execute_after, snapshot_tz=-300, live_tz=-240, chunk_id="c1", user_id="u1"):
    """Construye una fila tal como la devolvería el SELECT base con drift detectable.

    Defaults: snapshot_tz=-300 (CDT), live_tz=-240 (EDT) → drift=60m, delta=+60m
    (TZ al este, hora local más temprana).
    """
    return {
        "id": chunk_id,
        "user_id": user_id,
        "execute_after": execute_after,
        "snapshot_tz": snapshot_tz,
        "live_tz": live_tz,
    }


def _run_with_rows(rows):
    """Ejecuta `_sync_chunk_queue_tz_offsets` mockeando query/write y devuelve writes."""
    writes = []

    def fake_query(*_a, **_kw):
        return rows

    def fake_write(query, params=None):
        writes.append((query, params))
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        updated = cron_tasks._sync_chunk_queue_tz_offsets()
    return updated, writes


# ---------------------------------------------------------------------------
# 1. Vencido + delta positivo → execute_after = NOW() (caso del bug)
# ---------------------------------------------------------------------------
def test_overdue_with_positive_delta_forces_now():
    """El chunk vencido con TZ shift al este debe disparar YA, no retrasarse más."""
    overdue = datetime.now(timezone.utc) - timedelta(hours=2)
    rows = [_make_row(execute_after=overdue, snapshot_tz=-300, live_tz=-240)]

    updated, writes = _run_with_rows(rows)

    assert updated == 1
    assert len(writes) == 1
    query, params = writes[0]
    # El UPDATE forzado NO usa make_interval; usa execute_after = NOW().
    assert "execute_after = NOW()" in query, (
        "El path overdue+delta>0 debe forzar execute_after=NOW(), no aplicar shift. "
        f"Query: {query[:300]}"
    )
    assert "make_interval" not in query.split("execute_after")[1].split(",")[0], (
        "El UPDATE forzado no debe llamar a make_interval sobre execute_after"
    )
    # Params: (live_tz, live_tz, chunk_id) — sin delta_minutes.
    assert params == (-240, -240, "c1")


# ---------------------------------------------------------------------------
# 2. Vencido + delta negativo → shift normal
# ---------------------------------------------------------------------------
def test_overdue_with_negative_delta_uses_normal_shift():
    """Vencido pero TZ al oeste no es el caso del bug — preservamos comportamiento."""
    overdue = datetime.now(timezone.utc) - timedelta(hours=2)
    rows = [_make_row(execute_after=overdue, snapshot_tz=-240, live_tz=-300)]
    # delta = live - snapshot = -300 - (-240) = -60m (al oeste)

    updated, writes = _run_with_rows(rows)

    assert updated == 1
    query, params = writes[0]
    assert "make_interval(mins => %s)" in query
    # Params: (live_tz, live_tz, delta_minutes, chunk_id).
    assert params == (-300, -300, -60, "c1")


# ---------------------------------------------------------------------------
# 3. Futuro + delta positivo → shift normal (caso ya correcto)
# ---------------------------------------------------------------------------
def test_future_with_positive_delta_uses_normal_shift():
    future = datetime.now(timezone.utc) + timedelta(hours=5)
    rows = [_make_row(execute_after=future, snapshot_tz=-300, live_tz=-240)]
    # delta = +60m

    updated, writes = _run_with_rows(rows)

    assert updated == 1
    query, params = writes[0]
    assert "make_interval(mins => %s)" in query
    assert params == (-240, -240, 60, "c1")


# ---------------------------------------------------------------------------
# 4. Futuro + delta negativo → shift normal
# ---------------------------------------------------------------------------
def test_future_with_negative_delta_uses_normal_shift():
    future = datetime.now(timezone.utc) + timedelta(hours=5)
    rows = [_make_row(execute_after=future, snapshot_tz=-240, live_tz=-300)]

    updated, writes = _run_with_rows(rows)

    assert updated == 1
    query, params = writes[0]
    assert "make_interval(mins => %s)" in query
    assert params == (-300, -300, -60, "c1")


# ---------------------------------------------------------------------------
# 5. execute_after sin tzinfo → tratado como UTC (no crashea)
# ---------------------------------------------------------------------------
def test_naive_datetime_treated_as_utc():
    """Si la DB devuelve datetime naive, lo tratamos como UTC (defensivo)."""
    naive_overdue = datetime.utcnow() - timedelta(hours=2)  # naive a propósito
    assert naive_overdue.tzinfo is None
    rows = [_make_row(execute_after=naive_overdue, snapshot_tz=-300, live_tz=-240)]

    updated, writes = _run_with_rows(rows)

    assert updated == 1
    query, _params = writes[0]
    # Vencido + delta>0 → debe activar el guard.
    assert "execute_after = NOW()" in query


# ---------------------------------------------------------------------------
# 6. execute_after = None → shift normal (no podemos detectar overdue)
# ---------------------------------------------------------------------------
def test_none_execute_after_falls_back_to_normal_shift():
    rows = [_make_row(execute_after=None, snapshot_tz=-300, live_tz=-240)]

    updated, writes = _run_with_rows(rows)

    assert updated == 1
    query, _params = writes[0]
    # Sin execute_after no podemos saber si está vencido; default es shift normal.
    assert "make_interval(mins => %s)" in query


# ---------------------------------------------------------------------------
# 7. Drift bajo threshold → no UPDATE (sanity, comportamiento preservado)
# ---------------------------------------------------------------------------
def test_drift_below_threshold_skips():
    overdue = datetime.now(timezone.utc) - timedelta(hours=2)
    # snapshot=-300, live=-299 → drift=1m < THRESHOLD (default 15m).
    rows = [_make_row(execute_after=overdue, snapshot_tz=-300, live_tz=-299)]

    updated, writes = _run_with_rows(rows)

    assert updated == 0
    assert writes == []


# ---------------------------------------------------------------------------
# 8. Mezcla: 2 chunks, uno overdue+positivo, otro futuro+positivo → cada uno su path
# ---------------------------------------------------------------------------
def test_mixed_rows_routed_to_correct_paths():
    overdue = datetime.now(timezone.utc) - timedelta(hours=2)
    future = datetime.now(timezone.utc) + timedelta(hours=5)
    rows = [
        _make_row(execute_after=overdue, snapshot_tz=-300, live_tz=-240, chunk_id="c-overdue"),
        _make_row(execute_after=future, snapshot_tz=-300, live_tz=-240, chunk_id="c-future"),
    ]

    updated, writes = _run_with_rows(rows)

    assert updated == 2
    assert len(writes) == 2

    overdue_write = next(w for w in writes if w[1][-1] == "c-overdue")
    future_write = next(w for w in writes if w[1][-1] == "c-future")

    assert "execute_after = NOW()" in overdue_write[0]
    assert "make_interval" in future_write[0]
