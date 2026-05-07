"""[P1-C] Tests del coordinador de horizons de pantry para usuarios multi-plan.

Bug original: si un usuario tiene plan 7d (horizon=48h) y plan 30d (horizon=168h)
activos simultáneamente, sus chunks refrescan snapshots a cadencias distintas. Cuando
ambos generan días contemporáneos (el chunk 2 del 7d y el chunk 1 del 30d caen en la
misma semana), pueden ver el inventario con 60h+ de diferencia — uno planifica con
stock que el otro ya consumió.

Fix: `_coordinate_user_horizons(rows)` detecta usuarios con >=2 meal_plans en el batch
y devuelve un dict mapping `user_id → min_horizon` para forzar refresh coordinado.
Usuarios con 1 solo plan NO aparecen en el dict (caller usa el helper individual).

Cubre:
  1. Usuario con 1 plan → no aparece en el coordinated dict.
  2. Usuario con 2+ planes → aparece con el horizon más corto.
  3. Mezcla de usuarios single-plan y multi-plan.
  4. Filas sin user_id o meal_plan_id se ignoran.
  5. Empate en horizons (todos los planes mismo tamaño): emite ese valor único.
  6. Lista vacía → dict vacío.
  7. Múltiples chunks del MISMO plan no cuentan como multi-plan (deduplicación por meal_plan_id).
  8. Integración: el filtro del cron usa coordinated_horizons cuando aplicable.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_c_pantry_horizon_coordination.py -v
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row(user_id, plan_id, total_days):
    return {"user_id": user_id, "meal_plan_id": plan_id, "total_days_requested": total_days}


# ---------------------------------------------------------------------------
# 1. Usuario con 1 plan → no aparece en el dict
# ---------------------------------------------------------------------------
def test_single_plan_user_not_coordinated():
    """Un solo plan = no hay nada que coordinar."""
    rows = [_row("user-A", "plan-1", 7)]
    coordinated = cron_tasks._coordinate_user_horizons(rows)
    assert "user-A" not in coordinated
    assert coordinated == {}


# ---------------------------------------------------------------------------
# 2. Usuario con 2 planes → emite el horizon más corto
# ---------------------------------------------------------------------------
def test_multi_plan_user_uses_minimum_horizon():
    """Plan 7d (horizon=48h) + plan 30d (horizon=168h) → emite 48h para coordinar."""
    rows = [
        _row("user-A", "plan-7d", 7),
        _row("user-A", "plan-30d", 30),
    ]
    coordinated = cron_tasks._coordinate_user_horizons(rows)
    assert coordinated.get("user-A") == 48


# ---------------------------------------------------------------------------
# 3. Mezcla de usuarios single-plan y multi-plan
# ---------------------------------------------------------------------------
def test_mixed_users_only_multi_plan_coordinated():
    rows = [
        _row("user-multi", "plan-7d", 7),
        _row("user-multi", "plan-30d", 30),
        _row("user-single", "plan-15d", 15),
    ]
    coordinated = cron_tasks._coordinate_user_horizons(rows)
    assert coordinated == {"user-multi": 48}, (
        "user-single (con 1 plan) NO debe estar en coordinated"
    )


# ---------------------------------------------------------------------------
# 4. Filas sin user_id o meal_plan_id se ignoran
# ---------------------------------------------------------------------------
def test_rows_with_missing_keys_ignored():
    rows = [
        _row("user-A", "plan-1", 7),
        {"user_id": None, "meal_plan_id": "plan-2", "total_days_requested": 30},
        {"user_id": "user-A", "meal_plan_id": None, "total_days_requested": 15},
        _row("user-A", "plan-3", 30),
    ]
    coordinated = cron_tasks._coordinate_user_horizons(rows)
    # user-A tiene 2 planes válidos (plan-1 y plan-3) → coordinar.
    assert coordinated.get("user-A") == 48


# ---------------------------------------------------------------------------
# 5. Empate (todos los planes mismo tamaño) → ese valor único
# ---------------------------------------------------------------------------
def test_tied_horizons_emits_single_value():
    rows = [
        _row("user-A", "plan-1", 7),
        _row("user-A", "plan-2", 7),
        _row("user-A", "plan-3", 15),
    ]
    coordinated = cron_tasks._coordinate_user_horizons(rows)
    # Los 3 planes tienen total_days <30 → todos horizon=48. Min=48.
    assert coordinated.get("user-A") == 48


# ---------------------------------------------------------------------------
# 6. Lista vacía → dict vacío
# ---------------------------------------------------------------------------
def test_empty_input_returns_empty_dict():
    assert cron_tasks._coordinate_user_horizons([]) == {}
    assert cron_tasks._coordinate_user_horizons(None) == {}


# ---------------------------------------------------------------------------
# 7. Múltiples chunks del MISMO plan no cuentan como multi-plan
# ---------------------------------------------------------------------------
def test_multiple_chunks_same_plan_not_coordinated():
    """Coordinación es por meal_plan_id distinto, no por chunk individual."""
    rows = [
        _row("user-A", "plan-1", 30),
        _row("user-A", "plan-1", 30),  # mismo plan_id, otro chunk
        _row("user-A", "plan-1", 30),
    ]
    coordinated = cron_tasks._coordinate_user_horizons(rows)
    # Solo 1 plan distinto → no coordinar.
    assert "user-A" not in coordinated


# ---------------------------------------------------------------------------
# 8. Tres planes de tamaños distintos: 7d (48h), 15d (48h), 30d (168h) → min=48
# ---------------------------------------------------------------------------
def test_three_plans_minimum_wins():
    rows = [
        _row("user-A", "plan-7d", 7),
        _row("user-A", "plan-15d", 15),
        _row("user-A", "plan-30d", 30),
    ]
    coordinated = cron_tasks._coordinate_user_horizons(rows)
    assert coordinated.get("user-A") == 48


# ---------------------------------------------------------------------------
# 9. Integración: el cron filtra usando coordinated_horizons
# ---------------------------------------------------------------------------
def test_cron_filter_uses_coordinated_horizon_for_multi_plan_user():
    """End-to-end: usuario con plan 7d y plan 30d, chunk del 30d a 60h → con horizon
    individual (168h) entraría al refresh, pero con coordinación (48h) se filtra fuera.
    """
    now_utc = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    # Candidates devueltos por el SELECT inicial (todos dentro del horizon máximo 168h):
    #   - chunk del plan 7d, ejecuta en 12h (dentro de horizon 48h y 168h).
    #   - chunk del plan 30d, ejecuta en 60h (FUERA de horizon 48h, dentro de 168h).
    candidates_initial = [
        {
            "task_id": "chunk-7d",
            "user_id": "user-A",
            "meal_plan_id": "plan-7d",
            "week_number": 1,
            "execute_after": now_utc + timedelta(hours=12),
            "captured_at": now_utc - timedelta(hours=20),
            "total_days_requested": 7,
        },
        {
            "task_id": "chunk-30d",
            "user_id": "user-A",
            "meal_plan_id": "plan-30d",
            "week_number": 2,
            "execute_after": now_utc + timedelta(hours=60),
            "captured_at": now_utc - timedelta(hours=20),
            "total_days_requested": 30,
        },
    ]
    persist_calls = []

    def fake_query(*_a, **_kw):
        return candidates_initial

    def fake_persist(*args, **_kw):
        persist_calls.append(args)

    def fake_inventory(_uid):
        return [{"name": "pollo", "qty": 1}]

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks._persist_fresh_pantry_to_chunks", side_effect=fake_persist), \
         patch("cron_tasks.get_user_inventory_net", side_effect=fake_inventory):
        cron_tasks._proactive_refresh_pending_pantry_snapshots(now_utc=now_utc)

    # Solo chunk-7d entra al refresh (el del 30d cae fuera del horizon coordinado de 48h).
    refreshed_task_ids = {args[0] for args in persist_calls}
    assert refreshed_task_ids == {"chunk-7d"}, (
        f"Con coordinación P1-C, chunk-30d (a 60h) debe quedar fuera del horizon "
        f"coordinado de 48h aunque su horizon individual (168h) lo permitiría. "
        f"Refreshed: {refreshed_task_ids}"
    )
