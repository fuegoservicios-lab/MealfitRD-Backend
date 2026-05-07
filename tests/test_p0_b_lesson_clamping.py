"""[P0-B] Tests para clamping de rangos en `_validate_lesson_schema`.

Cubre:
  1. Valores en rango pasan sin modificar.
  2. Valores fuera de rango se clampean en sitio (mutación) y devuelven valid=True.
  3. NaN / inf / no-numérico / bool siguen rechazándose con valid=False.
  4. Percentages: [0, 100]. Counters: [0, 10000].
  5. El counter `_chunk_lesson_clamp_count` se incrementa por clamp y por campo.

Ejecutar:
    cd backend && python -m pytest tests/test_p0_b_lesson_clamping.py -v
"""
import sys
import os
import math
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


# Stubs mínimos para importar cron_tasks de forma aislada.
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


import pytest
import cron_tasks


def _valid_lesson(**overrides):
    """Construye una lesson válida y permite override de campos individuales."""
    base = {
        "repeat_pct": 25.0,
        "ingredient_base_repeat_pct": 30.0,
        "rejection_violations": 1,
        "allergy_violations": 0,
        "fatigued_violations": 2,
        "repeated_bases": ["pollo"],
        "repeated_meal_names": ["Pollo arroz"],
        "rejected_meals_that_reappeared": [],
        "allergy_hits": [],
        "chunk": 1,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _reset_clamp_counter():
    """Cada test arranca con counter limpio para aserciones determinísticas."""
    cron_tasks._chunk_lesson_clamp_count["total"] = 0
    cron_tasks._chunk_lesson_clamp_count["by_field"] = {}
    yield
    cron_tasks._chunk_lesson_clamp_count["total"] = 0
    cron_tasks._chunk_lesson_clamp_count["by_field"] = {}


# ---------------------------------------------------------------------------
# 1. Valores en rango: pasan sin modificar
# ---------------------------------------------------------------------------
def test_valid_lesson_passes_unchanged():
    lesson = _valid_lesson()
    snapshot = dict(lesson)
    ok, reason = cron_tasks._validate_lesson_schema(lesson)
    assert ok is True
    assert reason is None
    assert lesson == snapshot, "no debe mutar valores ya válidos"
    assert cron_tasks._chunk_lesson_clamp_count["total"] == 0


def test_boundary_values_pass_unchanged():
    """Valores justo en los bordes (0, 100, 10000) NO deben clampearse."""
    lesson = _valid_lesson(
        repeat_pct=0.0,
        ingredient_base_repeat_pct=100.0,
        rejection_violations=10000,
        allergy_violations=0,
        fatigued_violations=10000,
    )
    ok, _ = cron_tasks._validate_lesson_schema(lesson)
    assert ok is True
    assert lesson["repeat_pct"] == 0.0
    assert lesson["ingredient_base_repeat_pct"] == 100.0
    assert lesson["rejection_violations"] == 10000
    assert cron_tasks._chunk_lesson_clamp_count["total"] == 0


# ---------------------------------------------------------------------------
# 2. Clamp: valores fuera de rango pero finitos se clampean en sitio
# ---------------------------------------------------------------------------
def test_repeat_pct_above_100_is_clamped():
    lesson = _valid_lesson(repeat_pct=150.0)
    ok, reason = cron_tasks._validate_lesson_schema(lesson)
    assert ok is True
    assert reason is None
    assert lesson["repeat_pct"] == 100.0, "debe clampear 150 → 100"
    assert cron_tasks._chunk_lesson_clamp_count["total"] == 1
    assert cron_tasks._chunk_lesson_clamp_count["by_field"]["repeat_pct"] == 1


def test_repeat_pct_below_0_is_clamped():
    lesson = _valid_lesson(repeat_pct=-5.0)
    ok, _ = cron_tasks._validate_lesson_schema(lesson)
    assert ok is True
    assert lesson["repeat_pct"] == 0.0, "debe clampear -5 → 0"
    assert cron_tasks._chunk_lesson_clamp_count["total"] == 1


def test_ingredient_base_repeat_pct_above_100_is_clamped():
    lesson = _valid_lesson(ingredient_base_repeat_pct=200.0)
    ok, _ = cron_tasks._validate_lesson_schema(lesson)
    assert ok is True
    assert lesson["ingredient_base_repeat_pct"] == 100.0


def test_violation_counters_above_cap_are_clamped():
    """Counters absurdos (e.g. bug upstream multiplica por constante grande) → cap a 10000."""
    lesson = _valid_lesson(rejection_violations=999999, allergy_violations=50000)
    ok, _ = cron_tasks._validate_lesson_schema(lesson)
    assert ok is True
    assert lesson["rejection_violations"] == 10000
    assert lesson["allergy_violations"] == 10000
    assert cron_tasks._chunk_lesson_clamp_count["total"] == 2
    assert cron_tasks._chunk_lesson_clamp_count["by_field"]["rejection_violations"] == 1
    assert cron_tasks._chunk_lesson_clamp_count["by_field"]["allergy_violations"] == 1


def test_negative_violation_counters_are_clamped_to_zero():
    lesson = _valid_lesson(fatigued_violations=-3)
    ok, _ = cron_tasks._validate_lesson_schema(lesson)
    assert ok is True
    assert lesson["fatigued_violations"] == 0


def test_multiple_clamps_in_one_lesson():
    lesson = _valid_lesson(
        repeat_pct=120.0,
        ingredient_base_repeat_pct=-10.0,
        rejection_violations=20000,
    )
    ok, _ = cron_tasks._validate_lesson_schema(lesson)
    assert ok is True
    assert lesson["repeat_pct"] == 100.0
    assert lesson["ingredient_base_repeat_pct"] == 0.0
    assert lesson["rejection_violations"] == 10000
    assert cron_tasks._chunk_lesson_clamp_count["total"] == 3


# ---------------------------------------------------------------------------
# 3. NaN / inf / no-numérico / bool: siguen rechazándose
# ---------------------------------------------------------------------------
def test_nan_is_rejected_not_clamped():
    lesson = _valid_lesson(repeat_pct=float("nan"))
    ok, reason = cron_tasks._validate_lesson_schema(lesson)
    assert ok is False
    assert reason is not None and "non_finite_repeat_pct" in reason
    assert cron_tasks._chunk_lesson_clamp_count["total"] == 0


def test_inf_is_rejected_not_clamped():
    lesson = _valid_lesson(repeat_pct=float("inf"))
    ok, reason = cron_tasks._validate_lesson_schema(lesson)
    assert ok is False
    assert "non_finite" in reason
    lesson_neg = _valid_lesson(rejection_violations=float("-inf"))
    ok2, reason2 = cron_tasks._validate_lesson_schema(lesson_neg)
    assert ok2 is False
    assert "non_finite" in reason2


def test_string_numeric_is_rejected():
    lesson = _valid_lesson(repeat_pct="high")
    ok, reason = cron_tasks._validate_lesson_schema(lesson)
    assert ok is False
    assert "non_numeric" in reason


def test_bool_is_rejected():
    lesson = _valid_lesson(repeat_pct=True)
    ok, reason = cron_tasks._validate_lesson_schema(lesson)
    assert ok is False
    assert "bool_in_numeric" in reason


def test_non_dict_is_rejected():
    ok, reason = cron_tasks._validate_lesson_schema(["not", "a", "dict"])
    assert ok is False
    assert "not_a_dict" in reason


def test_non_list_field_is_rejected():
    lesson = _valid_lesson(repeated_bases="pollo, arroz")  # string, no list
    ok, reason = cron_tasks._validate_lesson_schema(lesson)
    assert ok is False
    assert "non_list_repeated_bases" in reason


# ---------------------------------------------------------------------------
# 4. Clamp counter incrementa de forma agregada
# ---------------------------------------------------------------------------
def test_clamp_counter_aggregates_across_calls():
    cron_tasks._validate_lesson_schema(_valid_lesson(repeat_pct=150.0))
    cron_tasks._validate_lesson_schema(_valid_lesson(repeat_pct=200.0))
    cron_tasks._validate_lesson_schema(_valid_lesson(rejection_violations=99999))
    assert cron_tasks._chunk_lesson_clamp_count["total"] == 3
    assert cron_tasks._chunk_lesson_clamp_count["by_field"]["repeat_pct"] == 2
    assert cron_tasks._chunk_lesson_clamp_count["by_field"]["rejection_violations"] == 1


# ---------------------------------------------------------------------------
# 5. Clamp con valor entero (no float) también funciona
# ---------------------------------------------------------------------------
def test_int_value_clamps_correctly():
    lesson = _valid_lesson(rejection_violations=int(15000))
    ok, _ = cron_tasks._validate_lesson_schema(lesson)
    assert ok is True
    assert lesson["rejection_violations"] == 10000
