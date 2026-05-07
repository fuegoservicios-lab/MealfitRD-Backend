"""
Test para verificar que el bug de la validación de pantry que estaba fuera del for-loop
(código muerto) ha sido corregido.

Verifica:
  1. Ingredientes fantasma (no en inventario) → disparan reintentos del LLM
  2. Ingredientes válidos → LLM se invoca exactamente 1 vez, validación pasa en el primer intento
"""
import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Stubs para módulos externos no disponibles en CI/test
# ---------------------------------------------------------------------------

def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module

if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_args, **_kwargs: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_args, **_kwargs: None)
if "langchain_google_genai" not in sys.modules:
    # [P0-5] Match the same stub surface as test_chunked_learning_propagation:
    # `ChatGoogleGenerativeAI` is patched by tests like
    # `test_smart_shuffle_excludes_high_fatigue_days_using_learned_bases` to force
    # the LLM probe to fail (so `is_degraded=True`). When this file loaded first,
    # the stub registered without that attribute and the downstream patch raised
    # `AttributeError: <module 'langchain_google_genai'> does not have the
    # attribute 'ChatGoogleGenerativeAI'`. Including it here keeps the cross-file
    # stub idempotent regardless of collection order.
    _install_stub(
        "langchain_google_genai",
        GoogleGenerativeAIEmbeddings=object,
        ChatGoogleGenerativeAI=object,
    )

_install_stub("db_core", execute_sql_query=lambda *_args, **_kwargs: None, execute_sql_write=lambda *_args, **_kwargs: None, connection_pool=None)
# [test fix] db_inventory.reserve_plan_ingredients returns int (count of items reserved).
# Mirror production: count ingredients with len>=3 in days[2] of the call args.
_install_stub(
    "db_inventory",
    deduct_consumed_meal_from_inventory=lambda *_args, **_kwargs: None,
    get_inventory_activity_since=lambda *_args, **_kwargs: [],
    get_raw_user_inventory=lambda *_args, **_kwargs: [],
    get_user_inventory_net=lambda *_args, **_kwargs: [],
    release_chunk_reservations=lambda *_args, **_kwargs: None,
    reserve_plan_ingredients=lambda *_args, **_kwargs: sum(
        1 for d in (_args[2] if len(_args) >= 3 else (_kwargs.get("days") or []))
        for m in ((d or {}).get("meals") or [])
        for i in (m.get("ingredients") or [])
        if i and len(str(i).strip()) >= 3
    ),
)
_install_stub("db", get_latest_meal_plan_with_id=lambda *_args, **_kwargs: None, get_user_likes=lambda *_args, **_kwargs: [], get_active_rejections=lambda *_args, **_kwargs: [], get_recent_plans=lambda *_args, **_kwargs: [])
_install_stub("db_facts", get_all_user_facts=lambda *_args, **_kwargs: [], get_consumed_meals_since=lambda *_args, **_kwargs: [], get_user_facts_by_metadata=lambda *_args, **_kwargs: [])
_install_stub("pydantic", BaseModel=object, Field=lambda default=None, **_kwargs: default)
_install_stub("schemas", HealthProfileSchema=object, ExpandedRecipeModel=object)
_install_stub("graph_orchestrator", run_plan_pipeline=lambda *_args, **_kwargs: {})
_install_stub("memory_manager", build_memory_context=lambda *_args, **_kwargs: "")
_install_stub("services", _save_plan_and_track_background=lambda *_args, **_kwargs: None)
_install_stub("agent", analyze_preferences_agent=lambda *_args, **_kwargs: {})
# [P0-2] Stub must expose `_parse_quantity` because cron_tasks.py lazy-imports it
# (e.g. cron_tasks.py:5226 reservation reconciliation, :11333 Edge Recipe builder,
# :16495 chunk-completion ingredient counter). Without it the import raises
# ImportError("cannot import name '_parse_quantity' from 'shopping_calculator'
# (unknown location)") and the reconcile path logs [P0-5/RECONCILE] Error in
# every retry. Mirror the stub used by the other 11 test files in this dir.
def _stub_parse_quantity(text, *_a, **_kw):
    return (1.0, "ud", str(text or ""))
_install_stub(
    "shopping_calculator",
    get_shopping_list_delta=lambda *_args, **_kwargs: [],
    _parse_quantity=_stub_parse_quantity,
)
apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg

import constants  # noqa: E402 - must be after stubs
import cron_tasks  # noqa: E402
from unittest.mock import patch, MagicMock, create_autospec  # noqa: E402
from tests.test_chunked_learning_propagation import _run_process, _make_tasks  # noqa: E402


def _make_vip_mock(reject_keywords=None):
    """Create a validate_ingredients_against_pantry mock.

    If *reject_keywords* is given, any ingredient containing one of those
    keywords (case-insensitive) will be rejected.  Otherwise everything passes.
    """
    reject_keywords = [k.lower() for k in (reject_keywords or [])]

    def _vip(gen_ing, inv, strict_quantities=False, tolerance=1.0):
        if not strict_quantities and reject_keywords:
            for ing in gen_ing:
                for kw in reject_keywords:
                    if kw in ing.lower():
                        return f"INEXISTENTES en inventario: {ing}"
        return True

    return _vip


def _make_3day_pipeline_return(ingredients_per_day):
    """Build a pipeline return with 3 days starting at day 4."""
    days = []
    for i, ings in enumerate(ingredients_per_day, start=4):
        days.append({
            "day": i,
            "meals": [{"name": f"Comida {i}", "ingredients": ings}],
        })
    return {"days": days}


# -------------------------------------------------------------------------

def test_phantom_ingredient_triggers_retry():
    """Ingredientes fantasma deben hacer que el LLM se reinvoque > 1 vez."""
    tasks = _make_tasks(
        week_number=2, days_offset=3, days_count=3,
        extra_snapshot={"current_pantry_ingredients": ["pollo", "arroz", "tomate"]},
    )
    prior_plan = {"total_days_requested": 7, "days": []}

    pipeline_return = _make_3day_pipeline_return(
        [["ingrediente fantasma"], ["pollo"], ["arroz"]],
    )

    # Patch where the function is looked up: constants module
    vip_mock = MagicMock(side_effect=_make_vip_mock(reject_keywords=["fantasma"]))
    with patch.object(constants, "validate_ingredients_against_pantry", vip_mock):
        with patch("cron_tasks.get_raw_user_inventory", return_value=[{"item": "pollo", "quantity": 1000}]):
            with patch("cron_tasks.get_user_inventory_net", return_value=["pollo", "arroz", "tomate"]):
                _, mocks = _run_process(
                    tasks, prior_plan,
                    mock_pipeline_return=pipeline_return,
                    inventory=["pollo", "arroz", "tomate"],
                    user_profile={"_pantry_quantity_mode": "strict"},
                )

                # La validación debió ejecutarse >= 1 vez (ahora NO es código muerto)
                assert vip_mock.call_count >= 1, (
                    f"validate_ingredients_against_pantry no fue llamado (call_count={vip_mock.call_count}). "
                    "La validación probablemente sigue siendo código muerto."
                )

                # El pipeline debió reintentar > 1 vez porque fantasma dispara retry
                assert mocks["mock_pipeline"].call_count > 1, (
                    f"Pipeline se invocó {mocks['mock_pipeline'].call_count} vez(es), "
                    "pero debería haber reintentado por ingrediente fantasma."
                )


def test_valid_ingredients_only_invokes_llm_once():
    """Sin violaciones de despensa, el LLM debe invocarse exactamente 1 vez."""
    tasks = _make_tasks(
        week_number=2, days_offset=3, days_count=3,
        extra_snapshot={"current_pantry_ingredients": ["pollo", "arroz", "tomate"]},
    )
    prior_plan = {"total_days_requested": 7, "days": []}

    pipeline_return = _make_3day_pipeline_return(
        [["pollo", "arroz"], ["pollo"], ["arroz"]],
    )

    # Everything valid — no rejections
    vip_mock = MagicMock(side_effect=_make_vip_mock())
    with patch.object(constants, "validate_ingredients_against_pantry", vip_mock):
        with patch("cron_tasks.get_raw_user_inventory", return_value=[{"item": "pollo", "quantity": 1000}]):
            with patch("cron_tasks.get_user_inventory_net", return_value=["pollo", "arroz", "tomate"]):
                _, mocks = _run_process(
                    tasks, prior_plan,
                    mock_pipeline_return=pipeline_return,
                    inventory=["pollo", "arroz", "tomate"],
                    user_profile={"_pantry_quantity_mode": "strict"},
                )

                # La validación debió ejecutarse
                assert vip_mock.call_count >= 1, (
                    f"validate_ingredients_against_pantry no fue llamado (call_count={vip_mock.call_count}). "
                    "La validación probablemente sigue siendo código muerto."
                )

                # Sin violaciones → exactamente 1 invocación del pipeline
                assert mocks["mock_pipeline"].call_count == 1, (
                    f"Pipeline se invocó {mocks['mock_pipeline'].call_count} vez(es), "
                    "pero con ingredientes válidos debería ser exactamente 1."
                )
