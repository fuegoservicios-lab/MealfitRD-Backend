"""[P0-1] Validación post-LLM contra nevera en el endpoint /api/plans/analyze.

Contexto:
    El endpoint síncrono `api_analyze` (routers/plans.py) genera el chunk 1
    del plan llamando `run_plan_pipeline`. Antes de cerrar P0-1 GAP-A, la
    validación post-LLM contra pantry sólo se invocaba cuando había un
    `actual_user_id`, dejando a los guests fuera del guardrail. Un guest
    podía mandar `current_pantry_ingredients` en el payload, pasar el guard
    de pantry mínima, y recibir un plan con ingredientes que su pantry no
    contiene — rompiendo la promesa "platos solo con alimentos de la nevera".

    Cambio cubierto por estos tests (routers/plans.py:846):
        - Antes: `if actual_user_id and _live_pantry:`
        - Después: `if _live_pantry:` (auth o guest da igual; sólo importa
          que haya pantry contra la cual validar).

Estos tests cubren:
    1. Helper acepta `user_id=None` y degrada igual cuando hay violación
       (regresión por si alguien re-introduce un check de user_id en el helper).
    2. El endpoint invoca `_validate_and_retry_initial_chunk_against_pantry`
       cuando un guest manda pantry en el payload (gap-A cerrado).
    3. El endpoint sigue invocando la validación para usuarios autenticados
       (guard-rail original sigue intacto).
    4. Si el helper retorna `degraded=True`, el endpoint sella
       `_initial_chunk_pantry_degraded=True` en el plan devuelto.

NOTA SOBRE MOCKING:
    El validador real `validate_ingredients_against_pantry` usa Vector Search
    semántico, no determinístico para tests aislados. Mockeamos el helper
    `_validate_and_retry_initial_chunk_against_pantry` completo para verificar
    que el endpoint lo invoca con los argumentos correctos según el branch
    (guest con pantry vs user auth). El comportamiento interno del helper
    está cubierto por `test_p0_5_pantry_per_meal_violation_marker.py`.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Stubs para módulos externos no disponibles en CI/test
# ---------------------------------------------------------------------------

def _install_stub(module_name, **attrs):
    # [P1-NEON-DB-MIGRATION · 2026-06-12] Try-real-first: si el módulo real es
    # importable en este entorno, usarlo y solo rellenar attrs faltantes. Antes
    # el stub sintético se instalaba siempre que el módulo no estuviera ya en
    # sys.modules — eso envenenaba runs aislados/cluster (e.g. el stub de
    # `db_inventory` ocultaba `_compute_dynamic_consumption_rates` a otros test
    # files del mismo proceso, y el stub de `graph_orchestrator` rompía el
    # `from graph_orchestrator import _env_int...` de cron_tasks). El fallback
    # sintético se conserva para CI minimal sin deps instaladas.
    if module_name not in sys.modules:
        try:
            __import__(module_name)
        except Exception:
            pass
    if module_name in sys.modules:
        existing = sys.modules[module_name]
        for key, value in attrs.items():
            if not hasattr(existing, key):
                setattr(existing, key, value)
        return existing
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


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
    restock_inventory=lambda *_a, **_kw: None,
    consume_inventory_items_completely=lambda *_a, **_kw: None,
)
_install_stub(
    "db",
    supabase=None,
    get_user_likes=lambda *_a, **_kw: [],
    get_active_rejections=lambda *_a, **_kw: [],
    get_or_create_session=lambda *_a, **_kw: None,
    save_message=lambda *_a, **_kw: None,
    update_user_health_profile=lambda *_a, **_kw: None,
    log_api_usage=lambda *_a, **_kw: None,
    get_latest_meal_plan=lambda *_a, **_kw: None,
    get_latest_meal_plan_with_id=lambda *_a, **_kw: None,
    get_recent_plans=lambda *_a, **_kw: [],
    update_meal_plan_data=lambda *_a, **_kw: None,
    insert_like=lambda *_a, **_kw: None,
)
_install_stub(
    "db_facts",
    get_all_user_facts=lambda *_a, **_kw: [],
    get_consumed_meals_since=lambda *_a, **_kw: [],
    get_user_facts_by_metadata=lambda *_a, **_kw: [],
)
_install_stub("pydantic", BaseModel=object, Field=lambda default=None, **_kw: default)
_install_stub(
    "schemas",
    HealthProfileSchema=object,
    ExpandedRecipeModel=object,
    # [P1-11] Stub del schema de eventos SSE — el router lo importa para
    # filtrar eventos públicos del progress_callback.
    PUBLIC_SSE_EVENTS=frozenset({
        "phase", "day_started", "day_complete", "day_completed",
        "complete", "error", "heartbeat",
    }),
)
_install_stub(
    "graph_orchestrator",
    run_plan_pipeline=lambda *_a, **_kw: {},
    arun_plan_pipeline=lambda *_a, **_kw: {},
    # Funciones helper que `routers/plans.py` importa además de los pipelines.
    # Stubs no-op: el test no ejerce lógica de strip/cap/merge, solo necesita
    # que el import del router resuelva.
    _strip_untrusted_internal_keys=lambda *_a, **_kw: [],
    _enforce_days_to_generate_cap=lambda *_a, **_kw: False,
    # [P1-FORM-6] Añadido como cuarto helper importado por el router.
    _merge_other_text_fields=lambda *_a, **_kw: 0,
    # [P1-NEON-DB-MIGRATION · 2026-06-12] cron_tasks importa los knob helpers
    # (`from graph_orchestrator import run_plan_pipeline, _env_int, _env_float,
    # _env_bool` — P1-A auto-registry). El fallback sintético debe exportarlos
    # o el import de cron_tasks revienta en CI sin el módulo real.
    _env_int=lambda _name, default=0, **_kw: default,
    _env_float=lambda _name, default=0.0, **_kw: default,
    _env_bool=lambda _name, default=False, **_kw: default,
)
_install_stub(
    "memory_manager",
    build_memory_context=lambda *_a, **_kw: {"recent_messages": [], "full_context_str": ""},
    summarize_and_prune=lambda *_a, **_kw: None,
)
_install_stub(
    "services",
    _save_plan_and_track_background=lambda *_a, **_kw: None,
    _process_swap_rejection_background=lambda *_a, **_kw: None,
    save_partial_plan_get_id=lambda *_a, **_kw: None,
)
_install_stub(
    "agent",
    analyze_preferences_agent=lambda *_a, **_kw: "",
    swap_meal=lambda *_a, **_kw: None,
)
_install_stub("ai_helpers", expand_recipe_agent=lambda *_a, **_kw: None)
_install_stub(
    "auth",
    get_verified_user_id=lambda *_a, **_kw: None,
    verify_api_quota=lambda *_a, **_kw: None,
)


def _stub_parse_quantity(text, *_a, **_kw):
    return (1.0, "ud", str(text or ""))


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


import pytest  # noqa: E402
from unittest.mock import patch, MagicMock  # noqa: E402

import constants  # noqa: E402
import cron_tasks  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# [P1-PANTRY-GUARD-INITIAL-SKIP · 2026-05-18] PANTRY_OK debe tener ≥10 items
# para activar el guard estricto (knob `PANTRY_GUARD_MIN_ITEMS` default 10).
# Pre-fix tenía 5 items pero el guard ahora skipea por debajo del threshold
# (asumiendo plan inicial / nevera no poblada). Para validar el path estricto
# necesitamos simular un usuario con ciclo de compras vivo.
PANTRY_OK = [
    "pollo 500g", "arroz 1000g", "tomate 300g", "cebolla 200g", "aceite 250ml",
    "huevo 12 unidades", "leche 1L", "queso 250g", "pan integral 500g", "yogurt 500g",
    "ajo 100g", "limón 200g",
]

# [P1-5] El endpoint valida campos mínimos del formulario antes de invocar al
# pipeline. Antes los tests mandaban payloads minimalistas (solo `user_id`,
# `totalDays`, `current_pantry_ingredients`, `tzOffset`) y el handler no
# rechazaba — ahora sí (correctamente). Los tests de pantry validation
# extienden con este preset para superar el guard biométrico y enfocarse
# en lo que cubren originalmente.
MIN_FORM_DATA = {
    "age": 30,
    "mainGoal": "lose_fat",
    "weight": 154,
    "height": 170,
    "gender": "male",
    "activityLevel": "moderate",
    # [P0-FORM-4] `weightUnit` añadido como required en `_REQUIRED_FORM_FIELDS`.
    "weightUnit": "lb",
    # [P0-FORM-1] `householdSize` y `groceryDuration` añadidos como required.
    "householdSize": 1,
    "groceryDuration": "weekly",
    # [P0-FORM-3] `motivation` ahora required (cableado al planner + day gen).
    "motivation": "Quiero recuperar mi energía y sentirme bien para mi familia.",
    # [P1-2] `allergies` y `medicalConditions` con enforcement nonempty
    # (sentinel "Ninguna" cuenta como answer válida, length=1).
    "allergies": ["Ninguna"],
    "medicalConditions": ["Ninguna"],
    # [P0-FORM-6] Sync con `REQUIRED_FORM_FIELDS` del frontend. Antes el
    # backend aceptaba estos 7 como ausentes y los defaultaba al consumirlos
    # (LLM con contexto vacío de timing/conducta) → plan degradado en clientes
    # que evitaran el wizard. Ahora `_validate_form_data_min` rechaza con 422.
    "scheduleType": "standard",
    "cookingTime": "30min",
    "budget": "medium",
    "sleepHours": "7-8 horas",
    "stressLevel": "Moderado",
    "dislikes": ["Ninguno"],
    "struggles": ["Ninguno"],
}


def _result_with_violation():
    return {
        "days": [
            {
                "day": 1,
                "meals": [
                    {"name": "OK", "ingredients": ["pollo", "arroz"]},
                    {"name": "BAD-salmon", "ingredients": ["salmón", "arroz"]},
                ],
            }
        ]
    }


def _result_clean():
    return {
        "days": [
            {
                "day": 1,
                "meals": [
                    {"name": "OK1", "ingredients": ["pollo", "arroz"]},
                    {"name": "OK2", "ingredients": ["tomate", "cebolla"]},
                ],
            }
        ]
    }


def _vip_factory(invalid: set):
    def _mock(generated, pantry, strict_quantities=True, tolerance=1.30):
        bad = [g for g in generated if g in invalid]
        if not bad:
            return True
        return f"INEXISTENTES: {', '.join(bad)}."
    return _mock


# ---------------------------------------------------------------------------
# Test 1 — Helper acepta user_id=None y degrada igual (regresión)
# ---------------------------------------------------------------------------

def test_helper_accepts_none_user_id_and_degrades_with_violation():
    """[P0-1 GAP-A regression] El helper debe tratar `user_id=None` (guest)
    idénticamente a un user autenticado: validar, reintentar, y devolver
    `degraded=True` cuando el plan viola pantry tras retries agotados.
    """
    pantry = ["pollo", "arroz"]
    initial = _result_with_violation()

    with patch(
        "constants.validate_ingredients_against_pantry",
        side_effect=_vip_factory({"salmón"}),
    ), patch("cron_tasks.run_plan_pipeline", side_effect=AssertionError("no debería llamarse")), \
         patch("cron_tasks.CHUNK_PANTRY_MAX_RETRIES", 0):
        final, audit = cron_tasks._validate_and_retry_initial_chunk_against_pantry(
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_context="",
            background_tasks=None,
            pantry_ingredients=pantry,
            initial_result=initial,
            user_id=None,  # <-- guest
        )

    assert audit.get("degraded") is True, (
        f"Helper debe degradar para guest cuando hay violación. audit={audit}"
    )
    assert audit.get("meals_marked_violated") == 1
    assert final["days"][0]["meals"][1].get("_pantry_violated") is True


def test_helper_accepts_none_user_id_and_passes_when_clean():
    """[P0-1 GAP-A regression] El helper debe validar OK con `user_id=None`
    cuando el plan no viola pantry."""
    pantry = ["pollo", "arroz", "tomate", "cebolla"]
    initial = _result_clean()

    with patch(
        "constants.validate_ingredients_against_pantry",
        side_effect=_vip_factory(set()),
    ), patch("cron_tasks.CHUNK_PANTRY_MAX_RETRIES", 0):
        _final, audit = cron_tasks._validate_and_retry_initial_chunk_against_pantry(
            pipeline_data={},
            history=[],
            taste_profile="",
            memory_context="",
            background_tasks=None,
            pantry_ingredients=pantry,
            initial_result=initial,
            user_id=None,
        )

    assert audit.get("validated_ok") is True
    assert audit.get("degraded") is False
    assert audit.get("meals_marked_violated") is None


# ---------------------------------------------------------------------------
# Test 2 — Endpoint: la validación se invoca para guest con pantry en payload
# ---------------------------------------------------------------------------

def _import_api_analyze():
    """Lazy-import del endpoint para que los stubs estén instalados primero."""
    from routers.plans import api_analyze
    return api_analyze


def _build_endpoint_call_kwargs(payload: dict, verified_user_id):
    """Empaqueta los argumentos del endpoint usando una BackgroundTasks real
    para no romper expectations internas."""
    from fastapi import BackgroundTasks, Response
    return {
        "background_tasks": BackgroundTasks(),
        "response": Response(),
        "data": payload,
        "verified_user_id": verified_user_id,
    }


def _common_endpoint_patches():
    """Conjunto mínimo de patches para que `api_analyze` corra sin tocar I/O.
    Devuelve una lista de context managers para usar con ExitStack o anidados.
    """
    return [
        patch("routers.plans.get_or_create_session", return_value=None),
        patch(
            "routers.plans.build_memory_context",
            return_value={"recent_messages": [], "full_context_str": ""},
        ),
        patch("routers.plans.get_user_likes", return_value=[]),
        patch("routers.plans.get_active_rejections", return_value=[]),
        patch("routers.plans.analyze_preferences_agent", return_value=""),
        patch("routers.plans._user_has_profile", return_value=False),
        patch("routers.plans.update_user_health_profile", return_value=None),
        patch("routers.plans.log_api_usage", return_value=None),
        patch("routers.plans._resolve_request_tz_offset", return_value=0),
        # Inventario live vacío → cae al payload (current_pantry_ingredients).
        patch("db_inventory.get_user_inventory_net", return_value=[]),
    ]


def test_endpoint_invokes_validation_for_guest_with_payload_pantry():
    """[P0-1 GAP-A] Un guest que envía `current_pantry_ingredients` en el
    payload DEBE disparar la validación post-LLM (antes era saltada por el
    `if actual_user_id and _live_pantry:`).
    """
    from contextlib import ExitStack

    api_analyze = _import_api_analyze()

    payload = {
        # Sin user_id → guest. session_id ausente para evitar memoria.
        "user_id": "guest",
        "totalDays": 3,
        "current_pantry_ingredients": PANTRY_OK,
        "tzOffset": 0,
        **MIN_FORM_DATA,
    }

    pipeline_return = _result_with_violation()

    helper_mock = MagicMock(
        return_value=(
            pipeline_return,
            {
                "validated_ok": False,
                "attempts": 1,
                "degraded": True,
                "last_violation": "INEXISTENTES: salmón.",
                "mode": "strict",
                "pantry_size": len(PANTRY_OK),
                "meals_marked_violated": 1,
            },
        )
    )

    with ExitStack() as stack:
        for cm in _common_endpoint_patches():
            stack.enter_context(cm)
        stack.enter_context(
            patch("routers.plans.run_plan_pipeline", return_value=pipeline_return)
        )
        stack.enter_context(
            patch("cron_tasks._validate_and_retry_initial_chunk_against_pantry", helper_mock)
        )

        kwargs = _build_endpoint_call_kwargs(payload, verified_user_id=None)
        resp = api_analyze(**kwargs)

    assert helper_mock.called, (
        "Gap-A roto: el helper de validación post-LLM NO se invocó para "
        "guest con pantry en el payload."
    )
    call_kwargs = helper_mock.call_args.kwargs
    assert call_kwargs["pantry_ingredients"] == PANTRY_OK
    assert call_kwargs["user_id"] is None, (
        f"Para guest, user_id debe propagarse como None al helper "
        f"(recibido: {call_kwargs['user_id']!r})"
    )

    # El endpoint debe sellar el flag de degradación en el plan devuelto.
    assert isinstance(resp, dict)
    plan_data = resp.get("plan_data") or resp.get("data") or resp
    # El endpoint puede envolver o no el plan según el branch; buscamos el flag
    # en cualquier nivel razonable.
    found_flag = (
        plan_data.get("_initial_chunk_pantry_degraded") is True
        or pipeline_return.get("_initial_chunk_pantry_degraded") is True
    )
    assert found_flag, (
        "El endpoint debe sellar `_initial_chunk_pantry_degraded=True` cuando "
        f"el helper retorna degraded=True. resp={resp}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Endpoint: la validación sigue invocándose para usuarios auth
# ---------------------------------------------------------------------------

def test_endpoint_invokes_validation_for_authenticated_user():
    """[P0-1 regression] El cambio de condición (`if _live_pantry:`) NO debe
    haber roto el path original de usuarios autenticados.
    """
    from contextlib import ExitStack

    api_analyze = _import_api_analyze()

    payload = {
        "user_id": "user-abc",
        "totalDays": 3,
        "current_pantry_ingredients": [],  # no se usa: get_user_inventory_net retorna pantry
        "tzOffset": 0,
        **MIN_FORM_DATA,
    }

    pipeline_return = _result_clean()

    helper_mock = MagicMock(
        return_value=(
            pipeline_return,
            {
                "validated_ok": True,
                "attempts": 1,
                "degraded": False,
                "last_violation": None,
                "mode": "strict",
                "pantry_size": len(PANTRY_OK),
            },
        )
    )

    with ExitStack() as stack:
        # Override get_user_inventory_net para retornar pantry para este user
        stack.enter_context(
            patch("db_inventory.get_user_inventory_net", return_value=PANTRY_OK)
        )
        for cm in [
            patch("routers.plans.get_or_create_session", return_value=None),
            patch(
                "routers.plans.build_memory_context",
                return_value={"recent_messages": [], "full_context_str": ""},
            ),
            patch("routers.plans.get_user_likes", return_value=[]),
            patch("routers.plans.get_active_rejections", return_value=[]),
            patch("routers.plans.analyze_preferences_agent", return_value=""),
            patch("routers.plans._user_has_profile", return_value=False),
            patch("routers.plans.update_user_health_profile", return_value=None),
            patch("routers.plans.log_api_usage", return_value=None),
            patch("routers.plans._resolve_request_tz_offset", return_value=0),
        ]:
            stack.enter_context(cm)
        stack.enter_context(
            patch("routers.plans.run_plan_pipeline", return_value=pipeline_return)
        )
        stack.enter_context(
            patch("cron_tasks._validate_and_retry_initial_chunk_against_pantry", helper_mock)
        )

        kwargs = _build_endpoint_call_kwargs(payload, verified_user_id="user-abc")
        resp = api_analyze(**kwargs)

    assert helper_mock.called, (
        "Regresión: el helper post-LLM no se invoca para users auth."
    )
    call_kwargs = helper_mock.call_args.kwargs
    assert call_kwargs["user_id"] == "user-abc"
    assert call_kwargs["pantry_ingredients"] == PANTRY_OK
    # Plan limpio: NO debe haber flag degraded.
    assert pipeline_return.get("_initial_chunk_pantry_degraded") is None
    assert isinstance(resp, dict)


# ---------------------------------------------------------------------------
# Test 4 — Endpoint: sin pantry NO se invoca el helper (corto-circuito sano)
# ---------------------------------------------------------------------------

def test_endpoint_skips_validation_when_no_pantry_available():
    """Si `_live_pantry` queda vacío (auth con inventario vacío que pasó el
    guard mínimo por payload, edge case), el endpoint no invoca el helper.
    El guard de pantry mínima (línea 767) ya rechaza la mayoría de estos
    casos con HTTPException 400, pero esta verificación cierra el contrato:
    sin pantry, no hay validación que aplicar.
    """
    from contextlib import ExitStack
    from fastapi import HTTPException

    api_analyze = _import_api_analyze()

    payload = {
        "user_id": "guest",
        "totalDays": 3,
        "current_pantry_ingredients": [],  # vacío → guard 400
        "tzOffset": 0,
        **MIN_FORM_DATA,
    }

    helper_mock = MagicMock()

    with ExitStack() as stack:
        for cm in _common_endpoint_patches():
            stack.enter_context(cm)
        stack.enter_context(
            patch("routers.plans.run_plan_pipeline", return_value=_result_clean())
        )
        stack.enter_context(
            patch("cron_tasks._validate_and_retry_initial_chunk_against_pantry", helper_mock)
        )

        kwargs = _build_endpoint_call_kwargs(payload, verified_user_id=None)
        with pytest.raises(HTTPException) as exc_info:
            api_analyze(**kwargs)

    # El endpoint envuelve cualquier excepción (incluido el HTTPException 400
    # del guard de pantry mínima) en un HTTPException 500 con el detail original
    # concatenado (routers/plans.py:1069). Verificamos que el rechazo sucedió
    # antes de la validación post-LLM (lo único que importa para este contrato).
    detail = str(exc_info.value.detail)
    assert "Actualiza tu nevera" in detail, (
        f"Esperaba mensaje del guard de pantry mínima en el detail, recibido: {detail}"
    )
    assert helper_mock.called is False, (
        "El helper no debe invocarse cuando el guard de pantry mínima "
        "rechaza la request antes."
    )
