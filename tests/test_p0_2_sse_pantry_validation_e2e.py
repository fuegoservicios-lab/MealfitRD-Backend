"""[P1-10] Tests E2E del endpoint `/api/plans/analyze/stream` usando TestClient.

Antes la única cobertura del path SSE era manual (smoke test del frontend).
Tests unitarios del endpoint sync (`/analyze`) en
`test_p0_1_endpoint_pantry_postvalidation.py` no protegen contra regresiones
en el SSE — el path PRIMARIO del frontend. Este archivo cierra esa brecha
ejerciendo el handler entero (validación form, rate limit, pipeline mock,
pantry validation, post-procesamiento, evento `complete`) vía TestClient.

Cubre los siguientes gaps cerrados en este sprint:

- **P0-2** (`/analyze/stream` ejecuta validación pantry):
    - `test_sse_emits_degraded_flag_on_pantry_violation`
    - `test_sse_no_degraded_flag_on_clean_pantry`

- **P1-2** (`_pantry_degraded_summary` en evento complete):
    - mismos tests que P0-2 verifican el summary en el body.

- **P1-5** (422 cuando faltan campos del form):
    - `test_sse_returns_422_on_missing_form_data`
    - `test_sse_returns_422_on_partial_form_data`

- **P1-6** (rate limit 3/60s per IP/user):
    - `test_sse_rate_limit_returns_429_after_3_requests`

A diferencia de los tests de `test_p0_1_endpoint_pantry_postvalidation.py`
(que usan stubs heavy para evitar importar FastAPI/Pydantic completo y poder
correr en CI sin el env de producción), este archivo asume el env de
desarrollo `mealfit` (conda) con todas las dependencias instaladas. Solo
patcheamos los puntos de I/O que tocarían DB real o LLM real — el resto
del wiring (FastAPI, Pydantic, dependency injection, rate limiter) corre
con su comportamiento real para validar el contrato E2E.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# [P1-10] Guard de aislamiento: si `test_p0_1_endpoint_pantry_postvalidation`
# ya corrió en la misma sesión pytest, sus stubs (auth.get_verified_user_id
# como `lambda *_a, **_kw`, pydantic con BaseModel=object, etc.) rompen la
# introspección de dependencias de FastAPI cuando intentamos usar TestClient
# real. FastAPI lee la firma `(*_a, **_kw)` del stub e interpreta `_a`/`_kw`
# como query params required → 422 inválido. Solución pragmática: skipear
# todo este módulo cuando detectamos cualquier stub conflictivo y dirigir al
# desarrollador a correrlo aislado. CI puede ejecutar:
#   pytest tests/test_p0_2_sse_pantry_validation_e2e.py
# para tener cobertura completa de este archivo. El P0-1 file ya cubre los
# mismos contratos en sus tests aislados (con mocks unitarios).
def _detect_p0_1_stubs() -> str | None:
    """Retorna mensaje de conflicto si stubs de P0-1 contaminan el env."""
    if "pydantic" in sys.modules and not hasattr(sys.modules["pydantic"], "create_model"):
        return "pydantic stubeado (falta create_model)"

    # Detectar stub de `auth.get_verified_user_id`: el real tiene parámetros
    # tipados; el stub es `lambda *_a, **_kw: None`.
    import inspect
    auth_mod = sys.modules.get("auth")
    if auth_mod:
        fn = getattr(auth_mod, "get_verified_user_id", None)
        if fn and callable(fn):
            try:
                sig = inspect.signature(fn)
                params = list(sig.parameters.values())
                if params and all(
                    p.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                    for p in params
                ):
                    return "auth.get_verified_user_id stubeado (firma *_a, **_kw)"
            except (ValueError, TypeError):
                pass
    return None


_conflict = _detect_p0_1_stubs()
if _conflict:
    pytest.skip(
        f"P1-10: tests E2E con TestClient requieren env limpio. "
        f"Conflicto detectado: {_conflict}. "
        f"Correr aislado: pytest tests/test_p0_2_sse_pantry_validation_e2e.py",
        allow_module_level=True,
    )

from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient


# [P1-PANTRY-GUARD-INITIAL-SKIP · 2026-05-18] ≥10 items para activar guard.
# Ver memoria del fix en `project_p1_pantry_guard_initial_skip_2026_05_18.md`.
PANTRY_OK = [
    "pollo 500g", "arroz 1000g", "tomate 300g", "cebolla 200g", "aceite 250ml",
    "huevo 12 unidades", "leche 1L", "queso 250g", "pan integral 500g", "yogurt 500g",
    "ajo 100g", "limón 200g",
]

# Replica de `_REQUIRED_FORM_FIELDS` en `routers/plans.py:_validate_form_data_min`.
# Si en el futuro se relaja un requisito, ambos archivos se actualizan en un
# solo sitio (el helper `_validate_form_data_min` es la fuente de verdad; este
# preset solo asegura que los tests envían payload realista).
MIN_FORM_DATA = {
    "age": 30,
    "mainGoal": "lose_fat",
    "weight": 154,
    "height": 170,
    "gender": "male",
    "activityLevel": "moderate",
    # [P0-FORM-4] `weightUnit` añadido como required en `_REQUIRED_FORM_FIELDS`.
    # Sin él, el backend defaulteaba silenciosamente a "lb" — bug de cálculo
    # nutricional. Tests E2E que esperan 200 deben enviarlo explícitamente.
    "weightUnit": "lb",
    # [P0-FORM-1] `householdSize` y `groceryDuration` añadidos como required.
    # Antes el backend defaulteaba silenciosamente a 1 / "weekly" → lista de
    # compras escalada erróneamente. Tests E2E que esperan 200 deben enviarlos.
    "householdSize": 1,
    "groceryDuration": "weekly",
    # [P0-FORM-3] `motivation` ahora required. Sin él, `_validate_form_data_min`
    # rechaza con 422. Inyectado al planner + day generator vía
    # `build_motivation_context`.
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


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _reset_rate_limiter():
    """Limpia el bucket sliding-window para que tests independientes no
    interfieran con el cap de 3/60s del `_PLAN_GEN_LIMITER` (P1-6)."""
    from routers.plans import _PLAN_GEN_LIMITER
    _PLAN_GEN_LIMITER._hits.clear()


def _build_app() -> FastAPI:
    """App mínima con solo el router de plans para aislar el test del lifespan
    completo de `app.py` (que arranca scheduler, conecta DB real, etc)."""
    from routers.plans import router as plans_router
    app = FastAPI()
    app.include_router(plans_router)
    return app


def _parse_sse_events(response_text: str) -> list:
    """Parsea el body completo de una respuesta SSE en lista de dicts.
    Cada evento del wire format `data: {...}\\n\\n` se decodifica como JSON;
    líneas vacías o malformadas se ignoran (heartbeats no JSON, etc.).
    """
    events = []
    for raw in response_text.splitlines():
        if not raw.startswith("data: "):
            continue
        try:
            events.append(json.loads(raw[6:]))
        except json.JSONDecodeError:
            continue
    return events


@pytest.fixture
def client():
    """TestClient con rate limiter limpio y app mínima.

    Usar `with` block para que httpx cierre las conexiones pendientes entre
    tests (importante con SSE).
    """
    _reset_rate_limiter()
    app = _build_app()
    with TestClient(app) as c:
        yield c


def _common_postprocess_patches():
    """Conjunto mínimo de patches para que el handler no toque DB real
    (Supabase, save_partial_plan, learning signals, etc.).

    Se patchean tanto los símbolos importados directamente en `routers.plans`
    como los lazy-imports dentro de `cron_tasks` que el helper invoca.
    """
    return [
        # Sesión / memoria / preferencias — DB writes auxiliares
        patch("routers.plans.get_or_create_session", return_value=None),
        patch("routers.plans.build_memory_context", return_value={
            "recent_messages": [], "full_context_str": "",
        }),
        patch("routers.plans.get_user_likes", return_value=[]),
        patch("routers.plans.get_active_rejections", return_value=[]),
        patch("routers.plans.analyze_preferences_agent", return_value=""),
        patch("routers.plans._user_has_profile", return_value=False),
        patch("routers.plans.update_user_health_profile", return_value=None),
        patch("routers.plans.log_api_usage", return_value=None),
        patch("routers.plans.save_message", return_value=None),
        patch("routers.plans._save_plan_and_track_background", return_value=None),
        patch("routers.plans.save_partial_plan_get_id", return_value=None),
        patch("routers.plans._resolve_request_tz_offset", return_value=0),
        # Auth — verify_api_quota stub para no exigir token real
        patch("routers.plans.verify_api_quota", return_value=None),
        # Bypass del rate limiter en tests que no son del rate limit (override
        # se hará per-test cuando sea necesario verificar 429).
        # Lazy-imports dentro de helpers
        patch("cron_tasks._seed_emergency_backup_if_empty", return_value=None),
        patch("cron_tasks.inject_learning_signals_from_profile",
              side_effect=lambda uid, pd: pd),
    ]


# ---------------------------------------------------------------------------
# P0-2 + P1-2: validación pantry y summary en complete
# ---------------------------------------------------------------------------

def test_sse_emits_degraded_flag_on_pantry_violation(client):
    """[P0-2/P1-2] El SSE debe:
    1) Invocar `_validate_and_retry_initial_chunk_against_pantry` (P0-2).
    2) Sellar `_initial_chunk_pantry_degraded=True` y `_initial_chunk_pantry_violation`.
    3) Adjuntar `_pantry_degraded_summary` con `degraded=True` (P1-2).
    """
    from contextlib import ExitStack

    pipeline_plan = {
        "days": [
            {"day": 1, "meals": [{"name": "BAD-salmon",
                                  "ingredients": ["salmón", "arroz"]}]}
        ]
    }

    async def _async_pipeline(*a, **k):
        return pipeline_plan

    helper_mock = MagicMock(
        return_value=(
            pipeline_plan,
            {
                "validated_ok": False,
                "attempts": 2,
                "degraded": True,
                "last_violation": "INEXISTENTES: salmón.",
                "mode": "strict",
                "pantry_size": len(PANTRY_OK),
                "meals_marked_violated": 1,
            },
        )
    )

    summary_mock = {
        "degraded": True,
        "degraded_days": [1],
        "reasons": ["initial_chunk_violation"],
        "initial_chunk_degraded": True,
        "current_mode": None,
    }

    with ExitStack() as stack:
        stack.enter_context(patch("routers.plans.arun_plan_pipeline",
                                  side_effect=_async_pipeline))
        stack.enter_context(patch("cron_tasks._validate_and_retry_initial_chunk_against_pantry",
                                  helper_mock))
        stack.enter_context(patch("cron_tasks.compute_pantry_degraded_summary",
                                  return_value=summary_mock))
        for cm in _common_postprocess_patches():
            stack.enter_context(cm)

        payload = {
            "user_id": "guest",
            "totalDays": 3,
            "current_pantry_ingredients": PANTRY_OK,
            "tzOffset": 0,
            **MIN_FORM_DATA,
        }
        response = client.post("/api/plans/analyze/stream", json=payload)

    assert response.status_code == 200, (
        f"SSE debe retornar 200; recibido {response.status_code}: {response.text[:300]}"
    )

    events = _parse_sse_events(response.text)
    complete_events = [e for e in events if e.get("event") == "complete"]
    assert complete_events, (
        f"Esperaba evento complete. Eventos recibidos: "
        f"{[e.get('event') for e in events]}"
    )

    plan = complete_events[-1].get("data") or {}

    # P0-2: helper invocado con argumentos correctos
    assert helper_mock.called, (
        "P0-2: el helper de validación pantry no se invocó en el path SSE"
    )
    helper_kwargs = helper_mock.call_args.kwargs
    assert helper_kwargs["pantry_ingredients"] == PANTRY_OK
    assert helper_kwargs["initial_result"] == pipeline_plan
    # `transport_label` es interno del wrapper `_run_pantry_validation_for_initial_chunk`
    # (uso solo para logs); NO se propaga al helper de cron_tasks. La distinción
    # SSE vs sync se valida indirectamente: este test corre solo a través del
    # endpoint `/analyze/stream`.

    # P0-2: flag de degradación sellado en el plan
    assert plan.get("_initial_chunk_pantry_degraded") is True, (
        f"P0-2: falta flag _initial_chunk_pantry_degraded. plan_keys={list(plan.keys())}"
    )
    assert "salmón" in (plan.get("_initial_chunk_pantry_violation") or ""), (
        "P0-2: violation message no se propagó al plan"
    )

    # P1-2: summary adjunto al complete event
    summary = plan.get("_pantry_degraded_summary")
    assert summary is not None, (
        "P1-2: _pantry_degraded_summary debe estar en el body del complete event"
    )
    assert summary.get("degraded") is True
    assert summary.get("initial_chunk_degraded") is True
    assert summary.get("degraded_days") == [1]


def test_sse_no_degraded_flag_on_clean_pantry(client):
    """[P1-2] Plan limpio → summary con `degraded=False`, sin flag degradado."""
    from contextlib import ExitStack

    pipeline_plan = {
        "days": [{"day": 1, "meals": [{"name": "OK",
                                       "ingredients": ["pollo", "arroz"]}]}]
    }

    async def _async_pipeline(*a, **k):
        return pipeline_plan

    helper_mock = MagicMock(
        return_value=(
            pipeline_plan,
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

    summary_mock = {
        "degraded": False,
        "degraded_days": [],
        "reasons": [],
        "initial_chunk_degraded": False,
        "current_mode": None,
    }

    with ExitStack() as stack:
        stack.enter_context(patch("routers.plans.arun_plan_pipeline",
                                  side_effect=_async_pipeline))
        stack.enter_context(patch("cron_tasks._validate_and_retry_initial_chunk_against_pantry",
                                  helper_mock))
        stack.enter_context(patch("cron_tasks.compute_pantry_degraded_summary",
                                  return_value=summary_mock))
        for cm in _common_postprocess_patches():
            stack.enter_context(cm)

        payload = {
            "user_id": "guest",
            "totalDays": 3,
            "current_pantry_ingredients": PANTRY_OK,
            "tzOffset": 0,
            **MIN_FORM_DATA,
        }
        response = client.post("/api/plans/analyze/stream", json=payload)

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    complete = next((e for e in events if e.get("event") == "complete"), None)
    assert complete is not None
    plan = complete.get("data") or {}

    assert plan.get("_initial_chunk_pantry_degraded") is not True, (
        "Plan limpio NO debe llevar flag de degradación"
    )
    summary = plan.get("_pantry_degraded_summary") or {}
    assert summary.get("degraded") is False, (
        "P1-2: summary debe estar presente con degraded=False en planes limpios"
    )


# ---------------------------------------------------------------------------
# P1-5: validación temprana de form_data
# ---------------------------------------------------------------------------

def test_sse_returns_422_on_missing_form_data(client):
    """[P1-5] Cuando el payload no tiene los campos mínimos, el SSE debe
    rechazar con 422 ANTES de abrir el StreamingResponse — el cliente recibe
    un JSON estándar, no un stream SSE con un único evento error."""
    payload = {
        "user_id": "guest",
        "totalDays": 3,
        "current_pantry_ingredients": [],
        "tzOffset": 0,
        # Intencionalmente sin MIN_FORM_DATA
    }
    # `verify_api_quota` también necesita patch para no exigir auth real
    with patch("routers.plans.verify_api_quota", return_value=None):
        response = client.post("/api/plans/analyze/stream", json=payload)

    assert response.status_code == 422, (
        f"P1-5: payload incompleto debe ser 422; recibido {response.status_code}: "
        f"{response.text[:200]}"
    )

    body = response.json()
    detail = body.get("detail")
    # Debe ser nuestro detail estructurado (dict), no el detail-list que
    # devuelve la validación de Pydantic.
    assert isinstance(detail, dict), (
        f"P1-5: detail debe ser dict (custom 422), no {type(detail).__name__}: {detail}"
    )
    assert detail.get("code") == "missing_required_fields", f"detail={detail}"

    missing = set(detail.get("missing_fields") or [])
    # [P0-FORM-4] `weightUnit` añadido al set requerido — el payload vacío no
    # incluye ningún campo, así que todos faltan, incluyendo weightUnit.
    # [P0-FORM-1] `householdSize`, `groceryDuration` añadidos como required
    # (antes defaulteaban silenciosamente a 1 / "weekly").
    # [P0-FORM-3] `motivation` añadida como required (antes campo huérfano,
    # ahora cableada al planner LLM).
    # [P1-2] `allergies` y `medicalConditions` añadidas como required (presence-
    # required + nonempty-array via `_REQUIRED_NONEMPTY_ARRAY_FIELDS`).
    # [P0-FORM-6] Sync con frontend: `scheduleType`, `cookingTime`, `budget`,
    # `sleepHours`, `stressLevel`, `dislikes`, `struggles` ahora required.
    expected = {
        "age", "mainGoal", "weight", "height", "gender", "activityLevel",
        "weightUnit", "householdSize", "groceryDuration", "motivation",
        "allergies", "medicalConditions",
        "scheduleType", "cookingTime", "budget", "sleepHours", "stressLevel",
        "dislikes", "struggles",
    }
    assert missing == expected, (
        f"P1-5: missing_fields esperado {expected}, recibido {missing}"
    )

    # Confirma que el frontend puede leer el detail.message para mostrar al usuario
    assert "Faltan campos críticos" in (detail.get("message") or "")


def test_sse_returns_422_on_partial_form_data(client):
    """[P1-5] Frontend valida `age` + `mainGoal`; pero si faltan biométricos
    el backend debe seguir rechazando — el frontend pre-existente pasaba
    pero clientes legacy/no oficiales no."""
    payload = {
        "user_id": "guest",
        "totalDays": 3,
        "current_pantry_ingredients": [],
        "tzOffset": 0,
        # Solo age + mainGoal (lo que el frontend valida)
        "age": 30,
        "mainGoal": "lose_fat",
    }
    with patch("routers.plans.verify_api_quota", return_value=None):
        response = client.post("/api/plans/analyze/stream", json=payload)
    assert response.status_code == 422
    detail = response.json().get("detail") or {}
    missing = set(detail.get("missing_fields") or [])
    # [P0-FORM-4] `weightUnit` ahora también se valida como required.
    # [P0-FORM-1] `householdSize`, `groceryDuration` añadidos como required.
    # [P0-FORM-3] `motivation` añadida como required.
    # [P1-2] `allergies`, `medicalConditions` required.
    # [P0-FORM-6] Sync con frontend: `scheduleType`, `cookingTime`, `budget`,
    # `sleepHours`, `stressLevel`, `dislikes`, `struggles` ahora required.
    assert missing == {
        "weight", "height", "gender", "activityLevel",
        "weightUnit", "householdSize", "groceryDuration", "motivation",
        "allergies", "medicalConditions",
        "scheduleType", "cookingTime", "budget", "sleepHours", "stressLevel",
        "dislikes", "struggles",
    }


# ---------------------------------------------------------------------------
# P1-6: rate limit per-endpoint
# ---------------------------------------------------------------------------

def test_sse_filters_internal_events_and_renames_day_completed(client):
    """[P1-11] El SSE handler debe:
    1) Filtrar eventos internos (`metric`, `token`, `tool_call`, `token_reset`)
       que el frontend ignora — no enviarlos por la wire.
    2) Renombrar `day_completed` → `day_complete` (el orquestador emite el
       primero pero el frontend escucha el segundo; bug latente fixeado como
       side-effect de P1-11).
    3) Pasar eventos públicos (`phase`, `day_started`, etc.) sin tocar.

    Test strategy: en lugar de mockear `arun_plan_pipeline`, mockeamos para
    que invoque el `progress_callback` con varios tipos de eventos antes de
    devolver el plan. Luego inspeccionamos los eventos parseados del response.
    """
    from contextlib import ExitStack

    pipeline_plan = {
        "days": [{"day": 1, "meals": [{"name": "OK", "ingredients": ["pollo"]}]}]
    }

    async def _async_pipeline_emitting_events(*a, **k):
        # `progress_callback` es el 5to arg posicional o kwarg
        cb = k.get("progress_callback")
        if cb is None and len(a) >= 5:
            cb = a[4]
        if cb is not None:
            # Eventos públicos: deben pasar
            cb({"event": "phase", "data": {"phase": "skeleton"}})
            cb({"event": "day_started", "data": {"day": 1}})
            # Evento legacy: debe renombrarse a "day_complete"
            cb({"event": "day_completed", "data": {"day": 1}})
            # Eventos internos: deben filtrarse (no llegar al cliente)
            cb({"event": "metric", "data": {"node": "test", "duration_ms": 100}})
            cb({"event": "token", "data": {"day": 1, "chunk": "Pollo a la "}})
            cb({"event": "tool_call", "data": {"day": 1, "tool": "lookup"}})
        return pipeline_plan

    def _helper(**kw):
        return (
            kw["initial_result"],
            {
                "validated_ok": True, "attempts": 1, "degraded": False,
                "last_violation": None, "mode": "off", "pantry_size": 0,
            },
        )

    summary_mock = {
        "degraded": False, "degraded_days": [], "reasons": [],
        "initial_chunk_degraded": False, "current_mode": None,
    }

    payload = {
        "user_id": "guest", "totalDays": 3,
        "current_pantry_ingredients": [], "tzOffset": 0, **MIN_FORM_DATA,
    }

    with ExitStack() as stack:
        stack.enter_context(patch("routers.plans.arun_plan_pipeline",
                                  side_effect=_async_pipeline_emitting_events))
        stack.enter_context(patch("cron_tasks._validate_and_retry_initial_chunk_against_pantry",
                                  side_effect=_helper))
        stack.enter_context(patch("cron_tasks.compute_pantry_degraded_summary",
                                  return_value=summary_mock))
        for cm in _common_postprocess_patches():
            stack.enter_context(cm)
        response = client.post("/api/plans/analyze/stream", json=payload)

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    event_names = [e.get("event") for e in events]

    # Eventos públicos pasaron
    assert "phase" in event_names, f"P1-11: 'phase' debe pasar el filtro. eventos={event_names}"
    assert "day_started" in event_names, "P1-11: 'day_started' debe pasar el filtro"
    assert "complete" in event_names, "P1-11: 'complete' debe pasar el filtro"

    # Alias: orquestador emitió `day_completed` pero wire debe llevar `day_complete`
    assert "day_complete" in event_names, (
        f"P1-11 alias: 'day_completed' del orquestador debe renombrarse a "
        f"'day_complete' en el wire. eventos={event_names}"
    )
    assert "day_completed" not in event_names, (
        f"P1-11 alias: 'day_completed' NO debe llegar al wire (renombrado). "
        f"eventos={event_names}"
    )

    # Eventos internos filtrados
    for internal in ("metric", "token", "tool_call"):
        assert internal not in event_names, (
            f"P1-11: '{internal}' es interno y debe filtrarse antes del yield. "
            f"eventos={event_names}"
        )


def test_sse_rate_limit_returns_429_after_3_requests(client):
    """[P1-6] El cap es 3/60s per user_id|ip. El 4to request en <60s debe ser 429.

    El TestClient se conecta desde la misma IP simulada en cada call
    (`testclient`), así que todos comparten el mismo bucket. Confirma:
      1. Los primeros 3 son aceptados (200).
      2. El 4to es 429.
      3. El detail menciona el límite (mensaje accionable para el cliente).
    """
    from contextlib import ExitStack

    pipeline_plan = {
        "days": [{"day": 1, "meals": [{"name": "OK",
                                       "ingredients": ["pollo"]}]}]
    }

    async def _async_pipeline(*a, **k):
        return pipeline_plan

    def _helper(**kw):
        return (
            kw["initial_result"],
            {
                "validated_ok": True,
                "attempts": 1,
                "degraded": False,
                "last_violation": None,
                "mode": "off",
                "pantry_size": 0,
            },
        )

    summary_mock = {
        "degraded": False,
        "degraded_days": [],
        "reasons": [],
        "initial_chunk_degraded": False,
        "current_mode": None,
    }

    payload = {
        "user_id": "guest",
        "totalDays": 3,
        "current_pantry_ingredients": [],
        "tzOffset": 0,
        **MIN_FORM_DATA,
    }

    with ExitStack() as stack:
        stack.enter_context(patch("routers.plans.arun_plan_pipeline",
                                  side_effect=_async_pipeline))
        stack.enter_context(patch("cron_tasks._validate_and_retry_initial_chunk_against_pantry",
                                  side_effect=_helper))
        stack.enter_context(patch("cron_tasks.compute_pantry_degraded_summary",
                                  return_value=summary_mock))
        for cm in _common_postprocess_patches():
            stack.enter_context(cm)

        # Primeros 3 requests: deben pasar
        for i in range(3):
            r = client.post("/api/plans/analyze/stream", json=payload)
            assert r.status_code == 200, (
                f"P1-6: request {i + 1} dentro del cap debe ser 200; "
                f"recibido {r.status_code}: {r.text[:200]}"
            )

        # 4to request: debe ser 429
        r4 = client.post("/api/plans/analyze/stream", json=payload)
        assert r4.status_code == 429, (
            f"P1-6: 4to request en <60s debe ser 429; recibido {r4.status_code}"
        )
        body = r4.json()
        # El RateLimiter retorna `detail` como string descriptivo
        detail = body.get("detail")
        assert isinstance(detail, str) and "Demasiadas solicitudes" in detail, (
            f"P1-6: detail del 429 debe ser human-readable. recibido: {detail!r}"
        )
