"""[P1-PLAN-DISPLAY-I18N · 2026-08-19] Capa de display i18n del plan (_display.<locale>).

tooltip-anchor: P1-PLAN-DISPLAY-I18N

Este archivo crece por task (Task 1 = motor). Tasks posteriores (disparadores, mutadores,
frontend contract) añaden secciones NUEVAS a este MISMO archivo — no crear ficheros nuevos
por task, es la convención fijada en el brief de Task 1.

================================================================================
SECCIÓN: MOTOR (Task 1 + fix round 1) — `plan_display_i18n.enrich_plan_display` /
`plan_display_i18n.schedule_plan_display_enrichment`
================================================================================

Todo el LLM y el I/O real (DB, cross-worker lock, circuit breaker) están mockeados: estos
tests NUNCA tocan Neon ni ningún provider — verifican el CONTRATO del motor (guards,
validación determinista, fail-open total, TOCTOU, lotes, telemetría). El fake de
`update_plan_data_atomic` captura el callback (mutator) y lo aplica a un dict `plan_data`
local, exactamente como pide el brief, para poder assertar el jsonb `_display` resultante
sin una DB real.

Fix round 1 (review `task-1-review.md`, CHANGES_REQUESTED — 3 CRITICAL, 6 IMPORTANT):
    - Finding 1 (CRITICAL): `_extract_canonical_name` descartaba ~10/11 líneas reales.
      Sección "extractor de canónicos" abajo prueba contra las 11 líneas del review.
    - Finding 2 (CRITICAL): el contrato "el callback escribe SOLO `_display`" no estaba
      anclado — `test_mutator_writes_only_display_key`.
    - Finding 3 (CRITICAL): prompt en español con el literal "English gloss" para los 4
      locales — sección "prompt nativo por locale".
    - Finding 4 (IMPORTANT): lock cross-worker sin liberar + clave ciega a `day_indices` —
      `test_cross_worker_lock_released_in_finally_and_includes_day_indices`.
    - Finding 5 (IMPORTANT): TOCTOU de índices (swap concurrente) — `test_toctou_mismatch_*`.
    - Finding 6 (IMPORTANT): sin cota de lote — sección "batching".
    - Finding 7 (IMPORTANT): telemetría solo en camino feliz — assertions de
      `telemetry_calls` actualizadas en los tests de json roto / 0 válidos.
    - Finding 8 (IMPORTANT): ownership del SELECT no anclado (mockeado por completo) —
      `test_all_meal_plans_queries_filter_user_id` (parser-based).
    - Finding 9 (IMPORTANT): import directo de `db_core` en vez de la fachada `db` —
      `test_source_imports_db_via_facade_not_db_core` (parser-based).
    - Finding 10/11 (MINOR): circuit breaker + `build_chat_llm` en vez de `ChatDeepSeek`
      hardcodeado — `test_circuit_breaker_open_skips_early`,
      `test_source_uses_build_chat_llm_not_hardcoded_chatdeepseek`.
    - Finding 12 (MINOR): `enriched_meals` contaba validados, no escritos — ya cubierto
      implícitamente por `test_toctou_mismatch_skips_meal_and_reports_reason` (un meal
      validado pero NO escrito ya no cuenta).
    - Finding 13 (MINOR): fixtures sintéticas que congelaban el bug del Finding 1 —
      reemplazadas por las 11 líneas reales del corpus.
    - Finding 14 (MINOR): `import copy` sin usar + orden de imports — `copy` ahora se usa
      en `test_mutator_writes_only_display_key`; el módulo movió sus imports antes de
      `logger = logging.getLogger(__name__)`.
"""
from __future__ import annotations

import asyncio
import copy
import re
from pathlib import Path

import pytest

import db as db_module
import plan_display_i18n as pdi
from routers.user_data import api_patch_profile, ProfilePatchBody


_MODULE_SRC_PATH = Path(pdi.__file__)


# ------------------------------------------------------------------
# Dobles de prueba
# ------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata or {}


class _FakeLLM:
    """Sustituye el cliente que `build_chat_llm(...)` devolvería: `.invoke(messages)` ->
    respuesta programada por el test (o excepción, para "excepción del provider")."""

    NEXT_RESPONSE = None
    NEXT_EXCEPTION = None
    captured_prompts: list = []
    invoke_count = 0

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def invoke(self, messages):
        type(self).invoke_count += 1
        type(self).captured_prompts.append(messages[0].content)
        if type(self).NEXT_EXCEPTION is not None:
            raise type(self).NEXT_EXCEPTION
        return type(self).NEXT_RESPONSE


def _base_meal() -> dict:
    return {
        "name": "Habichuelas guisadas",
        "description": "Guiso tradicional dominicano con habichuelas rojas.",
        "recipe": [
            "Sofreir el sazon en aceite caliente.",
            "Agregar las habichuelas y cocinar 20 minutos.",
        ],
        "ingredients": ["30 g Habichuelas rojas", "1 unidad Cebolla"],
    }


def _make_plan(meals: list) -> dict:
    return {"days": [{"meals": meals}]}


@pytest.fixture(autouse=True)
def _reset_fake_llm():
    _FakeLLM.NEXT_RESPONSE = None
    _FakeLLM.NEXT_EXCEPTION = None
    _FakeLLM.captured_prompts = []
    _FakeLLM.invoke_count = 0
    yield


@pytest.fixture
def engine(monkeypatch):
    """Instala los dobles de infraestructura y expone un `state` con:
      - `state["plan_data"]`: el dict mutable que `_fetch_plan_data`/`update_plan_data_atomic`
        comparten (simula la fila de `meal_plans.plan_data`).
      - `state["persist_calls"]`: lista de kwargs con los que se llamó al fake de
        `update_plan_data_atomic`.
      - `state["telemetry_calls"]`: lista de kwargs con los que se llamó al fake de
        `log_llm_usage_event`.
    """
    state = {"plan_data": None, "persist_calls": [], "telemetry_calls": []}

    monkeypatch.setattr(
        pdi, "_try_claim_enrich_lock_cross_worker",
        lambda plan_id, locale, day_indices: True,
    )
    monkeypatch.setattr(
        pdi, "_release_enrich_lock_cross_worker",
        lambda plan_id, locale, day_indices: None,
    )
    monkeypatch.setattr(pdi, "_circuit_breaker_can_proceed", lambda model_name: True)
    monkeypatch.setattr(pdi, "build_chat_llm", lambda model, **kwargs: _FakeLLM(**kwargs))
    monkeypatch.setattr(pdi, "_fetch_plan_data", lambda plan_id, user_id: state["plan_data"])

    def _fake_update_plan_data_atomic(plan_id, mutator, user_id=None, **kwargs):
        state["persist_calls"].append({"plan_id": plan_id, "user_id": user_id})
        result = mutator(state["plan_data"])
        if isinstance(result, dict):
            state["plan_data"] = result
        return state["plan_data"]

    monkeypatch.setattr(pdi, "update_plan_data_atomic", _fake_update_plan_data_atomic)

    def _fake_log_llm_usage_event(**kwargs):
        state["telemetry_calls"].append(kwargs)

    monkeypatch.setattr(pdi, "log_llm_usage_event", _fake_log_llm_usage_event)

    return state


def _set_plan(engine_state: dict, meals: list) -> None:
    engine_state["plan_data"] = _make_plan(meals)


def _valid_response_for_base_meal() -> _FakeResponse:
    return _FakeResponse(
        content=(
            '{"meals":[{"i":0,'
            '"name":"Stewed red beans",'
            '"description":"Traditional Dominican stew with red beans.",'
            '"recipe":["Saute the seasoning in hot oil.","Add the beans and cook 20 minutes."],'
            '"ingredients":["30 g red beans (Habichuelas rojas)","1 unit onion (Cebolla)"]}]}'
        ),
        usage_metadata={"input_tokens": 120, "output_tokens": 80},
    )


# ------------------------------------------------------------------
# 1. Respuesta válida completa
# ------------------------------------------------------------------


def test_respuesta_valida_completa_persiste_display(engine):
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _valid_response_for_base_meal()

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 1, "skipped": None}
    meal = engine["plan_data"]["days"][0]["meals"][0]
    display = meal["_display"]["en-US"]
    assert display["name"] == "Stewed red beans"
    assert display["description"] == "Traditional Dominican stew with red beans."
    assert display["recipe"] == [
        "Saute the seasoning in hot oil.",
        "Add the beans and cook 20 minutes.",
    ]
    assert display["ingredients"] == [
        "30 g red beans (Habichuelas rojas)",
        "1 unit onion (Cebolla)",
    ]
    assert engine["persist_calls"] == [{"plan_id": "plan-1", "user_id": "user-1"}]
    assert len(engine["telemetry_calls"]) == 1
    assert engine["telemetry_calls"][0]["node"] == "plan_display_i18n"
    assert engine["telemetry_calls"][0]["plan_id"] == "plan-1"
    assert engine["telemetry_calls"][0]["user_id"] == "user-1"


def test_mutator_writes_only_display_key(engine):
    """[Finding 2 · fix round 1] Ancla la línea roja de la spec: el callback de
    `update_plan_data_atomic` escribe SOLO `_display`, nunca `name`/`recipe`/
    `ingredients` (que deben permanecer en español canónico SIEMPRE — son
    identificadores del sistema). Snapshot ANTES del persist + comparación
    campo a campo tras excluir `_display` del meal resultante. Un mutation-test
    que añada `meal["name"] = display["name"]` dentro del mutator DEBE hacer
    fallar este test (el review lo verificó contra Task 1: pasaba los 26)."""
    original_meal = _base_meal()
    _set_plan(engine, [original_meal])
    snapshot_before = copy.deepcopy(engine["plan_data"]["days"][0]["meals"][0])
    _FakeLLM.NEXT_RESPONSE = _valid_response_for_base_meal()

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result["enriched_meals"] == 1
    meal_after = engine["plan_data"]["days"][0]["meals"][0]
    meal_after_sans_display = {k: v for k, v in meal_after.items() if k != "_display"}
    assert meal_after_sans_display == snapshot_before, (
        "El mutator tocó una clave del meal fuera de `_display` — el plan debe permanecer "
        "en español canónico SIEMPRE (línea roja de la spec)."
    )
    assert "_display" in meal_after and "en-US" in meal_after["_display"]


# ------------------------------------------------------------------
# 2. Línea de ingrediente sin canónico identificable en el original
#    -> pasa SIN check (no se descarta, no se exige el gloss).
# ------------------------------------------------------------------


def test_ingrediente_sin_canonico_identificable_pasa_sin_check(engine):
    meal = _base_meal()
    # "2 unidades" es consumido ENTERO por el regex de cantidad/unidad -> canónico vacío.
    meal["ingredients"] = ["2 unidades"]
    meal["recipe"] = ["Un solo paso."]
    _set_plan(engine, [meal])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(
        content=(
            '{"meals":[{"i":0,"name":"Something",'
            '"description":"A dish.",'
            '"recipe":["One step."],'
            '"ingredients":["2 units"]}]}'
        )
    )

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 1, "skipped": None}
    display = engine["plan_data"]["days"][0]["meals"][0]["_display"]["en-US"]
    assert display["ingredients"] == ["2 units"]


def test_ingrediente_con_canonico_perdido_cae_a_la_linea_original(engine):
    """Si la línea traducida NO contiene el canónico español, esa línea puntual se
    descarta y cae de vuelta al texto original — el resto del meal SÍ se enriquece
    (delete-on-write per-línea, no per-meal)."""
    meal = _base_meal()  # ingredients: ["30 g Habichuelas rojas", "1 unidad Cebolla"]
    _set_plan(engine, [meal])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(
        content=(
            '{"meals":[{"i":0,'
            '"name":"Stewed red beans",'
            '"description":"Traditional Dominican stew.",'
            '"recipe":["Saute the seasoning in hot oil.","Add the beans and cook 20 minutes."],'
            '"ingredients":["30 g red beans","1 unit onion (Cebolla)"]}]}'
            # ^ primera línea PERDIÓ el gloss "(Habichuelas rojas)" -> debe caer al original.
        )
    )

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 1, "skipped": None}
    display = engine["plan_data"]["days"][0]["meals"][0]["_display"]["en-US"]
    assert display["ingredients"][0] == "30 g Habichuelas rojas"  # fallback español
    assert display["ingredients"][1] == "1 unit onion (Cebolla)"  # traducida, se conserva


# ------------------------------------------------------------------
# 3. Arrays desalineados -> ESE meal se descarta; los demás del lote se conservan.
# ------------------------------------------------------------------


def test_arrays_desalineados_descarta_solo_ese_meal(engine):
    meal_ok = _base_meal()
    meal_bad = _base_meal()
    meal_bad["name"] = "Otro plato"
    _set_plan(engine, [meal_ok, meal_bad])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(
        content=(
            '{"meals":['
            '{"i":0,"name":"Stewed red beans","description":"Desc.",'
            '"recipe":["Step A.","Step B."],'
            '"ingredients":["30 g red beans (Habichuelas rojas)","1 unit onion (Cebolla)"]},'
            '{"i":1,"name":"Bad dish","description":"Desc.",'
            '"recipe":["Only one step, original had two."],'
            '"ingredients":["30 g red beans (Habichuelas rojas)","1 unit onion (Cebolla)"]}'
            "]}"
        )
    )

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 1, "skipped": None}
    meals = engine["plan_data"]["days"][0]["meals"]
    assert "_display" in meals[0] and "en-US" in meals[0]["_display"]
    assert "_display" not in meals[1]
    assert len(engine["telemetry_calls"]) == 1  # UN batch (misma día) -> UNA llamada


def test_todos_los_meals_desalineados_no_persiste_nada(engine):
    meal_bad = _base_meal()
    _set_plan(engine, [meal_bad])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(
        content=(
            '{"meals":[{"i":0,"name":"Bad","description":"Desc.",'
            '"recipe":["Only one step."],'
            '"ingredients":["only one ingredient"]}]}'
        )
    )

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "no_valid_meals"}
    assert "_display" not in engine["plan_data"]["days"][0]["meals"][0]
    assert engine["persist_calls"] == []
    # [Finding 7] La llamada LLM SÍ ocurrió (y se pagó) aunque la validación
    # posterior descartara todo -> la telemetría debe reflejar ese gasto.
    assert len(engine["telemetry_calls"]) == 1


# ------------------------------------------------------------------
# 4. JSON roto -> fail-open, no persiste nada (pero SÍ registra el gasto).
# ------------------------------------------------------------------


def test_json_roto_fail_open(engine):
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(content="esto no es JSON {{{")

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "json_parse_error"}
    assert "_display" not in engine["plan_data"]["days"][0]["meals"][0]
    assert engine["persist_calls"] == []
    # [Finding 7 · fix round 1] Antes quedaba en `[]` — el invoke SÍ ocurrió (y se
    # pagó) antes de que fallara el parseo del JSON, así que el costo debe quedar
    # registrado aunque el resultado final sea 0 meals enriquecidos.
    assert len(engine["telemetry_calls"]) == 1


def test_json_envuelto_en_code_fence_se_parsea(engine):
    """El contrato pide JSON estricto, pero un LLM real a veces envuelve en ```json — el
    parser tolera el fence sin relajar el resto de la validación."""
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(
        content=(
            "```json\n"
            '{"meals":[{"i":0,"name":"Stewed red beans","description":"Desc.",'
            '"recipe":["Step A.","Step B."],'
            '"ingredients":["30 g red beans (Habichuelas rojas)","1 unit onion (Cebolla)"]}]}'
            "\n```"
        )
    )

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 1, "skipped": None}


# ------------------------------------------------------------------
# 5. Knob off -> no-op, cero llamadas LLM.
# ------------------------------------------------------------------


def test_knob_off_es_no_op(engine, monkeypatch):
    monkeypatch.setenv("MEALFIT_PLAN_DISPLAY_I18N", "false")
    _set_plan(engine, [_base_meal()])

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "knob_off"}
    assert _FakeLLM.invoke_count == 0
    assert engine["persist_calls"] == []


# ------------------------------------------------------------------
# 6. es-DO / locale inválido -> no-op.
# ------------------------------------------------------------------


def test_es_do_es_no_op(engine):
    _set_plan(engine, [_base_meal()])

    result = pdi.enrich_plan_display("plan-1", "user-1", "es-DO")

    assert result == {"enriched_meals": 0, "skipped": "locale"}
    assert _FakeLLM.invoke_count == 0


def test_locale_desconocido_es_no_op(engine):
    _set_plan(engine, [_base_meal()])

    result = pdi.enrich_plan_display("plan-1", "user-1", "xx-XX")

    assert result == {"enriched_meals": 0, "skipped": "locale"}
    assert _FakeLLM.invoke_count == 0


def test_locale_none_es_no_op(engine):
    _set_plan(engine, [_base_meal()])

    result = pdi.enrich_plan_display("plan-1", "user-1", None)

    assert result == {"enriched_meals": 0, "skipped": "locale"}


# ------------------------------------------------------------------
# 7. Excepción del provider -> fail-open TOTAL, nunca lanza.
# ------------------------------------------------------------------


def test_excepcion_del_provider_fail_open(engine):
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_EXCEPTION = RuntimeError("provider caído")

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "llm_exception"}
    assert "_display" not in engine["plan_data"]["days"][0]["meals"][0]
    assert engine["persist_calls"] == []
    # El invoke NUNCA retornó un `response` -> no hay nada que registrar.
    assert engine["telemetry_calls"] == []


def test_excepcion_inesperada_en_cualquier_punto_jamas_lanza(engine, monkeypatch):
    """Fail-open TOTAL (Global Constraint de la spec): incluso una excepción fuera de los
    puntos de guard conocidos (aquí, en la construcción del plan) debe devolver un dict,
    NUNCA propagar."""
    def _boom(plan_id, user_id):
        raise RuntimeError("fetch roto")

    monkeypatch.setattr(pdi, "_fetch_plan_data", _boom)

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "exception"}


# ------------------------------------------------------------------
# Guards adicionales: plan no encontrado / sin days / sin meals.
# ------------------------------------------------------------------


def test_plan_no_encontrado_ownership(engine):
    engine["plan_data"] = None  # simula 0 filas (id no existe o user_id no matchea)

    result = pdi.enrich_plan_display("plan-1", "otro-user", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "not_found"}


def test_plan_sin_days(engine):
    engine["plan_data"] = {"days": []}

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "no_days"}


def test_plan_sin_meals_en_los_days_pedidos(engine):
    engine["plan_data"] = {"days": [{"meals": []}]}

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "no_meals"}
    assert _FakeLLM.invoke_count == 0


def test_day_indices_filtra_los_dias_incluidos(engine):
    """`day_indices` acota el lote — solo esos días entran al prompt/lote."""
    meal_day0 = _base_meal()
    meal_day1 = _base_meal()
    meal_day1["name"] = "Otro día"
    engine["plan_data"] = {"days": [{"meals": [meal_day0]}, {"meals": [meal_day1]}]}
    _FakeLLM.NEXT_RESPONSE = _valid_response_for_base_meal()

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US", day_indices=[0])

    assert result == {"enriched_meals": 1, "skipped": None}
    assert "_display" in engine["plan_data"]["days"][0]["meals"][0]
    assert "_display" not in engine["plan_data"]["days"][1]["meals"][0]
    # El prompt solo debe mencionar el meal del día 0.
    assert "Otro día" not in _FakeLLM.captured_prompts[0]


def test_day_indices_vacio_es_no_op(engine):
    _set_plan(engine, [_base_meal()])

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US", day_indices=[])

    assert result == {"enriched_meals": 0, "skipped": "no_meals"}
    assert _FakeLLM.invoke_count == 0


# ------------------------------------------------------------------
# Finding 5 (TOCTOU): race de índices entre la lectura (fuera del lock) y el
# persist (dentro del FOR UPDATE) — un swap concurrente cambia el meal en esa
# posición y el mutator debe SALTARLO, no pegarle la traducción vieja encima.
# ------------------------------------------------------------------


def test_toctou_mismatch_skips_meal_and_reports_reason(engine, monkeypatch):
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _valid_response_for_base_meal()

    def _fake_update_with_concurrent_swap(plan_id, mutator, user_id=None, **kwargs):
        engine["persist_calls"].append({"plan_id": plan_id, "user_id": user_id})
        # Simula un swap-meal/persist concurrente: el nombre del meal en esa
        # posición cambia DESPUÉS de que `targets` ya se había leído.
        engine["plan_data"]["days"][0]["meals"][0]["name"] = "Plato reemplazado por swap"
        result = mutator(engine["plan_data"])
        if isinstance(result, dict):
            engine["plan_data"] = result
        return engine["plan_data"]

    monkeypatch.setattr(pdi, "update_plan_data_atomic", _fake_update_with_concurrent_swap)

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "persist_stale_mismatch"}
    meal = engine["plan_data"]["days"][0]["meals"][0]
    assert "_display" not in meal
    assert meal["name"] == "Plato reemplazado por swap"  # el swap gana, no lo pisamos


def test_toctou_mismatch_no_bloquea_otros_meals_del_mismo_lote(engine, monkeypatch):
    """El TOCTOU es POR MEAL: si solo uno de los dos meals de un lote cambió
    concurrentemente, el otro debe seguir escribiéndose."""
    meal0 = _base_meal()
    meal1 = _base_meal()
    meal1["name"] = "Segundo plato"
    _set_plan(engine, [meal0, meal1])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(
        content=(
            '{"meals":['
            '{"i":0,"name":"Stewed red beans","description":"Desc.",'
            '"recipe":["Step A.","Step B."],'
            '"ingredients":["30 g red beans (Habichuelas rojas)","1 unit onion (Cebolla)"]},'
            '{"i":1,"name":"Second dish","description":"Desc.",'
            '"recipe":["Step A.","Step B."],'
            '"ingredients":["30 g red beans (Habichuelas rojas)","1 unit onion (Cebolla)"]}'
            "]}"
        )
    )

    def _fake_update_partial_swap(plan_id, mutator, user_id=None, **kwargs):
        engine["persist_calls"].append({"plan_id": plan_id, "user_id": user_id})
        # Solo el meal 0 cambió concurrentemente; el meal 1 sigue intacto.
        engine["plan_data"]["days"][0]["meals"][0]["name"] = "Reemplazado por swap"
        result = mutator(engine["plan_data"])
        if isinstance(result, dict):
            engine["plan_data"] = result
        return engine["plan_data"]

    monkeypatch.setattr(pdi, "update_plan_data_atomic", _fake_update_partial_swap)

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 1, "skipped": None}
    meals = engine["plan_data"]["days"][0]["meals"]
    assert "_display" not in meals[0]  # saltado por TOCTOU
    assert "_display" in meals[1] and "en-US" in meals[1]["_display"]  # este sí escribió


# ------------------------------------------------------------------
# Finding 4: el lock cross-worker se libera SIEMPRE (finally) y su clave
# incluye los `day_indices` normalizados del lote.
# ------------------------------------------------------------------


def test_cross_worker_lock_released_in_finally_and_includes_day_indices(engine, monkeypatch):
    claimed = []
    released = []
    monkeypatch.setattr(
        pdi, "_try_claim_enrich_lock_cross_worker",
        lambda plan_id, locale, day_indices: (claimed.append((plan_id, locale, day_indices)) or True),
    )
    monkeypatch.setattr(
        pdi, "_release_enrich_lock_cross_worker",
        lambda plan_id, locale, day_indices: released.append((plan_id, locale, day_indices)),
    )
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _valid_response_for_base_meal()

    pdi.enrich_plan_display("plan-1", "user-1", "en-US", day_indices=[2, 0, 1, 0])

    assert claimed == [("plan-1", "en-US", [0, 1, 2])]  # normalizado: sorted + dedup
    assert released == [("plan-1", "en-US", [0, 1, 2])], (
        "el lock cross-worker NUNCA se liberó — Finding 4 sigue abierto"
    )


def test_cross_worker_lock_released_even_on_failure_path(engine, monkeypatch):
    """El release debe ocurrir en TODAS las ramas de salida, no solo el camino
    feliz — aquí el lote entero falla por JSON roto."""
    released = []
    monkeypatch.setattr(
        pdi, "_release_enrich_lock_cross_worker",
        lambda plan_id, locale, day_indices: released.append((plan_id, locale, day_indices)),
    )
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(content="esto no es JSON {{{")

    pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert released, "el lock cross-worker no se liberó en el camino de fallo"


def test_cross_worker_lock_not_released_if_never_claimed(engine, monkeypatch):
    """Si el claim falla (otro worker tiene el lock), NO se debe invocar el
    release — no hay nada que liberar y sería un DELETE inútil/engañoso."""
    released = []
    monkeypatch.setattr(
        pdi, "_try_claim_enrich_lock_cross_worker",
        lambda plan_id, locale, day_indices: False,
    )
    monkeypatch.setattr(
        pdi, "_release_enrich_lock_cross_worker",
        lambda plan_id, locale, day_indices: released.append(1),
    )
    _set_plan(engine, [_base_meal()])

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "dedupe_locked"}
    assert released == []


# ------------------------------------------------------------------
# Finding 10: circuit breaker.
# ------------------------------------------------------------------


def test_circuit_breaker_open_skips_early(engine, monkeypatch):
    monkeypatch.setattr(pdi, "_circuit_breaker_can_proceed", lambda model_name: False)
    _set_plan(engine, [_base_meal()])

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "circuit_breaker_open"}
    assert _FakeLLM.invoke_count == 0
    assert engine["persist_calls"] == []


# ------------------------------------------------------------------
# Finding 6: cota de lote (`MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS`).
# ------------------------------------------------------------------


def test_batching_splits_by_day_and_persists_each_batch(engine, monkeypatch):
    """Con `BATCH_DAYS=1` un plan de 2 días dispara DOS llamadas LLM y DOS
    persists independientes — si un lote falla, el otro sigue enriqueciéndose
    (recuperación parcial, en vez de una única llamada gigante todo-o-nada)."""
    monkeypatch.setenv("MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS", "1")
    meal0 = _base_meal()
    meal1 = _base_meal()
    meal1["name"] = "Día 2"
    engine["plan_data"] = {"days": [{"meals": [meal0]}, {"meals": [meal1]}]}

    responses = [
        _FakeResponse(
            content=(
                '{"meals":[{"i":0,"name":"Day 1 dish","description":"Desc.",'
                '"recipe":["Step A.","Step B."],'
                '"ingredients":["30 g red beans (Habichuelas rojas)","1 unit onion (Cebolla)"]}]}'
            )
        ),
        _FakeResponse(content="esto no es JSON {{{"),  # el lote 2 falla — recuperación parcial
    ]

    class _SequencedLLM:
        invoke_count = 0
        captured_prompts: list = []

        def __init__(self, *a, **kw):
            pass

        def invoke(self, messages):
            _SequencedLLM.captured_prompts.append(messages[0].content)
            _SequencedLLM.invoke_count += 1
            return responses.pop(0)

    monkeypatch.setattr(pdi, "build_chat_llm", lambda model, **kw: _SequencedLLM())

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 1, "skipped": None}
    assert "_display" in engine["plan_data"]["days"][0]["meals"][0]
    assert "_display" not in engine["plan_data"]["days"][1]["meals"][0]
    assert len(engine["persist_calls"]) == 1  # solo el lote que sí validó persiste
    assert _SequencedLLM.invoke_count == 2  # DOS llamadas — una por lote
    assert len(engine["telemetry_calls"]) == 2  # AMBAS se registran (Finding 7)


def test_max_output_tokens_y_modelo_se_pasan_a_build_chat_llm(engine, monkeypatch):
    monkeypatch.setenv("MEALFIT_PLAN_DISPLAY_I18N_MAX_OUTPUT_TOKENS", "1234")
    captured_kwargs = {}

    def _fake_build_chat_llm(model, **kwargs):
        captured_kwargs["model"] = model
        captured_kwargs.update(kwargs)
        return _FakeLLM()

    monkeypatch.setattr(pdi, "build_chat_llm", _fake_build_chat_llm)
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(content="esto no es JSON {{{")

    pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert captured_kwargs.get("max_output_tokens") == 1234
    assert captured_kwargs.get("model") == pdi._plan_display_i18n_model_name()


def test_batch_days_knob_registrado_y_clampa():
    import os

    old = os.environ.get("MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS")
    try:
        os.environ["MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS"] = "0"  # fuera del clamp [1,30]
        assert pdi._plan_display_i18n_batch_days() == 4  # cae al default
        os.environ["MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS"] = "7"
        assert pdi._plan_display_i18n_batch_days() == 7
    finally:
        if old is None:
            os.environ.pop("MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS", None)
        else:
            os.environ["MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS"] = old


# ------------------------------------------------------------------
# `_extract_canonical_name`: unidad interna reusada por la validación per-línea.
#
# [Finding 1 / 13 · fix round 1] Las 11 líneas del corpus REAL que el reviewer
# midió contra Task 1 (todas DESCARTABAN con la v1, salvo "Oregano dominicano").
# El "gloss correcto" de cada una es el que un LLM razonable produciría — el
# canónico extraído DEBE aparecer ahí como substring accent-insensitive.
# ------------------------------------------------------------------

_REAL_CORPUS_LINES = [
    ("30 g de Habichuelas rojas secas", "30 g dried red beans (Habichuelas rojas)"),
    ("1½ tazas de judías pintas cocidas", "1½ cups cooked pinto beans (judías pintas)"),
    ("Oregano dominicano", "Dominican oregano (Oregano dominicano)"),
    ("3 tazas de acelgas picadas", "3 cups chopped chard (acelgas)"),
    ("180g de arroz blanco", "180 g white rice (arroz blanco)"),
    ("50 g de zanahoria rallada", "50 g grated carrot (zanahoria)"),
    ("10 g de semillas de linaza", "10 g flaxseeds (semillas de linaza)"),
    ("100g de Pechuga de pollo", "100 g chicken breast (Pechuga de pollo)"),
    ("1 taza de Lechosa", "1 cup papaya (Lechosa)"),
    ("Sal al gusto", "Salt to taste (Sal)"),
    ("200 g de Salami", "200 g salami (Salami)"),
]


@pytest.mark.parametrize("original, gloss", _REAL_CORPUS_LINES)
def test_canonical_extraction_validates_against_real_corpus(original, gloss):
    canon = pdi._extract_canonical_name(original)
    assert canon, f"{original!r} debería rendir un canónico identificable"
    assert pdi.strip_accents(canon).lower() in pdi.strip_accents(gloss).lower(), (
        f"original={original!r} canon={canon!r} NO encontrado en gloss={gloss!r}"
    )


def test_canonical_extraction_corpus_floor_at_least_10_of_11():
    """Ancla el piso pactado con el controller (10/11) de forma independiente
    del test parametrizado de arriba — el fix real mide 11/11."""
    hits = 0
    for original, gloss in _REAL_CORPUS_LINES:
        canon = pdi._extract_canonical_name(original)
        if canon and pdi.strip_accents(canon).lower() in pdi.strip_accents(gloss).lower():
            hits += 1
    assert hits >= 10, f"Solo {hits}/11 líneas reales validan (mínimo requerido: 10/11)"
    assert hits == 11, (
        f"Regresión: el fix medía 11/11 y ahora mide {hits}/11 — investigar antes de "
        f"relajar el piso a 10."
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("30 g Habichuelas rojas", "Habichuelas rojas"),  # sin "de" conector
        ("1 unidad Cebolla", "Cebolla"),
        ("2 unidades", ""),                                # solo cantidad -> sin canónico
        ("", ""),
        ("Sal al gusto", "Sal"),                            # [Finding 1] "al gusto" SE quita
        ("100 g Pollo (sin piel)", "Pollo"),
        ("2 lonjas de queso", "queso"),                     # [Finding 1] "lonjas" faltaba
    ],
)
def test_extract_canonical_name_edge_cases(raw, expected):
    assert pdi._extract_canonical_name(raw) == expected


def test_canonical_extraction_preserves_food_identity_modifiers():
    """Modificadores que SÍ cambian la identidad del alimento (no son estado de
    cocción) deben CONSERVARSE — lo contrario de los modificadores de cocción."""
    assert pdi._extract_canonical_name("180g de arroz blanco") == "arroz blanco"
    assert "pintas" in pdi._extract_canonical_name("1½ tazas de judías pintas cocidas")
    assert pdi._extract_canonical_name("Oregano dominicano") == "Oregano dominicano"


def test_canonical_extraction_preserves_internal_de_in_compound_names():
    """Solo el conector LÍDER (cantidad/unidad -> "de" -> alimento) se quita —
    un "de" interno del propio nombre del alimento sobrevive."""
    assert pdi._extract_canonical_name("10 g de semillas de linaza") == "semillas de linaza"
    assert pdi._extract_canonical_name("100g de Pechuga de pollo") == "Pechuga de pollo"


# ------------------------------------------------------------------
# Prompt nativo por locale (Finding 3).
# ------------------------------------------------------------------

_PROMPT_TEST_TARGETS = [
    {
        "day_idx": 0, "meal_idx": 0,
        "name": "Habichuelas guisadas", "description": "Guiso.",
        "recipe": ["Paso 1."], "ingredients": ["30 g Habichuelas rojas"],
    }
]


def test_prompt_native_directive_no_english_gloss_leak_for_non_english_locales():
    for locale in ("pt-BR", "fr-FR", "it-IT"):
        prompt = pdi._build_prompt(_PROMPT_TEST_TARGETS, locale)
        assert "English gloss" not in prompt, (
            f"{locale}: el prompt sigue usando el literal en inglés 'English gloss'"
        )

    en_prompt = pdi._build_prompt(_PROMPT_TEST_TARGETS, "en-US")
    assert "English gloss" in en_prompt  # legítimo ÚNICAMENTE para en-US


def test_prompt_directive_is_native_per_locale():
    markers = {
        "en-US": "EXCLUSIVELY in English",
        "pt-BR": "EXCLUSIVAMENTE em Português",
        "fr-FR": "EXCLUSIVEMENT en Français",
        "it-IT": "ESCLUSIVAMENTE in Italiano",
    }
    for locale, marker in markers.items():
        prompt = pdi._build_prompt(_PROMPT_TEST_TARGETS, locale)
        assert marker in prompt, f"{locale}: falta la directiva nativa esperada ({marker!r})"


def test_prompt_directive_demonstrates_gloss_example_in_native_language():
    """Cada directiva debe DEMOSTRAR el formato del gloss con un ejemplo real,
    no solo describirlo (lección P1-COACH-LANGUAGE-NATIVE: 'instrucción Y
    demostración a la vez')."""
    for locale in pdi._COACH_LANGUAGE_NAMES:
        prompt = pdi._build_prompt(_PROMPT_TEST_TARGETS, locale)
        assert "Habichuelas rojas" in prompt, f"{locale}: falta el ejemplo de gloss demostrado"


def test_prompt_no_spanish_instruction_for_non_spanish_locales():
    """Ninguna palabra de la INSTRUCCIÓN (no de los datos, que sí van en
    español) debe quedar en español para los locales no-es. Chequeo barato:
    la palabra 'Responde' (instrucción típica en español del v1) no debe
    aparecer en el directive de pt-BR/fr-FR/it-IT."""
    for locale in ("pt-BR", "fr-FR", "it-IT"):
        directive = pdi._DISPLAY_LANGUAGE_DIRECTIVES[locale]
        assert "Responde" not in directive
        assert "Traduce estos platos" not in directive


# ------------------------------------------------------------------
# schedule_plan_display_enrichment: wrapper fire-and-forget.
# ------------------------------------------------------------------


def test_schedule_dispara_enrich_en_background(engine, monkeypatch):
    calls = []
    real_enrich = pdi.enrich_plan_display

    def _spy(plan_id, user_id, locale, day_indices=None):
        calls.append((plan_id, user_id, locale, day_indices))
        return real_enrich(plan_id, user_id, locale, day_indices=day_indices)

    monkeypatch.setattr(pdi, "enrich_plan_display", _spy)

    class _SyncThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(pdi.threading, "Thread", _SyncThread)

    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _valid_response_for_base_meal()

    pdi.schedule_plan_display_enrichment("plan-1", "user-1", "en-US", day_indices=[0])

    assert calls == [("plan-1", "user-1", "en-US", [0])]
    assert "_display" in engine["plan_data"]["days"][0]["meals"][0]


def test_schedule_knob_off_no_lanza_thread(monkeypatch):
    monkeypatch.setenv("MEALFIT_PLAN_DISPLAY_I18N", "false")
    started = {"count": 0}

    class _CountingThread:
        def __init__(self, target=None, daemon=None):
            pass

        def start(self):
            started["count"] += 1

    monkeypatch.setattr(pdi.threading, "Thread", _CountingThread)

    pdi.schedule_plan_display_enrichment("plan-1", "user-1", "en-US")

    assert started["count"] == 0


def test_schedule_locale_invalido_no_lanza_thread(monkeypatch):
    started = {"count": 0}

    class _CountingThread:
        def __init__(self, target=None, daemon=None):
            pass

        def start(self):
            started["count"] += 1

    monkeypatch.setattr(pdi.threading, "Thread", _CountingThread)

    pdi.schedule_plan_display_enrichment("plan-1", "user-1", "es-DO")

    assert started["count"] == 0


# ================================================================================
# SECCIÓN: OWNERSHIP + FACADE (parser-based, Findings 8 y 9)
#
# Estos tests parsean el SOURCE del módulo — no el runtime mockeado — porque
# el review demostró que los 26 tests funcionales de Task 1 mockeaban
# `_fetch_plan_data` por completo y por tanto NUNCA ejercían el SELECT real:
# un mutation-test que le quitara `AND user_id = %s` seguía pasando los 26.
# ================================================================================


@pytest.fixture(scope="module")
def _module_src() -> str:
    return _MODULE_SRC_PATH.read_text(encoding="utf-8")


def test_all_meal_plans_queries_filter_user_id(_module_src):
    """[Finding 8 · fix round 1] Toda query `FROM`/`UPDATE meal_plans` en el
    módulo debe filtrar `user_id = %s` cerca (ventana de 150 chars a cada
    lado — suficiente para SELECTs de una línea como los de este módulo)."""
    matches = list(re.finditer(r"(?:FROM|UPDATE)\s+meal_plans\b", _module_src))
    assert matches, (
        "sanity: no se encontró ninguna query FROM/UPDATE meal_plans en el módulo — "
        "si esto es intencional (el módulo dejó de tocar meal_plans directo), "
        "actualizar/retirar este test explícitamente."
    )
    offenders = []
    for m in matches:
        window = _module_src[max(0, m.start() - 150): m.end() + 150]
        if not re.search(r"user_id\s*=\s*%s", window):
            snippet = _module_src[max(0, m.start() - 40): m.end() + 10].replace("\n", " ")
            offenders.append(snippet)
    assert not offenders, (
        "query(s) sobre meal_plans sin `user_id = %s` cerca (invariante I2, CLAUDE.md): "
        + " | ".join(offenders)
    )


def test_source_imports_db_via_facade_not_db_core(_module_src):
    """[Finding 9 · fix round 1] Global Constraint del plan: imports vía la
    fachada `from db import ...` (P3-DB-IMPORTS-FACADE) — nunca `db_core`
    directo. `db_core.py` no define `__all__`, así que el import vía fachada
    ya funciona hoy sin cambio de comportamiento."""
    assert "from db_core import" not in _module_src, (
        "el módulo importa directo de db_core — debe ir vía la fachada `from db import ...`"
    )
    assert re.search(r"^from db import .*\bexecute_sql_query\b", _module_src, re.MULTILINE), (
        "falta el import de execute_sql_query vía la fachada `from db import ...`"
    )
    assert re.search(r"^from db import .*\bexecute_sql_write\b", _module_src, re.MULTILINE), (
        "falta el import de execute_sql_write vía la fachada `from db import ...`"
    )


def test_source_uses_build_chat_llm_not_hardcoded_chatdeepseek(_module_src):
    """[Finding 11 · fix round 1] `ChatDeepSeek(...)` hardcodeado enviaría un
    modelo OpenAI (p.ej. gpt-5.6-luna) al base_url de DeepSeek con la key
    equivocada (mismo bug que P1-DAYGEN-LUNA-CANARY). Debe construirse vía
    `llm_provider.build_chat_llm`, que rutea por prefijo del model id."""
    assert "build_chat_llm(" in _module_src, "el módulo debe usar la fábrica build_chat_llm"
    assert "ChatDeepSeek(" not in _module_src, (
        "hallado un ChatDeepSeek(...) hardcodeado — usar build_chat_llm(model, ...) en su lugar"
    )


def test_no_print_statements(_module_src):
    """Convención del repo (P2-LOGGER-MIGRATION): cero print() en código
    productivo. Sanity barato, redundante con test_p2_logger_migration.py
    pero ancla la intención EN este archivo."""
    assert "print(" not in _module_src


# ================================================================================
# SECCIÓN: DISPARADORES 1A, 1B, 2 y 4 (Task 2 + fix round 1) — parser-based + funcional
#
# Numeración de la spec: 1 = persist inicial (1A chunked / 1B no-chunked — el
# review de fix round 1 encontró que el disparador 1 original solo cubría el
# camino chunked; F1), 2 = chunk worker, 3 = mutadores (Task 3, NO aquí), 4 =
# cambio de idioma (PATCH /profile — el anchor vivía mal-nombrado `TRIGGER-3`
# en Task 2 original; F4 lo renombró). El disparador 5 (backfill) es "no hay
# job masivo" — no produce call site.
#
# Cuatro call sites, cuatro archivos:
#   - TRIGGER-1A: services.py::save_partial_plan_get_id (semana 1, chunked).
#   - TRIGGER-1B: routers/plans.py, rama `elif actual_user_id:` (plan completo
#     de una sola vez, tier gratis no-chunked) — cierra el gap de F1.
#   - TRIGGER-2: cron_tasks.py::_chunk_worker T1 (post-commit).
#   - TRIGGER-4: routers/user_data.py::api_patch_profile (PATCH de locale).
#
# Estos tests parsean el SOURCE de los archivos (no runtime mockeado): cada
# tooltip-anchor debe existir Y el dispatch debe aparecer DESPUÉS (por
# posición de string, O — donde el nombre de variable no basta, ver F2 abajo —
# por indentación) del punto de persist correspondiente — así un futuro
# refactor que mueva el dispatch ANTES del persist (dispatch sobre datos que
# aún no existen en DB, o que aún viven dentro de una transacción abierta)
# rompe el test antes de romper producción. `assert src.count(MARKER) == 1`
# en cada archivo (F2, nota menor del review) blinda contra que un futuro 2º
# call site en el mismo archivo (p.ej. Task 3) haga que `.index()` mida el
# call site equivocado sin que el test se entere.
# ================================================================================

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SERVICES_SRC_PATH = _BACKEND_ROOT / "services.py"
_CRON_TASKS_SRC_PATH = _BACKEND_ROOT / "cron_tasks.py"
_USER_DATA_SRC_PATH = _BACKEND_ROOT / "routers" / "user_data.py"
_PLANS_SRC_PATH = _BACKEND_ROOT / "routers" / "plans.py"

_SCHEDULE_IMPORT_MARKER = "from plan_display_i18n import schedule_plan_display_enrichment"


@pytest.fixture(scope="module")
def _services_src() -> str:
    return _SERVICES_SRC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def _cron_tasks_src() -> str:
    return _CRON_TASKS_SRC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def _user_data_src() -> str:
    return _USER_DATA_SRC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def _plans_src() -> str:
    return _PLANS_SRC_PATH.read_text(encoding="utf-8")


def test_trigger_1a_anchor_exists_after_semana_1_persist(_services_src):
    """Disparador 1A (camino CHUNKED): services.py, `save_partial_plan_get_id`.
    El anchor y el dispatch deben aparecer DESPUÉS del log `💾 [CHUNK] Plan
    parcial (semana 1) guardado` — ese log es la señal de que el INSERT de la
    semana 1 ya commiteó (`save_new_meal_plan_atomic` ya retornó el
    `plan_id`)."""
    assert "P1-PLAN-DISPLAY-I18N-TRIGGER-1A" in _services_src
    assert _services_src.count(_SCHEDULE_IMPORT_MARKER) == 1, (
        "más de un dispatch en services.py — este test mide el primero por "
        "posición, un 2º call site lo dejaría ciego (F2)."
    )
    persist_idx = _services_src.index("Plan parcial (semana 1) guardado")
    dispatch_idx = _services_src.index(_SCHEDULE_IMPORT_MARKER)
    assert dispatch_idx > persist_idx, (
        "el dispatch de trigger 1A aparece ANTES del persist de semana 1 en "
        "el archivo — el plan_id podría no existir aún en DB cuando el "
        "thread background intente leerlo."
    )


def test_trigger_1b_anchor_exists_after_nonchunked_persist(_plans_src):
    """[F1 fix-round 1] Disparador 1B (camino NO-chunked, tier gratis):
    routers/plans.py, rama `elif actual_user_id:` de `run_pipeline_and_format`
    (o equivalente). El plan gratis NO pasa por `save_partial_plan_get_id` ni
    por el chunk worker — sin este call site, un usuario de tier gratis que
    eligió `en-US` en el wizard generaba su plan completo y se quedaba en
    español PARA SIEMPRE (el PATCH de locale, disparador 4, solo rescata a
    quien CAMBIA de idioma después). `result["id"] = _pid` solo ejecuta una
    vez que la llamada SÍNCRONA a `_save_plan_and_track_background` ya
    retornó un id truthy (P1-NONCHUNKED-PERSIST-SYNC: persist inline, no
    fire-and-forget) — es la señal de persist confirmado."""
    assert "P1-PLAN-DISPLAY-I18N-TRIGGER-1B" in _plans_src
    assert _plans_src.count(_SCHEDULE_IMPORT_MARKER) == 1
    persist_idx = _plans_src.index('result["id"] = _pid')
    dispatch_idx = _plans_src.index(_SCHEDULE_IMPORT_MARKER)
    assert dispatch_idx > persist_idx, (
        "el dispatch de trigger 1B aparece ANTES de confirmar el persist "
        "no-chunked (`result[\"id\"] = _pid`) en el archivo."
    )


def test_trigger_2_anchor_present(_cron_tasks_src):
    assert "P1-PLAN-DISPLAY-I18N-TRIGGER-2" in _cron_tasks_src
    assert _cron_tasks_src.count(_SCHEDULE_IMPORT_MARKER) == 1


def test_trigger_2_dispatch_is_outside_t1_open_transaction_by_indentation(_cron_tasks_src):
    """[F2 fix-round 1] El test original de Task 2 anclaba contra
    `.index("_t1_persist_view")` (primera ocurrencia) — pero esa variable
    VIVE dentro del `with connection_pool.connection() as conn:` que abre la
    transacción T1 (línea ~30831) y su nombre sigue siendo una substring
    válida del archivo 61 líneas después de que la transacción cierra, así
    que `.index()` NO demuestra "post-commit": un dispatch reinsertado DENTRO
    del `FOR UPDATE` seguía pasando ese assert (review Finding 2).

    La propiedad que sí demuestra "fuera de la transacción abierta" es la
    INDENTACIÓN: `with connection_pool.connection() as conn:` (que abre T1)
    vive a 12 espacios; todo lo que cuelga de ese `with` (el `SELECT ... FOR
    UPDATE`, el UPDATE de `plan_data`, `_t1_persist_view`) está a indent >12.
    La primera línea tras el cierre del `with` vuelve a indent 12 — así que
    "el dispatch está a indent 12" ES "el dispatch no cuelga de ningún `with`
    abierto por T1".

    Verificación manual (documentada aquí per la petición del controller, no
    vive como fixture): tomé una copia en memoria del archivo con el bloque
    de dispatch REINDENTADO a 16 espacios e insertado justo después de
    `_t1_persist_view = dict(plan_data)` (dentro del `FOR UPDATE`, simulando
    la regresión que el review teme) y confirmé dos cosas — (1) el test viejo
    basado en `.index("_t1_persist_view")` seguía en VERDE contra esa copia
    rota (reproduce el gap del review), y (2) este assert de indentación ==
    12 SÍ falla contra la misma copia (mide indent=16, correctamente rojo).
    Script de la verificación: `/tmp/_verify_f2.py` (ad-hoc, no committeado).
    """
    match = re.search(r"^( *)if _p1_i18n_new_day_indices:\s*$", _cron_tasks_src, re.MULTILINE)
    assert match, "no se encontró la línea `if _p1_i18n_new_day_indices:` en el archivo"
    indent = len(match.group(1))
    assert indent == 12, (
        f"el dispatch del trigger 2 vive a indent {indent}, se esperaba 12 "
        "(el mismo nivel que `with connection_pool.connection() as conn:` "
        "que abre la transacción T1) — si cambió intencionalmente, verificar "
        "manualmente que el dispatch sigue fuera de los `with` de T1 antes "
        "de actualizar este assert."
    )


def test_trigger_2_only_dispatches_on_new_merge_not_idempotent_replay(_cron_tasks_src):
    """El día indices solo se puebla en la rama 'Merge normal' (días NUEVOS
    persistidos); la rama `chunk_already_merged` (replay idempotente, sin días
    nuevos) debe dejarlo en `None` — cubre el matiz del brief 'cubre las que
    persisten días nuevos'."""
    default_idx = _cron_tasks_src.index("_p1_i18n_new_day_indices = None")
    already_merged_idx = _cron_tasks_src.index("if chunk_already_merged:")
    populate_idx = _cron_tasks_src.index(
        "_p1_i18n_new_day_indices = list(range(prior_count, len(merged_days)))"
    )
    assert default_idx < already_merged_idx < populate_idx, (
        "el default None debe preceder al if/else, y la asignación real de "
        "day_indices debe vivir en la rama 'Merge normal' (después del if)."
    )


def test_trigger_2_locale_lookup_is_targeted_select_not_get_user_profile(_cron_tasks_src):
    """[F7 fix-round 1] `_chunk_worker` NO debe usar `get_user_profile` (SELECT
    * + posible UPDATE lateral de downgrade de tier, `db_profiles.py`) para
    resolver el locale DENTRO del bloque de dispatch del trigger 2 — un
    SELECT dirigido de una sola columna por PK. Scoped al bloque (no al
    archivo entero): `cron_tasks.py` tiene un `get_user_profile` preexistente
    y NO relacionado (`_chunk_worker`, ~línea 33685, otra feature) que un
    assert global marcaría como falso positivo."""
    assert "SELECT locale FROM user_profiles WHERE id = %s" in _cron_tasks_src
    # `rindex`: el anchor aparece DOS veces en el archivo (el default `= None`
    # arriba del if/else, y el bloque de dispatch real) — la ÚLTIMA es la que
    # contiene el SELECT/lookup real.
    trigger_block_start = _cron_tasks_src.rindex("P1-PLAN-DISPLAY-I18N-TRIGGER-2")
    trigger_block = _cron_tasks_src[trigger_block_start:trigger_block_start + 2500]
    assert "SELECT locale FROM user_profiles" in trigger_block, (
        "sanity: el bloque de dispatch (última ocurrencia del anchor) no "
        "contiene el SELECT dirigido esperado — revisar el offset de la ventana."
    )
    # Se busca la LLAMADA (`get_user_profile(`), no la palabra suelta — el
    # propio comentario explicativo de F7 la menciona en prosa (sin paréntesis
    # inmediato) para documentar qué NO usar, y eso no debe contar como fallo.
    assert "get_user_profile(" not in trigger_block, (
        "get_user_profile() volvió a colarse en el bloque de dispatch del "
        "trigger 2 — debe seguir usando el SELECT dirigido de locale (F7)."
    )


def test_trigger_4_anchor_exists_after_locale_update_commits(_user_data_src):
    """Disparador 4 (PATCH de locale en Configuración): routers/user_data.py,
    `api_patch_profile`. El dispatch debe ir DESPUÉS de que el UPDATE de
    `user_profiles` haya sido confirmado (`updated = await
    asyncio.to_thread(_patch)` + el guard 404 si no afectó filas)."""
    assert "P1-PLAN-DISPLAY-I18N-TRIGGER-4" in _user_data_src
    assert "P1-PLAN-DISPLAY-I18N-TRIGGER-3" not in _user_data_src, (
        "el anchor viejo (mal numerado, F4) sigue vivo — la spec reserva "
        "TRIGGER-3 para los mutadores de Task 3."
    )
    assert _user_data_src.count(_SCHEDULE_IMPORT_MARKER) == 1
    persist_idx = _user_data_src.index("updated = await asyncio.to_thread(_patch)")
    dispatch_idx = _user_data_src.index(_SCHEDULE_IMPORT_MARKER)
    assert dispatch_idx > persist_idx, (
        "el dispatch de trigger 4 aparece ANTES de confirmar el UPDATE de "
        "locale — podría despachar sobre un PATCH que en realidad falló."
    )


def test_trigger_4_es_do_never_reaches_dispatch_code(_user_data_src):
    """es-DO es el locale base (0 bytes de `_display`, P1-I18N-DASHBOARD) — el
    gate `!= "es-DO"` debe preceder cualquier lectura del plan activo."""
    gate_idx = _user_data_src.index('_p1_i18n_new_locale != "es-DO"')
    query_idx = _user_data_src.index("_p1_i18n_active_plan_display_edges")
    assert gate_idx < query_idx


def test_trigger_4_select_is_o1_projection_not_full_plan_data(_user_data_src):
    """[F3 fix-round 1] El SELECT del plan activo debe proyectar SOLO las dos
    claves `_display` (primer/último día) vía jsonb path — nunca bajar la
    columna `plan_data` completa (puede ser cientos de KB-MB con 30 días de
    recetas expandidas) solo para mirar si dos claves existen."""
    assert "plan_data->'days'->0->'meals'->0->'_display'" in _user_data_src
    assert "plan_data->'days'->-1->'meals'->0->'_display'" in _user_data_src
    # El patrón viejo (bajar plan_data entero + json.loads defensivo) no debe
    # sobrevivir en esta zona del archivo.
    trigger_block_start = _user_data_src.index("P1-PLAN-DISPLAY-I18N-TRIGGER-4")
    trigger_block = _user_data_src[trigger_block_start:trigger_block_start + 3000]
    assert "SELECT id::text AS id, plan_data FROM meal_plans" not in trigger_block, (
        "el SELECT del trigger 4 volvió a bajar plan_data completo (F3)."
    )


# ------------------------------------------------------------------
# Funcional: el gate del PATCH (disparador 4) — mock de
# `schedule_plan_display_enrichment` + mock del SELECT proyectado del plan
# activo (`{id, disp_first, disp_last}`, ver F3/F5).
# ------------------------------------------------------------------


def _patch_profile_locale(monkeypatch, locale, select_result, uid="user-trig4"):
    """Ejecuta el PATCH /profile real (función del router, sin TestClient —
    mismo patrón que test_p1_tracking_finish_sensitive_guard.py) con el UPDATE
    de `user_profiles` mockeado a éxito y el SELECT proyectado del plan activo
    devolviendo `select_result` (dict `{id, disp_first, disp_last}`, o `None`
    si no hay plan)."""
    monkeypatch.setattr(db_module, "execute_sql_write", lambda *a, **kw: [{"id": uid}])
    monkeypatch.setattr(db_module, "execute_sql_query", lambda *a, **kw: select_result)
    calls = []
    monkeypatch.setattr(
        pdi, "schedule_plan_display_enrichment",
        lambda *a, **kw: calls.append((a, kw)),
    )
    body = ProfilePatchBody(fields={"locale": locale})
    result = asyncio.run(api_patch_profile(body=body, verified_user_id=uid))
    return result, calls


def test_patch_profile_locale_en_us_dispatches_when_plan_not_enriched(monkeypatch):
    select_result = {"id": "plan-trig4-a", "disp_first": None, "disp_last": None}
    result, calls = _patch_profile_locale(monkeypatch, "en-US", select_result)

    assert result == {"success": True}
    assert calls == [(("plan-trig4-a", "user-trig4", "en-US"), {})]


def test_patch_profile_locale_es_do_never_dispatches(monkeypatch):
    select_result = {"id": "plan-trig4-b", "disp_first": None, "disp_last": None}
    result, calls = _patch_profile_locale(monkeypatch, "es-DO", select_result)

    assert result == {"success": True}
    assert calls == []


def test_patch_profile_locale_already_has_display_both_edges_does_not_redispatch(monkeypatch):
    """Check barato: PRIMER y ÚLTIMO día YA tienen `_display['en-US']` -> no
    re-despachar (el brief pide evitar redundancia)."""
    select_result = {
        "id": "plan-trig4-c",
        "disp_first": {"en-US": {"name": "Beans"}},
        "disp_last": {"en-US": {"name": "Rice"}},
    }
    result, calls = _patch_profile_locale(monkeypatch, "en-US", select_result)

    assert result == {"success": True}
    assert calls == []


def test_patch_profile_locale_partial_enrichment_first_only_still_dispatches(monkeypatch):
    """[F5 fix-round 1] El motor troceo por lotes permite recuperación
    parcial — si el lote del primer día escribió pero el del último NO
    (chunk enriquecido, semanas siguientes aún no), el check debe seguir
    viendo el plan como INCOMPLETO y despachar. Mirar solo el primer día
    congelaría este plan como 'ya enriquecido' para siempre, sin ningún
    disparador que lo complete."""
    select_result = {
        "id": "plan-trig4-d",
        "disp_first": {"en-US": {"name": "Beans"}},
        "disp_last": None,
    }
    result, calls = _patch_profile_locale(monkeypatch, "en-US", select_result)

    assert result == {"success": True}
    assert calls == [(("plan-trig4-d", "user-trig4", "en-US"), {})]


def test_patch_profile_locale_partial_enrichment_last_only_still_dispatches(monkeypatch):
    """Espejo del test anterior: solo el ÚLTIMO día enriquecido (p.ej. un
    enrich manual disparado sobre un day_indices específico) tampoco cuenta
    como 'completo'."""
    select_result = {
        "id": "plan-trig4-e",
        "disp_first": None,
        "disp_last": {"en-US": {"name": "Rice"}},
    }
    result, calls = _patch_profile_locale(monkeypatch, "en-US", select_result)

    assert result == {"success": True}
    assert calls == [(("plan-trig4-e", "user-trig4", "en-US"), {})]


def test_patch_profile_locale_no_active_plan_is_noop(monkeypatch):
    """Sin plan activo (usuario recién registrado, aún no generó nada): el
    dispatch se salta limpio, sin excepción."""
    result, calls = _patch_profile_locale(monkeypatch, "en-US", None)

    assert result == {"success": True}
    assert calls == []


def test_patch_profile_locale_select_failure_is_best_effort(monkeypatch):
    """Si el SELECT proyectado lanza, el PATCH igual debe devolver
    success:true — el cambio de idioma NUNCA puede fallar por esto."""
    def _boom(*a, **kw):
        raise RuntimeError("select roto")

    monkeypatch.setattr(db_module, "execute_sql_write", lambda *a, **kw: [{"id": "user-trig4"}])
    monkeypatch.setattr(db_module, "execute_sql_query", _boom)

    body = ProfilePatchBody(fields={"locale": "en-US"})
    result = asyncio.run(api_patch_profile(body=body, verified_user_id="user-trig4"))

    assert result == {"success": True}


def test_patch_profile_locale_schedule_failure_is_best_effort(monkeypatch):
    """[F6 fix-round 1] El test anterior (heredado de Task 2) SOLO ejercitaba
    el fallo del SELECT — la excepción ocurría ANTES de alcanzar
    `schedule_plan_display_enrichment`, así que esa rama del try/except NUNCA
    se cubría (review Finding 6). Este test hace que el SELECT tenga ÉXITO
    (plan sin enriquecer, alcanzable) y sea `schedule_plan_display_enrichment`
    el que lance — la rama que de verdad importa, porque es la que corre con
    el plan ya leído y a punto de despachar el thread."""
    select_result = {"id": "plan-trig4-f", "disp_first": None, "disp_last": None}
    monkeypatch.setattr(db_module, "execute_sql_write", lambda *a, **kw: [{"id": "user-trig4"}])
    monkeypatch.setattr(db_module, "execute_sql_query", lambda *a, **kw: select_result)

    def _boom_schedule(*a, **kw):
        raise RuntimeError("schedule roto")

    monkeypatch.setattr(pdi, "schedule_plan_display_enrichment", _boom_schedule)

    body = ProfilePatchBody(fields={"locale": "en-US"})
    result = asyncio.run(api_patch_profile(body=body, verified_user_id="user-trig4"))

    assert result == {"success": True}
