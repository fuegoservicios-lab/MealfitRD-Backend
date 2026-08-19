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
import json
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
    # [Task 3 · P1-PLAN-DISPLAY-I18N-MUTATOR-swap/-regenday + Fix round 1 ·
    # -recipeexpand (F5)] `routers/plans.py` ganó TRES call sites legítimos más
    # (swap-persist + regenerate-day + recipe/expand) tras Task 2 — el count()==1
    # original ya no aplica. La invariante que SÍ importa (la que F2 protegía) es que
    # `.index()` siga midiendo el de TRIGGER-1B: su import es el ÚNICO con el alias
    # `as _p1_i18n_schedule` (los mutadores de Task 3 importan SIN alias) Y vive ANTES
    # (por posición de línea) que los tres — así que la PRIMERA ocurrencia sigue siendo
    # la correcta. Si un futuro 5º call site se cuela ANTES de TRIGGER-1B en el archivo,
    # este assert de count() lo atrapa (deja de ser 4) sin que nadie tenga que recordar
    # mirar.
    assert _plans_src.count(_SCHEDULE_IMPORT_MARKER) == 4, (
        "el nº de call sites de `schedule_plan_display_enrichment` en routers/plans.py "
        "cambió — si es un 5º disparador legítimo, actualizar este count Y verificar que "
        "`.index()` abajo siga midiendo TRIGGER-1B (no el nuevo)."
    )
    first_import_idx = _plans_src.index(_SCHEDULE_IMPORT_MARKER)
    first_import_line_end = _plans_src.index("\n", first_import_idx)
    assert "as _p1_i18n_schedule" in _plans_src[first_import_idx:first_import_line_end], (
        "la PRIMERA ocurrencia del import ya no es la de TRIGGER-1B (identificada por su "
        "alias `as _p1_i18n_schedule`, único entre los 3 call sites) — un nuevo call site "
        "se coló ANTES en el archivo y el `.index()` de abajo mide el call site equivocado."
    )
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


# ================================================================================
# SECCIÓN: MUTADORES (Task 3 + Fix round 1) — DELETE-on-write + re-enrich en los
# 4 mutadores de meal_plans.plan_data.
#
# tooltip-anchors: `P1-PLAN-DISPLAY-I18N-MUTATOR-swap` (routers/plans.py,
# `/swap-meal/persist`), `P1-PLAN-DISPLAY-I18N-MUTATOR-regenday` (routers/plans.py,
# `/regenerate-day`), `P1-PLAN-DISPLAY-I18N-MUTATOR-chatmod` (tools.py,
# `execute_modify_single_meal`), `P1-PLAN-DISPLAY-I18N-MUTATOR-recipeexpand`
# (routers/plans.py, `/recipe/expand` — F5 del review de fix round 1).
#
# Spec (sección "Invalidación: DELETE-on-write"): toda mutación de un meal BORRA su
# `_display` completo en el MISMO write. Los tests parser-based anclan POSICIÓN, no
# solo existencia (la lección del F2 de Task 2: un `.index()` de la primera
# ocurrencia de un nombre de variable reusado en dos sitios mide el call site
# equivocado sin que el test se entere).
#
# [Fix round 1] F1 (CRITICAL): regen-day popeaba SOLO los slots regenerados — falso
# en este endpoint, porque el rebalance/refine/relevel de DÍA COMPLETO re-cuantiza
# los gramos de TODOS los meals del día, conservados incluidos (ver el comment del
# pop en producción, que cita las funciones exactas). El pop volvió a ser TOTAL
# (todos los meals de `new_meals`, sin condición) — la lista paralela
# `_p1_i18n_regenerated_flags` desapareció por completo (con F4, que dependía de
# ella). F9 (MINOR): las ventanas de tamaño mágico (`+900`, `+1100`, `+1200`, `+400`)
# eran exactamente la fragilidad que produjo F3 (una inserción de producción movió un
# offset y rompió un guard ajeno) — reemplazadas por ventanas acotadas entre DOS
# anclas de texto reales (el marker del despacho y el siguiente comment/línea única
# que le sigue en producción), nunca un número de caracteres a mano.
# ================================================================================

_TOOLS_SRC_PATH = _BACKEND_ROOT / "tools.py"


@pytest.fixture(scope="module")
def _tools_src() -> str:
    return _TOOLS_SRC_PATH.read_text(encoding="utf-8")


def _bounded_block(src: str, start_marker: str, end_marker: str) -> str:
    """[Fix round 1 · F9] Extrae `src[start:end]` entre DOS anclas de texto reales
    (nunca un número de caracteres a mano) — `end_marker` es la línea que sigue al
    bloque en el source de producción, así que el slice crece/encoge con el bloque
    en vez de truncarlo si un futuro comment se alarga."""
    start = src.index(start_marker)
    end = src.index(end_marker, start)
    assert end > start, f"{end_marker!r} aparece ANTES que {start_marker!r} — anclas invertidas."
    return src[start:end]


# ------------------------------------------------------------------
# 1. /swap-meal/persist (routers/plans.py) — parser-based + funcional (F12).
# ------------------------------------------------------------------

_SWAP_DISPATCH_START = "P1-PLAN-DISPLAY-I18N-MUTATOR-swap] Despacho best-effort post-persist"
_SWAP_DISPATCH_END = "Señal de gusto aprendido: el usuario"  # [P1-NEXT-LEVEL-BATCH] siguiente bloque


def test_mutator_swap_anchor_present(_plans_src):
    assert "P1-PLAN-DISPLAY-I18N-MUTATOR-swap" in _plans_src


def test_mutator_swap_pop_is_inside_swap_mutator_callback(_plans_src):
    """El pop de `_display` debe vivir DENTRO de `_swap_mutator` — ANTES de la
    llamada a `update_plan_data_atomic` que lo invoca como callback (fuera de la
    función, el pop correría sobre nada o sobre el objeto equivocado)."""
    cb_start = _plans_src.index("def _swap_mutator(")
    pop_idx = _plans_src.index('meals[meal_index].pop("_display", None)')
    call_idx = _plans_src.index(
        "result = update_plan_data_atomic(\n            plan_id,\n            _swap_mutator,"
    )
    assert cb_start < pop_idx < call_idx, (
        "el pop de `_display` en swap-persist debe estar DENTRO de `_swap_mutator` "
        "(entre su `def` y la llamada a `update_plan_data_atomic(plan_id, "
        "_swap_mutator, ...)` que cierra el scope del callback)."
    )


def test_mutator_swap_pop_is_after_meal_assignment_not_before(_plans_src):
    """El pop debe ir DESPUÉS de `meals[meal_index] = new_meal` (el punto donde el
    meal de reemplazo se coloca) — nunca antes, o no habría nada que popear."""
    assign_idx = _plans_src.index("meals[meal_index] = new_meal")
    pop_idx = _plans_src.index('meals[meal_index].pop("_display", None)')
    assert assign_idx < pop_idx


def test_mutator_swap_pop_comment_documents_client_spoofing_vector(_plans_src):
    """[Fix round 1 · F12] El comment del pop de swap debe decir la VERDAD: en
    swap-persist `new_meal` sale del BODY de la request y se persiste
    verbatim — un cliente PUEDE colar un `_display` arbitrario, y el pop es lo
    ÚNICO que lo impide. La redacción vieja decía lo contrario ("no una defensa
    contra un vector conocido hoy") — un futuro lector podía borrar el pop
    creyéndolo decorativo."""
    comment_start = _plans_src.index("P1-PLAN-DISPLAY-I18N-MUTATOR-swap] DELETE-on-write")
    pop_idx = _plans_src.index('meals[meal_index].pop("_display", None)')
    comment_block = _plans_src[comment_start:pop_idx]
    assert "body" in comment_block.lower(), (
        "el comment del pop de swap ya no menciona que `new_meal` viene del BODY "
        "de la request (el vector de spoofing que F12 pidió documentar)."
    )
    assert "no lo borres" in comment_block.lower() or "no es decorativo" in comment_block.lower(), (
        "el comment del pop de swap debe advertir explícitamente contra borrarlo "
        "por 'decorativo' — esa fue la redacción incorrecta que F12 corrigió."
    )


def test_mutator_swap_dispatch_after_atomic_result_confirmed(_plans_src):
    """El despacho de `schedule_plan_display_enrichment` debe vivir DESPUÉS de
    confirmar `result` (post-persist, FUERA del lock) — nunca antes del 404 guard,
    que podría despachar sobre un persist que en realidad falló (row desaparecida)."""
    result_guard_idx = _plans_src.index(
        'if not result:\n            # Row desapareció entre el SELECT inicial y el FOR UPDATE'
    )
    dispatch_idx = _plans_src.index(_SWAP_DISPATCH_START)
    assert dispatch_idx > result_guard_idx


def test_mutator_swap_dispatch_uses_day_index_scoped_batch(_plans_src):
    """El despacho acota `day_indices` a los días REALMENTE tocados — el swapeado
    (`day_index`) [Fix round 2] UNIDO a los días colaterales que
    `apply_update_macro_engine` (plan-wide) reportó vía `_swap_macroengine_touched_days`
    — nunca el plan entero (recuperación parcial/costo acotado)."""
    dispatch_block = _bounded_block(_plans_src, _SWAP_DISPATCH_START, _SWAP_DISPATCH_END)
    assert "schedule_plan_display_enrichment(" in dispatch_block
    assert "day_indices=sorted(_p1i18n_days_sw)" in dispatch_block
    assert "_p1i18n_days_sw.add(day_index)" in dispatch_block
    assert "_p1i18n_days_sw = set(_swap_macroengine_touched_days)" in dispatch_block


def test_mutator_swap_dispatch_uses_should_enrich_locale_helper(_plans_src):
    """[Fix round 1 · F10] El gate de locale usa el helper SSOT exportado por el
    motor (`plan_display_i18n.should_enrich_locale`), no un literal `!= "es-DO"` a
    mano en el call site."""
    dispatch_block = _bounded_block(_plans_src, _SWAP_DISPATCH_START, _SWAP_DISPATCH_END)
    assert "should_enrich_locale" in dispatch_block
    assert '!= "es-DO"' not in dispatch_block


# ------------------------------------------------------------------
# Funcional (F12): un `new_meal` del BODY con `_display` inyectado por el cliente
# NO sobrevive al persist. Invoca `api_swap_meal_persist` de VERDAD (mismo criterio
# que el funcional de chat-modify: no hay forma de aislar `_swap_mutator` sin
# ejecutar el handler completo), mockeando DB (`db_core.execute_sql_query` para el
# SELECT de ownership, `db.get_user_profile`, `db_plans.update_plan_data_atomic`
# capturado igual que el `engine` de la sección MOTOR). Los ~40 bloques
# finalize/closer/motor/micros del mutator son best-effort (try/except con
# fallback), así que corren en modo no-op sin DB real — mismo comportamiento ya
# verificado para chat-modify.
# ------------------------------------------------------------------


@pytest.fixture
def _swap_engine(monkeypatch):
    import routers.plans as plans_module
    import db_core as db_core_module
    import db as db_module_local
    import db_plans as db_plans_module

    state = {"plan_data": None, "atomic_calls": []}

    def _fake_execute_sql_query_sw(sql, params=None, **kwargs):
        if "SELECT id FROM meal_plans" in sql:
            return {"id": (params[0] if params else "plan-swap-1")}
        return None

    monkeypatch.setattr(db_core_module, "execute_sql_query", _fake_execute_sql_query_sw)
    monkeypatch.setattr(db_module_local, "get_user_profile", lambda uid: {"health_profile": {}})

    def _fake_atomic_sw(plan_id, mutator, user_id=None, **kwargs):
        state["atomic_calls"].append({"plan_id": plan_id, "user_id": user_id})
        result = mutator(state["plan_data"])
        if isinstance(result, dict):
            state["plan_data"] = result
        return state["plan_data"]

    monkeypatch.setattr(db_plans_module, "update_plan_data_atomic", _fake_atomic_sw)
    monkeypatch.setenv("MEALFIT_SWAP_PERSIST_CLINICAL_GUARD", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_RECOMPUTE_MICROS", "false")
    monkeypatch.setenv("MEALFIT_SWAP_PERSIST_DAY_BAND", "false")

    return state, plans_module


def test_swap_functional_client_injected_display_does_not_survive_persist(_swap_engine):
    state, plans_module = _swap_engine
    old_meal = {
        "meal": "Almuerzo", "name": "Pollo guisado", "desc": "Pollo con arroz.",
        "cals": 500, "protein": 30, "carbs": 50, "fats": 15,
        "ingredients": ["150 g de pechuga de pollo", "1 taza de arroz"],
        "recipe": ["Sofreir el pollo.", "Servir con arroz."],
    }
    state["plan_data"] = {"days": [{"day": 1, "meals": [old_meal]}]}

    body = {
        "day_index": 0,
        "meal_index": 0,
        "new_meal": {
            "name": "Pescado al horno",
            "desc": "Pescado con vegetales.",
            "cals": 480, "protein": 32, "carbs": 40, "fats": 14,
            "ingredients": ["200 g de pescado", "1 taza de vegetales"],
            "recipe": ["Hornear el pescado.", "Servir con vegetales."],
            # [Fix round 1 · F12] inyectado por un cliente adversario/buggy —
            # NUNCA debe sobrevivir al persist.
            "_display": {
                "en-US": {
                    "name": "SPOOFED — should never persist",
                    "description": "attacker-controlled",
                    "recipe": ["fake step"],
                    "ingredients": ["fake ingredient"],
                }
            },
        },
    }

    result = plans_module.api_swap_meal_persist(
        plan_id="plan-swap-1", data=body, verified_user_id="user-swap-1",
    )

    assert result == {"success": True}
    persisted_meal = state["plan_data"]["days"][0]["meals"][0]
    assert persisted_meal["name"] == "Pescado al horno"
    assert "_display" not in persisted_meal, (
        "el meal PERSISTIDO conservó el `_display` inyectado por el cliente en el "
        "body de la request — el vector de spoofing display-only (F12) sigue "
        "abierto en swap-persist."
    )


# ------------------------------------------------------------------
# 2. /regenerate-day (routers/plans.py) — parser-based.
# ------------------------------------------------------------------

_REGENDAY_DISPATCH_START = "P1-PLAN-DISPLAY-I18N-MUTATOR-regenday] Despacho best-effort post-persist"
_REGENDAY_DISPATCH_END = "Cuota: 1 crédito por día completo"


def test_mutator_regenday_anchor_present(_plans_src):
    assert "P1-PLAN-DISPLAY-I18N-MUTATOR-regenday" in _plans_src


def test_mutator_regenday_pop_is_inside_day_mutator_callback(_plans_src):
    """El pop debe vivir DENTRO de `_day_mutator` — ANTES de la llamada a
    `update_plan_data_atomic(plan_id, _day_mutator, ...)` que cierra el callback."""
    cb_start = _plans_src.index("def _day_mutator(")
    pop_idx = _plans_src.index('_nm_disp.pop("_display", None)')
    call_idx = _plans_src.index(
        "result = update_plan_data_atomic(plan_id, _day_mutator, user_id=verified_user_id)"
    )
    assert cb_start < pop_idx < call_idx, (
        "el pop de `_display` en regenerate-day debe estar DENTRO de `_day_mutator`."
    )


def test_mutator_regenday_pop_is_after_meals_assignment_not_before(_plans_src):
    """El pop debe ir DESPUÉS de `_day['meals'] = new_meals` (donde el día
    reemplazado se coloca) — nunca antes."""
    assign_idx = _plans_src.index('_day["meals"] = new_meals')
    pop_idx = _plans_src.index('_nm_disp.pop("_display", None)')
    assert assign_idx < pop_idx


def test_mutator_regenday_pop_is_unconditional_over_all_new_meals(_plans_src):
    """[Fix round 1 · F1] El pop debe ser TOTAL — TODOS los meals de `new_meals`,
    sin condicionar por "fue regenerado" — porque el rebalance/refine/relevel de
    DÍA COMPLETO (que corre ARRIBA de este pop) re-cuantiza los gramos de
    ingredientes de TODOS los meals del día, conservados incluidos. Popear solo
    los regenerados (v1, revertida) dejaba a un slot CONSERVADO pero re-cuantizado
    con su `_display` mintiendo la cantidad vieja."""
    loop_idx = _plans_src.index("for _nm_disp in new_meals:")
    pop_idx = _plans_src.index('_nm_disp.pop("_display", None)')
    loop_block = _plans_src[loop_idx:pop_idx + len('_nm_disp.pop("_display", None)')]
    assert loop_idx < pop_idx
    # [F1/F4] La lista paralela debe desaparecer como CÓDIGO ACTIVO — la prosa del
    # comment (que cita el nombre viejo a propósito, para que un futuro
    # "optimizador" entienda qué NO reintroducir) queda exenta: se despoja cada
    # línea de su comentario `#...` antes de buscar el identificador.
    _src_no_comments = re.sub(r"#[^\n]*", "", _plans_src)
    assert "_p1_i18n_regenerated_flags" not in _src_no_comments, (
        "F1/F4: la lista paralela `_p1_i18n_regenerated_flags` volvió a aparecer "
        "como CÓDIGO ACTIVO (fuera de un comentario) — el pop condicional volvió."
    )
    # El único `if` permitido en el loop es el guard de tipo (`isinstance`) — NO un
    # gate sobre "fue regenerado".
    assert "isinstance(_nm_disp, dict)" in loop_block
    assert "regenerado" not in loop_block.lower() and "was_regen" not in loop_block.lower()


def test_mutator_regenday_dispatch_after_atomic_result_confirmed(_plans_src):
    result_guard_idx = _plans_src.index(
        'result = update_plan_data_atomic(plan_id, _day_mutator, user_id=verified_user_id)'
    )
    dispatch_idx = _plans_src.index(_REGENDAY_DISPATCH_START)
    assert dispatch_idx > result_guard_idx


def test_mutator_regenday_dispatch_uses_day_index_scoped_batch(_plans_src):
    dispatch_block = _bounded_block(_plans_src, _REGENDAY_DISPATCH_START, _REGENDAY_DISPATCH_END)
    assert "schedule_plan_display_enrichment(" in dispatch_block
    assert "day_indices=[day_index]" in dispatch_block


def test_mutator_regenday_dispatch_uses_should_enrich_locale_helper(_plans_src):
    """[Fix round 1 · F10]"""
    dispatch_block = _bounded_block(_plans_src, _REGENDAY_DISPATCH_START, _REGENDAY_DISPATCH_END)
    assert "should_enrich_locale" in dispatch_block
    assert '!= "es-DO"' not in dispatch_block


def test_mutator_regenday_locale_lookup_is_targeted_select(_plans_src):
    """Mismo patrón barato que TRIGGER-2 (cron_tasks.py): SELECT dirigido de UNA
    columna como FALLBACK — [Fix round 1 · F6] el holder `_p1i18n_locale_rd_holder`
    reusa el `get_user_profile` que el handler YA hace (retarget/clinical-parity) si
    alguno corrió; el SELECT solo corre si ninguno rellenó el holder."""
    dispatch_block = _bounded_block(_plans_src, _REGENDAY_DISPATCH_START, _REGENDAY_DISPATCH_END)
    assert "SELECT locale FROM user_profiles WHERE id = %s" in dispatch_block
    assert "_p1i18n_locale_rd_holder[0]" in dispatch_block


def test_mutator_regenday_locale_holder_reused_by_retarget_and_clinical_blocks(_plans_src):
    """[Fix round 1 · F6] El holder debe rellenarse desde AMBOS `get_user_profile`
    condicionales existentes (retarget + clinical-parity), no solo uno — si solo
    uno lo rellena, el otro sigue corriendo un SELECT redundante en su ventana."""
    assert _plans_src.count("_p1i18n_locale_rd_holder[0] = _full_profile_rd.get(\"locale\")") == 1
    assert _plans_src.count("_p1i18n_locale_rd_holder[0] = _full_profile_clin_rd.get(\"locale\")") == 1


# ------------------------------------------------------------------
# 3. tools.execute_modify_single_meal — parser-based.
# ------------------------------------------------------------------

_CHATMOD_DISPATCH_START = "P1-PLAN-DISPLAY-I18N-MUTATOR-chatmod] Despacho best-effort post-persist"
_CHATMOD_DISPATCH_END = "Señal de gusto aprendido del reemplazo"  # [P1-NEXT-LEVEL-BATCH] siguiente bloque


def test_mutator_chatmod_anchor_present(_tools_src):
    assert "P1-PLAN-DISPLAY-I18N-MUTATOR-chatmod" in _tools_src


def test_mutator_chatmod_pop_is_inside_apply_meal_modification_callback(_tools_src):
    """El pop debe vivir DENTRO de `_apply_meal_modification` — ANTES de la llamada
    a `update_plan_data_atomic(plan_id, _apply_meal_modification, ...)`."""
    cb_start = _tools_src.index("def _apply_meal_modification(")
    pop_idx = _tools_src.index('meals_fresh[target_idx_fresh].pop("_display", None)')
    call_idx = _tools_src.index(
        "merged_plan_data = update_plan_data_atomic(\n            plan_id, _apply_meal_modification, user_id=user_id\n        )"
    )
    assert cb_start < pop_idx < call_idx, (
        "el pop de `_display` en chat-modify debe estar DENTRO de "
        "`_apply_meal_modification` (entre su `def` y la llamada a "
        "`update_plan_data_atomic(plan_id, _apply_meal_modification, ...)`)."
    )


def test_mutator_chatmod_pop_is_after_meal_assignment_not_before(_tools_src):
    """El pop debe ir DESPUÉS de `meals_fresh[target_idx_fresh] = new_meal_data`
    (donde el meal de reemplazo se coloca en la copia FRESH) — nunca antes."""
    assign_idx = _tools_src.index("meals_fresh[target_idx_fresh] = new_meal_data")
    pop_idx = _tools_src.index('meals_fresh[target_idx_fresh].pop("_display", None)')
    assert assign_idx < pop_idx


def test_mutator_chatmod_pop_is_after_return_false_guards(_tools_src):
    """El pop opera sobre `meals_fresh[target_idx_fresh]` — debe vivir DESPUÉS de
    los dos `return False` que abortan el callback si el día/meal no se re-localiza
    tras el lock (si viviera antes, `target_idx_fresh` podría ser `None`/stale). El
    span se acota a `[cb_start, pop_idx)` — un `.rindex()` sobre una ventana fija
    puede aterrizar en un `return False` de OTRA función más abajo en el archivo."""
    cb_start = _tools_src.index("def _apply_meal_modification(")
    pop_idx = _tools_src.index('meals_fresh[target_idx_fresh].pop("_display", None)')
    cb_body_before_pop = _tools_src[cb_start:pop_idx]
    assert "return False" in cb_body_before_pop, (
        "sanity: no se encontró ningún `return False` entre el `def` del callback "
        "y el pop — refactor inesperado de los guards de re-localización."
    )
    guard_idx = cb_body_before_pop.rindex("return False")
    assert cb_start + guard_idx < pop_idx


def test_mutator_chatmod_dispatch_after_persist_confirmed(_tools_src):
    """El despacho vive DENTRO del `if merged_plan_data:` (post-persist confirmado),
    no antes — nunca despachar sobre un persist que en realidad falló."""
    persist_check_idx = _tools_src.index("if merged_plan_data:")
    dispatch_idx = _tools_src.index(_CHATMOD_DISPATCH_START)
    assert dispatch_idx > persist_check_idx


def test_mutator_chatmod_dispatch_uses_day_index_holder_not_prelock_var(_tools_src):
    """[Fix round 1 · F7] El despacho debe usar `_dispatch_day_index[0]` (holder
    rellenado DENTRO del callback, bajo el lock) — NO un índice capturado en un
    loop PRE-lock potencialmente stale. [Fix round 2] UNIDO a
    `_macroengine_touched_days` (días colaterales que `apply_update_macro_engine`,
    plan-wide, reportó)."""
    dispatch_block = _bounded_block(_tools_src, _CHATMOD_DISPATCH_START, _CHATMOD_DISPATCH_END)
    assert "schedule_plan_display_enrichment(" in dispatch_block
    assert "day_indices=sorted(_p1i18n_days_cm)" in dispatch_block
    assert "_p1i18n_days_cm.add(_dispatch_day_index[0])" in dispatch_block
    assert "_p1i18n_days_cm = set(_macroengine_touched_days)" in dispatch_block
    assert "target_day_index" not in dispatch_block


def test_mutator_chatmod_dispatch_uses_should_enrich_locale_helper(_tools_src):
    """[Fix round 1 · F10]"""
    dispatch_block = _bounded_block(_tools_src, _CHATMOD_DISPATCH_START, _CHATMOD_DISPATCH_END)
    assert "should_enrich_locale" in dispatch_block
    assert '!= "es-DO"' not in dispatch_block


def test_mutator_chatmod_dispatch_day_index_captured_inside_callback_under_lock(_tools_src):
    """[Fix round 1 · F7] `_dispatch_day_index[0]` debe capturarse DENTRO de
    `_apply_meal_modification` (bajo el lock), en el MISMO loop que re-localiza
    `target_day_fresh` — NO en un loop pre-lock (el propio callback ya re-localiza
    todo por esa razón: las posiciones pueden quedar stale entre la lectura y el
    lock)."""
    cb_start = _tools_src.index("def _apply_meal_modification(")
    loop_idx = _tools_src.index("for _di_p1i18n, d in enumerate(days_fresh):")
    capture_idx = _tools_src.index("_dispatch_day_index[0] = _di_p1i18n")
    call_idx = _tools_src.index(
        "merged_plan_data = update_plan_data_atomic(\n            plan_id, _apply_meal_modification, user_id=user_id\n        )"
    )
    assert cb_start < loop_idx < capture_idx < call_idx, (
        "la captura de `_dispatch_day_index[0]` debe vivir DENTRO del callback, "
        "en su loop de re-localización de `target_day_fresh`."
    )


def test_mutator_chatmod_locale_reuses_existing_profile_load(_tools_src):
    """[Fix round 1 · F6] `_p1i18n_locale_cm` se hidrata del MISMO round-trip que
    `_hp` (`_get_profile(user_id)`, ya usado por el backstop clínico) — no debe
    existir un SELECT dedicado nuevo para esta superficie."""
    assert '_p1i18n_locale_cm = _full_profile_cm.get("locale")' in _tools_src
    dispatch_block = _bounded_block(_tools_src, _CHATMOD_DISPATCH_START, _CHATMOD_DISPATCH_END)
    assert "SELECT locale" not in dispatch_block


def test_no_print_statements_tools_p1i18n_block(_tools_src):
    """Convención del repo (P2-LOGGER-MIGRATION): sanity barato sobre el bloque de
    dispatch específico de este P-fix (no sobre tools.py entero — fuera de alcance
    de Task 3 y ya cubierto por test_p2_logger_migration.py)."""
    dispatch_block = _bounded_block(_tools_src, _CHATMOD_DISPATCH_START, _CHATMOD_DISPATCH_END)
    assert "print(" not in dispatch_block


# ------------------------------------------------------------------
# 4. /recipe/expand (routers/plans.py) — [Fix round 1 · F5] 4º mutador, parser-based.
#
# `_meal["recipe"] = list(expanded_steps) + _kept` reemplaza el array de receta
# ENTERO (`_set_expanded_recipe_preserving_notes`) — `_display[locale].recipe` es un
# array alineado por índice con el original (spec §Contrato de datos), así que sin
# el pop queda desalineado/mintiendo pasos VIEJOS sobre la receta ya expandida.
# Camino 2 del endpoint propaga la expansión a TODAS las ocurrencias de
# nombre+receta idéntica SIN `break` (puede tocar varios días) — el pop y el
# despacho cubren la lista completa de meals/días REALMENTE tocados, no un único
# índice fijo como los otros 3 mutadores.
# ------------------------------------------------------------------

_RECIPEEXPAND_DISPATCH_START = "P1-PLAN-DISPLAY-I18N-MUTATOR-recipeexpand] Despacho best-effort post-persist"
_RECIPEEXPAND_DISPATCH_END = "P2-EXPAND-QUOTA-ABORT · 2026-07-02] callback abortó"


def test_mutator_recipeexpand_anchor_present(_plans_src):
    assert "P1-PLAN-DISPLAY-I18N-MUTATOR-recipeexpand" in _plans_src


def test_mutator_recipeexpand_pop_is_inside_apply_recipe_expansion_callback(_plans_src):
    """El pop debe vivir DENTRO de `_apply_recipe_expansion` — ANTES de la llamada
    a `update_plan_data_atomic(target_plan_id, _apply_recipe_expansion, ...)`."""
    cb_start = _plans_src.index("def _apply_recipe_expansion(")
    pop_idx = _plans_src.index("for _em_disp in _expand_touched_meals:")
    call_idx = _plans_src.index(
        "update_plan_data_atomic(\n                    target_plan_id, _apply_recipe_expansion, user_id=user_id\n                )"
    )
    assert cb_start < pop_idx < call_idx, (
        "el pop de `_display` en recipe/expand debe estar DENTRO de "
        "`_apply_recipe_expansion` (entre su `def` y la llamada a "
        "`update_plan_data_atomic(target_plan_id, _apply_recipe_expansion, ...)`)."
    )


def test_mutator_recipeexpand_pop_is_after_recipe_reassignment_not_before(_plans_src):
    """El pop debe ir DESPUÉS de `_meal["recipe"] = list(expanded_steps) + _kept`
    (donde `_set_expanded_recipe_preserving_notes` reemplaza la receta) — nunca
    antes."""
    assign_idx = _plans_src.index('_meal["recipe"] = list(expanded_steps) + _kept')
    pop_idx = _plans_src.index("for _em_disp in _expand_touched_meals:")
    assert assign_idx < pop_idx


def test_mutator_recipeexpand_pop_is_after_ok_flag_before_return(_plans_src):
    """El pop es el ÚLTIMO write del callback antes de `_expand_persisted["ok"] =
    True` + `return plan_data_fresh` (tras finalize/motor/micros/band-parity/rebuild
    de listas, mismo criterio que los otros 3 mutadores)."""
    pop_idx = _plans_src.index("for _em_disp in _expand_touched_meals:")
    ok_flag_idx = _plans_src.index('_expand_persisted["ok"] = True')
    return_idx = _plans_src.index("return plan_data_fresh")
    assert pop_idx < ok_flag_idx < return_idx


def test_mutator_recipeexpand_touched_meals_tracked_in_both_caminos(_plans_src):
    """Camino 1 (targeting por índices) Y Camino 2 (propagación por nombre+receta,
    sin `break`) deben AMBOS alimentar `_expand_touched_meals`/`_expand_touched_days`
    — si solo uno lo hiciera, el pop/despacho dejarían meals tocados sin cubrir en
    el camino que propaga a múltiples ocurrencias."""
    assert _plans_src.count("_expand_touched_meals.append(target_meal_fresh)") == 1  # Camino 1
    assert _plans_src.count("_expand_touched_meals.append(m)") == 1  # Camino 2
    assert _plans_src.count("_expand_touched_days.add(req_day_index)") == 1  # Camino 1
    assert _plans_src.count("_expand_touched_days.add(_day_idx_exp)") == 1  # Camino 2


def test_mutator_recipeexpand_dispatch_after_atomic_call(_plans_src):
    """El despacho vive DESPUÉS de la llamada a `update_plan_data_atomic` (post-
    persist, FUERA del lock) — nunca antes."""
    call_idx = _plans_src.index(
        "update_plan_data_atomic(\n                    target_plan_id, _apply_recipe_expansion, user_id=user_id\n                )"
    )
    dispatch_idx = _plans_src.index(_RECIPEEXPAND_DISPATCH_START)
    assert dispatch_idx > call_idx


def test_mutator_recipeexpand_dispatch_uses_touched_days_scoped_batch(_plans_src):
    """El despacho acota `day_indices=sorted(_expand_touched_days)` — los días
    REALMENTE tocados (puede ser >1, Camino 2 propaga), no el plan entero ni un
    índice fijo."""
    dispatch_block = _bounded_block(
        _plans_src, _RECIPEEXPAND_DISPATCH_START, _RECIPEEXPAND_DISPATCH_END
    )
    assert "schedule_plan_display_enrichment(" in dispatch_block
    assert "day_indices=sorted(_expand_touched_days)" in dispatch_block


def test_mutator_recipeexpand_dispatch_uses_should_enrich_locale_helper(_plans_src):
    """[Fix round 1 · F10]"""
    dispatch_block = _bounded_block(
        _plans_src, _RECIPEEXPAND_DISPATCH_START, _RECIPEEXPAND_DISPATCH_END
    )
    assert "should_enrich_locale" in dispatch_block
    assert '!= "es-DO"' not in dispatch_block


def test_mutator_recipeexpand_dispatch_gated_by_persisted_ok(_plans_src):
    """El despacho solo corre si el callback realmente persistió
    (`_expand_persisted["ok"]`) Y hay días tocados — un callback abortado (target
    raced) no debe disparar una retraducción sobre nada."""
    dispatch_block = _bounded_block(
        _plans_src, _RECIPEEXPAND_DISPATCH_START, _RECIPEEXPAND_DISPATCH_END
    )
    assert 'if _expand_persisted["ok"] and _expand_touched_days:' in dispatch_block


# ------------------------------------------------------------------
# Funcional: chat-modify — un meal viejo CON `_display` heredado no sobrevive al
# modify. Invoca `execute_modify_single_meal` de VERDAD (no hay forma de aislar el
# callback anidado sin ejecutar la función completa), mockeando la cadena LLM +
# los backstops que requieren DB/red: `get_latest_meal_plan_with_id`, el perfil
# (`db.get_user_profile`), el circuit breaker, `ChatDeepSeek` (constructor +
# `.with_structured_output(MealModel).invoke(...)`), y `update_plan_data_atomic`
# (capturado igual que el `engine` fixture de Task 1 — aplica el callback real
# sobre un `plan_data` local mutable y expone el resultado).
# ------------------------------------------------------------------


class _FakeCircuitBreaker:
    def can_proceed(self) -> bool:
        return True

    def record_success(self) -> None:
        pass

    def record_failure(self) -> None:
        pass


class _FakeStructuredLLM:
    def __init__(self, response):
        self._response = response

    def invoke(self, _messages):
        return self._response


class _FakeChatDeepSeek:
    """Sustituye `llm_provider.ChatDeepSeek` tal como lo usa `tools.py`:
    `ChatDeepSeek(model=..., temperature=..., timeout=...).with_structured_output(
    MealModel).invoke(prompt)`. `next_response` es de clase (no de instancia) porque
    `execute_modify_single_meal` construye DOS instancias (`modify_llm` +
    `_modify_llm_for_usage`, esta última sin `.with_structured_output`)."""

    next_response = None

    def __init__(self, *args, **kwargs):
        pass

    def with_structured_output(self, _model_cls):
        return _FakeStructuredLLM(_FakeChatDeepSeek.next_response)


@pytest.fixture
def _chatmod_engine(monkeypatch):
    """Instala los dobles de infraestructura para invocar
    `tools.execute_modify_single_meal` de punta a punta. `state["plan_data"]` es el
    dict mutable que el fake de `update_plan_data_atomic` re-lee y persiste (espejo
    del `engine` fixture del motor, sección MOTOR arriba)."""
    import tools as tools_module
    import db as db_module_local
    import db_plans as db_plans_module

    state = {"plan_data": None, "atomic_calls": []}

    monkeypatch.setattr(
        tools_module, "get_latest_meal_plan_with_id",
        lambda uid: {"id": "plan-chatmod-1", "plan_data": state["plan_data"]},
    )
    monkeypatch.setattr(
        db_module_local, "get_user_profile",
        lambda uid: {"health_profile": {}},
    )
    monkeypatch.setattr(tools_module, "_get_circuit_breaker", lambda model_name: _FakeCircuitBreaker())
    monkeypatch.setattr(tools_module, "ChatDeepSeek", _FakeChatDeepSeek)
    monkeypatch.setattr(tools_module, "SLOT_APPROPRIATENESS_GATE_ENABLED", False)

    def _fake_atomic(plan_id, mutator, user_id=None, **kwargs):
        state["atomic_calls"].append({"plan_id": plan_id, "user_id": user_id})
        result = mutator(state["plan_data"])
        if isinstance(result, dict):
            state["plan_data"] = result
        return state["plan_data"]

    monkeypatch.setattr(db_plans_module, "update_plan_data_atomic", _fake_atomic)

    # Validadores opcionales que requieren un LLM/DB real de comparación — el
    # contrato bajo prueba es el DELETE-on-write, no estos guardrails de calidad.
    monkeypatch.setenv("MEALFIT_SWAP_MACROS_VALIDATE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_MACRO_TRUTHUP", "false")
    monkeypatch.setenv("MEALFIT_UPDATE_MACRO_REBALANCE", "false")
    monkeypatch.setenv("MEALFIT_SWAP_PER_MEAL_MACRO_CLOSER", "false")

    return state, tools_module


def _chatmod_old_meal_with_display() -> dict:
    return {
        "meal": "Desayuno",
        "name": "Avena con canela",
        "desc": "Avena caliente con canela.",
        "time": "7:00 AM",
        "cals": 300,
        "protein": 10,
        "carbs": 45,
        "fats": 6,
        "ingredients": ["1 taza de avena"],
        "recipe": ["Mise en place: medir la avena.", "El Toque de Fuego: cocinar 5 min.", "Montaje: servir."],
        # heredado de un enrich previo — DEBE desaparecer tras el modify (delete-on-write).
        "_display": {
            "en-US": {
                "name": "Oatmeal with cinnamon",
                "description": "Hot oatmeal with cinnamon.",
                "recipe": ["Mise en place: ...", "..."],
                "ingredients": ["1 cup oats (avena)"],
            }
        },
    }


def _chatmod_new_meal_response() -> dict:
    """Dict crudo (rama `isinstance(new_meal_response, dict)` de
    `execute_modify_single_meal`, NO `MealModel.model_dump()`) — a propósito, para
    poder colar una `_display` en el meal DE REEMPLAZO y así ejercer de verdad el
    pop: `new_meal_data` normalmente NO trae `_display` porque el LLM no la genera
    (spec: 'el pop es el contrato LEGIBLE, no defensa contra un vector conocido
    hoy') — un `MealModel(...)` real jamás la tendría (no es un field del schema),
    así que un test con MealModel pasaría IGUAL con el pop comentado (falso verde,
    detectado por mutation-test manual antes de fijar este fixture). El dict crudo
    SÍ puede portar la clave (simula el caso 'un paso intermedio la reintroduce' que
    el comentario de producción anticipa) y hace el test sensible a la mutación."""
    return {
        "meal": "Desayuno",
        "name": "Avena con banana",
        "desc": "Avena caliente con banana en rodajas.",
        "prep_time": "10 min",
        "difficulty": "Fácil",
        "cals": 300,
        "protein": 10,
        "carbs": 45,
        "fats": 6,
        "ingredients": ["1 taza de avena", "1 banana"],
        "recipe": [
            "Mise en place: cortar la banana en rodajas.",
            "El Toque de Fuego: cocinar la avena 5 min.",
            "Montaje: coronar con la banana.",
        ],
        # `_display` colada en el meal DE REEMPLAZO — el escenario real que el pop
        # protege (spec "Invalidación": DELETE-on-write en TODA mutación de meal).
        "_display": {
            "en-US": {
                "name": "Oatmeal with banana",
                "description": "STALE — heredado, no debe sobrevivir al modify.",
                "recipe": ["stale step"],
                "ingredients": ["stale ingredient"],
            }
        },
    }


def test_chatmod_functional_old_display_does_not_survive_modify(_chatmod_engine):
    state, tools_module = _chatmod_engine
    state["plan_data"] = {"days": [{"day": 1, "meals": [_chatmod_old_meal_with_display()]}]}
    _FakeChatDeepSeek.next_response = _chatmod_new_meal_response()

    result_str = tools_module.execute_modify_single_meal(
        user_id="user-chatmod-1",
        day_number=1,
        meal_type="Desayuno",
        changes="agrégale banana",
        allow_pantry_expansion=True,
    )

    assert not result_str.startswith("ERROR"), result_str
    assert "FALLO" not in result_str, result_str
    parsed = json.loads(result_str)
    assert parsed["modified_meal"]["name"] == "Avena con banana"
    assert "_display" not in parsed["modified_meal"], (
        "el meal modificado conservó `_display` heredado del plato viejo — "
        "DELETE-on-write roto en el mutador de chat-modify."
    )
    persisted_meal = state["plan_data"]["days"][0]["meals"][0]
    assert "_display" not in persisted_meal, (
        "el `plan_data` PERSISTIDO (resultado del callback bajo el lock) conservó "
        "`_display` heredado — el pop no vive dentro de `_apply_meal_modification` "
        "o corre sobre el objeto equivocado."
    )
    assert persisted_meal["name"] == "Avena con banana"


# ================================================================================
# SECCIÓN: MOTOR PLAN-WIDE `apply_update_macro_engine` (Fix round 2)
#
# tooltip-anchor: `P1-PLAN-DISPLAY-I18N-MUTATOR-macroengine` (graph_orchestrator.py).
#
# Re-review (task-3-rereview.md, Parte 2 — PREMISA CONFIRMADA, más grave que F1 del round 1):
# `apply_update_macro_engine` es PLAN-WIDE (itera TODOS los días de `plan_data`, no solo el
# día que el caller tocó) y swap-persist/recipe-expand/chat-modify le pasan el PLAN ENTERO —
# un día colateral fuera de banda [0.90,1.12] que la generación ya traía se re-cuantiza
# (ingredients + recipe, vía `_sync_recipe_step_quantities`) sin que ningún caller lo sepa.
# Ruling del controller, recomendación (ii'): el pop baja al MOTOR (cubre los 5 call sites
# actuales y los futuros sin wiring por-caller) + el motor reporta los días tocados vía el
# parámetro opcional `touched_day_indices` para que los 3 mutadores del feature amplíen su
# despacho de re-enriquecimiento.
#
# Elección de (c) — "el despacho del swap incluye los días tocados": FUNCIONAL, no parser.
# El `_swap_engine` fixture (sección swap arriba) ya invoca `api_swap_meal_persist` de punta a
# punta con `update_plan_data_atomic` capturado — mockear `apply_update_macro_engine` para que
# reporte un día colateral y verificar que ese día aparece en `day_indices` del despacho
# ejercita la UNIÓN real (`_swap_macroengine_touched_days ∪ {day_index}`), no solo que el
# texto "touched_day_indices" aparezca cerca del despacho (un parser no distinguiría una unión
# correcta de una que solo referencia la variable sin usarla).
# ================================================================================

_GO_SRC_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def _go_src() -> str:
    return _GO_SRC_PATH.read_text(encoding="utf-8")


# ------------------------------------------------------------------
# (a) Unit del motor: pop + reporte de días tocados.
# ------------------------------------------------------------------

_MACROENGINE_DENS = {
    "pollo": {"kcal": 1.65, "protein": 0.31, "carbs": 0.0, "fats": 0.036},
    "arroz": {"kcal": 1.30, "protein": 0.027, "carbs": 0.28, "fats": 0.003},
    "aguacate": {"kcal": 1.60, "protein": 0.02, "carbs": 0.085, "fats": 0.147},
}


class _MacroEngineRefDB:
    """Mismo patrón que `test_p1_update_macro_parity.py::_RefDB` — fake determinista de
    `macros_from_ingredient_string`, sin tocar la DB real."""

    def macros_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*g de (\w+)", str(s).lower())
        if not m:
            return None
        g, food = float(m.group(1)), m.group(2)
        d = _MACROENGINE_DENS.get(food)
        return {k: v * g for k, v in d.items()} if d else None


def _macroengine_plan_two_days() -> dict:
    """Día 1 (índice 0) FUERA de banda — P≈75/C≈69/F≈16 vs target 90/70/22 (proteína .83, grasa
    .73, ambos fuera de [0.90,1.12]) — con `_display` heredado en su primer meal. Día 2 (índice
    1) EN banda: sus campos `protein`/`carbs`/`fats` (que `_out_of_band` suma DIRECTO del meal,
    no desde `ingredients` — desacopla este día de la regex de `_MacroEngineRefDB`) igualan el
    target exacto → ratio 1.0 en las 3 → nunca entra al `if _hit:` → su `_display` debe
    sobrevivir intacto y su índice NUNCA aparecer en `touched_day_indices`."""
    return {
        "macros": {"protein": 90, "carbs": 70, "fats": 22},
        "calories": 850,
        "days": [
            {
                "day": 1,
                "meals": [
                    {
                        "meal": "Almuerzo", "name": "Pollo con arroz",
                        "protein": 40, "carbs": 27, "fats": 12,
                        "ingredients": ["120g de pollo", "80g de arroz", "50g de aguacate"],
                        "ingredients_raw": ["120g de pollo", "80g de arroz", "50g de aguacate"],
                        "recipe": ["Cocinar el pollo.", "Servir con arroz."],
                        "_display": {
                            "en-US": {"name": "STALE — out of band day", "description": "x",
                                      "recipe": ["x"], "ingredients": ["x"]}
                        },
                    },
                    {
                        "meal": "Cena", "name": "Pollo con arroz II",
                        "protein": 35, "carbs": 42, "fats": 4,
                        "ingredients": ["100g de pollo", "150g de arroz"],
                        "ingredients_raw": ["100g de pollo", "150g de arroz"],
                        "recipe": ["Cocinar el pollo.", "Servir con arroz."],
                    },
                ],
            },
            {
                "day": 2,
                "meals": [
                    {
                        "meal": "Almuerzo", "name": "Plato en banda",
                        "protein": 90, "carbs": 70, "fats": 22,
                        "ingredients": ["1 plato balanceado"],
                        "ingredients_raw": ["1 plato balanceado"],
                        "recipe": ["Servir."],
                        "_display": {
                            "en-US": {"name": "In-band day, must survive", "description": "x",
                                      "recipe": ["x"], "ingredients": ["x"]}
                        },
                    },
                ],
            },
        ],
    }


def test_macroengine_pops_display_on_out_of_band_day_and_reports_it(monkeypatch):
    import graph_orchestrator as go

    monkeypatch.setattr(go, "UPDATE_MACRO_ENGINE_ENABLED", True)
    plan = _macroengine_plan_two_days()
    touched: list = []

    n = go.apply_update_macro_engine(
        plan, surface="test", db=_MacroEngineRefDB(), touched_day_indices=touched
    )

    assert n == 1, "solo el día de índice 0 está fuera de banda"
    assert touched == [0], "el motor debe reportar el ÍNDICE del día fuera de banda tocado"
    day0_meal0 = plan["days"][0]["meals"][0]
    assert "_display" not in day0_meal0, (
        "el `_display` heredado del meal del día FUERA de banda debe desaparecer — el motor "
        "lo re-cuantizó (rebalance/refine/qty-sync toca ingredients Y recipe)."
    )


def test_macroengine_leaves_in_band_day_display_intact_and_unreported(monkeypatch):
    import graph_orchestrator as go

    monkeypatch.setattr(go, "UPDATE_MACRO_ENGINE_ENABLED", True)
    plan = _macroengine_plan_two_days()
    touched: list = []

    go.apply_update_macro_engine(
        plan, surface="test", db=_MacroEngineRefDB(), touched_day_indices=touched
    )

    assert 1 not in touched, "el día EN banda (índice 1) no debe reportarse como tocado"
    day1_meal = plan["days"][1]["meals"][0]
    assert day1_meal.get("_display") == {
        "en-US": {"name": "In-band day, must survive", "description": "x",
                  "recipe": ["x"], "ingredients": ["x"]}
    }, "el `_display` de un día EN banda (nunca mutado) debe sobrevivir intacto."


def test_macroengine_touched_day_indices_none_is_backward_compatible(monkeypatch):
    """Callers legacy (los que NO pasan `touched_day_indices`, default `None`) no deben
    romperse — ni `AttributeError` por `.append()` sobre `None`, ni cambio en el retorno `int`."""
    import graph_orchestrator as go

    monkeypatch.setattr(go, "UPDATE_MACRO_ENGINE_ENABLED", True)
    plan = _macroengine_plan_two_days()
    n = go.apply_update_macro_engine(plan, surface="test", db=_MacroEngineRefDB())
    assert n == 1


# ------------------------------------------------------------------
# (b) Parser: anchor + firma del parámetro opcional + posición del pop.
# ------------------------------------------------------------------


def test_macroengine_anchor_present(_go_src):
    assert "P1-PLAN-DISPLAY-I18N-MUTATOR-macroengine" in _go_src


def test_macroengine_touched_day_indices_is_keyword_only_with_none_default(_go_src):
    """`touched_day_indices` debe vivir DESPUÉS del `*,` que hace keyword-only al resto de
    parámetros (firma preexistente) — un caller posicional legacy no debe poder pisarlo."""
    assert "touched_day_indices: list = None" in _go_src
    sig_idx = _go_src.index("def apply_update_macro_engine(")
    star_idx = _go_src.index("*,", sig_idx)
    param_idx = _go_src.index("touched_day_indices: list = None", sig_idx)
    assert sig_idx < star_idx < param_idx


def test_macroengine_pop_is_inside_hit_block(_go_src):
    """El pop debe vivir DENTRO del `if _hit:` que ya itera `for _m in _meals` — fuera de esa
    rama el motor no re-cuantizó nada de ese día, y popear ahí sería incorrecto (borraría
    `_display` de días que nunca se tocaron)."""
    hit_idx = _go_src.index("if _hit:\n                touched += 1")
    pop_idx = _go_src.index('_m.pop("_display", None)')
    append_idx = _go_src.index("touched_day_indices.append(_day_idx_ume)")
    assert hit_idx < append_idx < pop_idx, (
        "orden esperado dentro de `if _hit:`: touched += 1 → reporte del día → pop por meal."
    )


def test_macroengine_return_type_still_int(_go_src):
    """La firma pública sigue devolviendo `int` — los callers legacy que solo miran el conteo
    (sin usar `touched_day_indices`) no deben romperse."""
    sig_idx = _go_src.index("def apply_update_macro_engine(")
    sig_end = _go_src.index(":\n", sig_idx)
    signature = _go_src[sig_idx:sig_end]
    assert "-> int" in signature


# ------------------------------------------------------------------
# (c) Funcional — el despacho de swap-persist incluye los días colaterales que el motor
# reporta (ver nota de "Elección de (c)" arriba de esta sección).
# ------------------------------------------------------------------


def test_swap_functional_dispatch_includes_macroengine_touched_days(_swap_engine, monkeypatch):
    import graph_orchestrator as go_module

    state, plans_module = _swap_engine
    # Locale que SÍ dispara el despacho (el fixture base deja `get_user_profile` sin locale).
    monkeypatch.setattr(
        db_module, "get_user_profile", lambda uid: {"health_profile": {}, "locale": "en-US"}
    )
    # El wiring bajo prueba vive DENTRO del bloque `MEALFIT_UPDATE_RECOMPUTE_MICROS` — el
    # fixture base lo deja en "false" (para aislar el test de F12); aquí necesitamos que corra.
    monkeypatch.setenv("MEALFIT_UPDATE_RECOMPUTE_MICROS", "true")

    def _fake_engine(plan_data, *, surface, db=None, pantry_strict=False, form_data=None,
                      touched_day_indices=None):
        # Simula que el motor re-cuantizó un día COLATERAL (índice 4) fuera del swapeado (0).
        if touched_day_indices is not None:
            touched_day_indices.append(4)
        return 1

    monkeypatch.setattr(go_module, "apply_update_macro_engine", _fake_engine)

    calls: list = []
    monkeypatch.setattr(
        pdi, "schedule_plan_display_enrichment",
        lambda *a, **kw: calls.append((a, kw)),
    )

    old_meal = {
        "meal": "Almuerzo", "name": "Pollo guisado", "desc": "Pollo con arroz.",
        "cals": 500, "protein": 30, "carbs": 50, "fats": 15,
        "ingredients": ["150 g de pechuga de pollo", "1 taza de arroz"],
        "recipe": ["Sofreir el pollo.", "Servir con arroz."],
    }
    state["plan_data"] = {"days": [{"day": 1, "meals": [old_meal]}]}
    body = {
        "day_index": 0,
        "meal_index": 0,
        "new_meal": {
            "name": "Pescado al horno", "desc": "Pescado con vegetales.",
            "cals": 480, "protein": 32, "carbs": 40, "fats": 14,
            "ingredients": ["200 g de pescado", "1 taza de vegetales"],
            "recipe": ["Hornear el pescado.", "Servir con vegetales."],
        },
    }

    result = plans_module.api_swap_meal_persist(
        plan_id="plan-swap-1", data=body, verified_user_id="user-swap-1",
    )

    assert result == {"success": True}
    assert len(calls) == 1, "el despacho de re-enriquecimiento debe dispararse exactamente una vez"
    _args, kwargs = calls[0]
    day_indices = kwargs.get("day_indices")
    assert day_indices is not None, "schedule_plan_display_enrichment debe recibir day_indices"
    assert 4 in day_indices, (
        "el día COLATERAL que el motor plan-wide reportó (4) debe estar en day_indices del "
        "despacho — sin la unión, ese día quedaría sin re-enriquecer aunque el motor ya "
        "popeó su `_display`."
    )
    assert 0 in day_indices, "el día REALMENTE swapeado (0) debe seguir incluido."
