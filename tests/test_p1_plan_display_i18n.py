"""[P1-PLAN-DISPLAY-I18N · 2026-08-19] Capa de display i18n del plan (_display.<locale>).

tooltip-anchor: P1-PLAN-DISPLAY-I18N

Este archivo crece por task (Task 1 = motor). Tasks posteriores (disparadores, mutadores,
frontend contract) añaden secciones NUEVAS a este MISMO archivo — no crear ficheros nuevos
por task, es la convención fijada en el brief de Task 1.

================================================================================
SECCIÓN: MOTOR (Task 1) — `plan_display_i18n.enrich_plan_display` /
`plan_display_i18n.schedule_plan_display_enrichment`
================================================================================

Todo el LLM y el I/O real (DB, cross-worker lock) están mockeados: estos tests NUNCA tocan
Neon ni ningún provider — verifican el CONTRATO del motor (guards, validación determinista,
fail-open total). El fake de `update_plan_data_atomic` captura el callback (mutator) y lo
aplica a un dict `plan_data` local, exactamente como pide el brief, para poder assertar el
jsonb `_display` resultante sin una DB real.
"""
from __future__ import annotations

import copy

import pytest

import plan_display_i18n as pdi


# ------------------------------------------------------------------
# Dobles de prueba
# ------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata or {}


class _FakeLLM:
    """Sustituye `ChatDeepSeek`: instanciar -> `.invoke(messages)` -> respuesta programada
    por el test (o excepción, para el caso "excepción del provider")."""

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

    monkeypatch.setattr(pdi, "_try_claim_enrich_lock_cross_worker", lambda plan_id, locale: True)
    monkeypatch.setattr(pdi, "ChatDeepSeek", _FakeLLM)
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


# ------------------------------------------------------------------
# 1. Respuesta válida completa
# ------------------------------------------------------------------


def test_respuesta_valida_completa_persiste_display(engine):
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(
        content=(
            '{"meals":[{"i":0,'
            '"name":"Stewed red beans",'
            '"description":"Traditional Dominican stew with red beans.",'
            '"recipe":["Saute the seasoning in hot oil.","Add the beans and cook 20 minutes."],'
            '"ingredients":["30 g red beans (Habichuelas rojas)","1 unit onion (Cebolla)"]}]}'
        ),
        usage_metadata={"input_tokens": 120, "output_tokens": 80},
    )

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
    # Persistencia: UN write, con el user_id correcto (ownership I2).
    assert engine["persist_calls"] == [{"plan_id": "plan-1", "user_id": "user-1"}]
    # Telemetría: fue a llm_usage_events (node=plan_display_i18n implícito en el módulo),
    # NUNCA a api_usage — este módulo no importa nada de api_usage.
    assert len(engine["telemetry_calls"]) == 1
    assert engine["telemetry_calls"][0]["node"] == "plan_display_i18n"
    assert engine["telemetry_calls"][0]["plan_id"] == "plan-1"
    assert engine["telemetry_calls"][0]["user_id"] == "user-1"


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
    # Sin canónico que verificar, la línea traducida se conserva tal cual.
    assert display["ingredients"] == ["2 units"]


def test_ingrediente_con_canonico_perdido_cae_a_la_linea_original(engine):
    """Si la línea traducida NO contiene el canónico español, esa línea puntual se
    descarta y cae de vuelta al texto original — el resto del meal SÍ se enriquece
    (delete-on-write per-línea, no per-meal, per la spec: "un gloss que pierde el
    identificador es peor que no tener gloss")."""
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
    # 0 meals válidos -> jamás se invoca update_plan_data_atomic.
    assert engine["persist_calls"] == []


# ------------------------------------------------------------------
# 4. JSON roto -> fail-open, no persiste nada.
# ------------------------------------------------------------------


def test_json_roto_fail_open(engine):
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(content="esto no es JSON {{{")

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert result == {"enriched_meals": 0, "skipped": "json_parse_error"}
    assert "_display" not in engine["plan_data"]["days"][0]["meals"][0]
    assert engine["persist_calls"] == []
    assert engine["telemetry_calls"] == []


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
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(
        content=(
            '{"meals":[{"i":0,"name":"Stewed red beans","description":"Desc.",'
            '"recipe":["Step A.","Step B."],'
            '"ingredients":["30 g red beans (Habichuelas rojas)","1 unit onion (Cebolla)"]}]}'
        )
    )

    result = pdi.enrich_plan_display("plan-1", "user-1", "en-US", day_indices=[0])

    assert result == {"enriched_meals": 1, "skipped": None}
    assert "_display" in engine["plan_data"]["days"][0]["meals"][0]
    assert "_display" not in engine["plan_data"]["days"][1]["meals"][0]
    # El prompt solo debe mencionar el meal del día 0.
    assert "Otro día" not in _FakeLLM.captured_prompts[0]


# ------------------------------------------------------------------
# `_extract_canonical_name`: unidad interna reusada por la validación per-línea.
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("30 g Habichuelas rojas", "Habichuelas rojas"),
        ("1 unidad Cebolla", "Cebolla"),
        ("2 unidades", ""),
        ("", ""),
        ("Sal al gusto", "Sal al gusto"),
        ("100 g Pollo (sin piel)", "Pollo"),
    ],
)
def test_extract_canonical_name(raw, expected):
    assert pdi._extract_canonical_name(raw) == expected


# ------------------------------------------------------------------
# `schedule_plan_display_enrichment`: wrapper fire-and-forget.
# ------------------------------------------------------------------


def test_schedule_dispara_enrich_en_background(engine, monkeypatch):
    """El wrapper corre en un thread — para el test lo hacemos síncrono forzando
    `Thread.start` a ejecutar el target inline, y verificamos que `enrich_plan_display`
    corrió con los mismos argumentos."""
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
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(
        content=(
            '{"meals":[{"i":0,"name":"Stewed red beans","description":"Desc.",'
            '"recipe":["Step A.","Step B."],'
            '"ingredients":["30 g red beans (Habichuelas rojas)","1 unit onion (Cebolla)"]}]}'
        )
    )

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
