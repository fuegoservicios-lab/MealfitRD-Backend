"""[P1-I18N-DISPLAY-NIVEL-PLAN-SIN-VIA · 2026-08-22] El nombre del plan y el panel de
razonamiento viajaban de POLIZON en un lote de COMIDAS, y no tenian via propia.

EL DEFECTO. El troceo se construye SOLO desde los targets de comida:

    _todos_los_targets = _collect_targets(days, requested_day_indices, locale=locale)
    lotes_iniciales    = _particionar_targets(_todos_los_targets, ...)
    _pendientes        = list(reversed(lotes_iniciales))

En el estado NORMAL --todas las comidas ya traducidas para ese locale-- `_collect_targets`
devuelve [], `_particionar_targets` devuelve [] y el `while _pendientes` no corre ni una
vez. El `continue` de dentro del bucle distinguia explicitamente «lote vacio sin nada
pendiente» de «lote vacio con nombre/insights», o sea que la rama estaba PREVISTA y era
inalcanzable: nadie encolaba ese lote vacio.

MEDIDO con los dobles de este mismo fichero, antes del arreglo:

    enrich_plan_display(...) -> {'enriched_meals': 0, 'skipped': 'no_meals'}
    invocaciones al LLM: 0        _display de nivel plan: None

LO QUE VEIA EL USUARIO. Renombra su plan en el Historial. El rename popea `_display` --y
popea el de TODOS los locales, no solo el del suyo-- asi que a partir de ese momento el
titulo vuelve al espanol PARA SIEMPRE, y con el se va el panel «Diagnostico / Plan de
Accion / Tip del Chef». No hay disparador que lo recupere: el unico que existe cuelga de
que vuelva a aparecer trabajo de comidas, y en un plan ya traducido no vuelve nunca.
Mismo desenlace si la validacion del nombre falla una vez.

EL ARREGLO es una linea --encolar un lote VACIO cuando no hay lotes pero si hay nivel de
plan pendiente-- porque todo lo demas ya lo soportaba. Eso es lo que hace que sea seguro:
no se inventa un camino nuevo, se alcanza el que ya estaba escrito.

tooltip-anchor: P1-I18N-DISPLAY-NIVEL-PLAN-SIN-VIA
"""
from __future__ import annotations

import json

import pytest

from tests.test_p1_plan_display_i18n import (  # noqa: F401  (fixtures via import)
    _FakeLLM,
    _FakeResponse,
    _base_meal,
    _valid_response_for_base_meal,
    _make_plan,
    _reset_fake_llm,
    engine,
    pdi,
)

_MARKER = "P1-I18N-DISPLAY-NIVEL-PLAN-SIN-VIA"

_LOCALE = "en-US"


def _meal_ya_traducido() -> dict:
    """Una comida CON `_display[en-US]` valido: el estado estable tras el primer ciclo."""
    m = _base_meal()
    m["_display"] = {
        _LOCALE: {
            "name": "Stewed red beans",
            "desc": "Traditional Dominican stew with red beans.",
            "recipe": [
                "Saute the sofrito in hot oil.",
                "Add the beans and cook for 20 minutes.",
            ],
            "ingredients": ["30 g red beans (Habichuelas rojas)", "1 unit onion (Cebolla)"],
        }
    }
    return m


def _plan_traducido_con_nivel_plan_pendiente() -> dict:
    plan = _make_plan([_meal_ya_traducido()])
    plan["name"] = "Sazón Fuerte, Vida en Equilibrio"
    plan["insights"] = [
        "Diagnóstico: adherencia baja y fatiga por ingredientes repetidos.",
        "Estrategia: comidas ultra-simples con rotación radical de proteínas.",
        "Tip del Chef: usa las proteínas asignadas como anclas sabrosas.",
    ]
    return plan


def _respuesta_solo_nivel_plan() -> _FakeResponse:
    return _FakeResponse(json.dumps({
        "plan_name": "Strong Flavor, Life in Balance",
        "insights": [
            "Diagnosis: low adherence and fatigue from repeated ingredients.",
            "Strategy: ultra-simple meals with radical protein rotation.",
            "Chef's Tip: use the assigned proteins as tasty anchors.",
        ],
        "meals": [],
    }, ensure_ascii=False))


def test_un_plan_totalmente_traducido_sigue_pudiendo_traducir_su_titulo(engine) -> None:
    """EL test del gap. Cero comidas pendientes, nombre e insights pendientes."""
    engine["plan_data"] = _plan_traducido_con_nivel_plan_pendiente()
    _FakeLLM.NEXT_RESPONSE = _respuesta_solo_nivel_plan()

    res = pdi.enrich_plan_display("plan-1", "user-1", _LOCALE)

    assert _FakeLLM.invoke_count == 1, (
        f"no se llamo al LLM: sin lote de comidas no habia via para el nivel de plan, que "
        f"es exactamente el gap. resultado={res!r} [{_MARKER}]"
    )
    disp = (engine["plan_data"] or {}).get("_display") or {}
    assert _LOCALE in disp, (
        f"no se persistio `_display` de nivel plan. resultado={res!r} [{_MARKER}]"
    )
    assert disp[_LOCALE].get("name") == "Strong Flavor, Life in Balance", (
        f"el titulo no se tradujo: {disp[_LOCALE]!r} [{_MARKER}]"
    )
    assert len(disp[_LOCALE].get("insights") or []) == 3, (
        f"el panel de razonamiento no viajo con el titulo: {disp[_LOCALE]!r} [{_MARKER}]"
    )


def test_el_lote_vacio_NO_se_encola_si_no_hay_nada_de_nivel_plan(engine) -> None:
    """El control. Sin esto, el arreglo seria «llamar al LLM siempre», que costaria
    dinero en cada disparo sobre un plan ya completo."""
    plan = _make_plan([_meal_ya_traducido()])
    plan["name"] = "Sazón Fuerte, Vida en Equilibrio"
    plan["_display"] = {_LOCALE: {"name": "Strong Flavor, Life in Balance"}}
    engine["plan_data"] = plan
    _FakeLLM.NEXT_RESPONSE = _respuesta_solo_nivel_plan()

    res = pdi.enrich_plan_display("plan-1", "user-1", _LOCALE)

    assert _FakeLLM.invoke_count == 0, (
        f"se llamo al LLM sin nada que traducir: {res!r} [{_MARKER}]"
    )
    assert res.get("skipped") == "no_meals", res


def test_el_camino_normal_con_comidas_pendientes_no_cambia(engine) -> None:
    """No-regresion: el lote vacio se anade SOLO cuando no hay lotes de comida."""
    engine["plan_data"] = _make_plan([_base_meal()])
    engine["plan_data"]["name"] = "Sazón Fuerte, Vida en Equilibrio"
    # Se reusa el helper del fichero madre en vez de componer la respuesta a mano: la
    # primera version de este test la escribio a mano y daba `enriched_meals: 0` --el
    # validador la rechazaba por la clave `desc` en vez de `description`-- o sea que
    # habria «detectado» una regresion que no existia.
    _FakeLLM.NEXT_RESPONSE = _valid_response_for_base_meal()

    res = pdi.enrich_plan_display("plan-1", "user-1", _LOCALE)

    assert _FakeLLM.invoke_count == 1, res
    assert res.get("enriched_meals", 0) >= 1, (
        f"el camino con comidas pendientes dejo de traducirlas: {res!r} [{_MARKER}]"
    )


def test_el_rename_vuelve_a_abrir_la_via(engine) -> None:
    """La secuencia completa que el usuario vive: plan traducido -> rename (popea el
    `_display` de nivel plan) -> el siguiente enriquecimiento LO RECUPERA.

    Antes de este arreglo, este segundo ciclo devolvia `no_meals` y el titulo se quedaba
    en espanol para siempre.
    """
    plan = _make_plan([_meal_ya_traducido()])
    plan["name"] = "Mi plan renombrado"
    plan["_display"] = {}  # justo lo que deja el rename
    engine["plan_data"] = plan
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(json.dumps(
        {"plan_name": "My renamed plan", "meals": []}, ensure_ascii=False))

    res = pdi.enrich_plan_display("plan-1", "user-1", _LOCALE)

    disp = (engine["plan_data"] or {}).get("_display") or {}
    assert disp.get(_LOCALE, {}).get("name") == "My renamed plan", (
        f"tras un rename el titulo no vuelve a traducirse nunca: {res!r} {disp!r} "
        f"[{_MARKER}]"
    )
