"""[P1-I18N-DISPLAY-LOTE-PERDIDO-SIN-SENAL · 2026-08-22] Un lote que moria en el `invoke`
se tiraba sin reintento Y sin contarse, y el resumen lo reportaba como exito limpio.

LA PEOR COMBINACION POSIBLE, en dos mitades:

  1. SIN REINTENTO. El `except` del invoke era un `continue` seco. La rama hermana --JSON
     que no parsea-- SI parte el lote y reintenta cada mitad; la del invoke, no. O sea que
     un timeout transitorio de 60 s costaba el tramo entero, mientras que una salida
     truncada se recuperaba.

  2. SIN SENAL. `targets_perdidos` no se incrementaba, asi que la fila de telemetria decia
     0 perdidas; y `reason` colapsaba a `None` en cuanto se hubiera escrito ALGO. Un plan
     medio traducido daba exactamente la misma fila que uno completo.

LO QUE VIVE EL USUARIO: cambia de idioma, se dispara la traduccion, y se queda con medio
plan en frances y medio en espanol DE FORMA PERMANENTE -- el disparador 4 solo mira el
PRIMER y el ULTIMO dia del plan, asi que si el tramo perdido esta en medio no vuelve a
dispararse nunca, salvo que el usuario cambie de idioma otra vez. Y es justo el estado
«mitad y mitad en la misma pantalla» que el fallback per-linea existe para evitar.

Y NADIE PODIA VERLO: media traduccion permanente no dejaba ni una senal por encima de
`info`, y con `DEFAULT_EVENT_LEVEL=ERROR` un `info` no sube a Sentry. La eleccion de nivel
ES parte del arreglo, no cosmetica.

EL REINTENTO NO PUEDE HACER BUCLE: el techo `_max_invocaciones_por_ciclo` ya existia. Si el
proveedor esta caido de verdad, el techo para, y la rama de «presupuesto agotado» --que ya
contabilizaba bien-- recoge lo que quede.

tooltip-anchor: P1-I18N-DISPLAY-LOTE-PERDIDO-SIN-SENAL
"""
from __future__ import annotations

import json

from tests.test_p1_plan_display_i18n import (  # noqa: F401  (fixtures via import)
    _FakeLLM,
    _FakeResponse,
    _base_meal,
    _make_plan,
    _reset_fake_llm,
    _valid_response_for_base_meal,
    engine,
    pdi,
)

_MARKER = "P1-I18N-DISPLAY-LOTE-PERDIDO-SIN-SENAL"
_LOCALE = "en-US"


class _FallaUnaVezLLM:
    """Falla el PRIMER invoke y responde bien a partir del segundo."""

    invoke_count = 0

    def __init__(self, *a, **kw):
        pass

    def invoke(self, messages):
        type(self).invoke_count += 1
        if type(self).invoke_count == 1:
            raise TimeoutError("el proveedor tardo mas de 60 s")
        return _valid_response_for_base_meal()


class _FallaSiempreLLM:
    invoke_count = 0

    def __init__(self, *a, **kw):
        pass

    def invoke(self, messages):
        type(self).invoke_count += 1
        raise TimeoutError("proveedor caido")


def test_un_fallo_transitorio_del_invoke_se_reintenta(engine, monkeypatch) -> None:
    """LA mitad 1. Antes: `continue` seco y el tramo se perdia con un timeout."""
    _FallaUnaVezLLM.invoke_count = 0
    engine["plan_data"] = _make_plan([_base_meal()])
    monkeypatch.setattr(pdi, "build_chat_llm", lambda model, **kw: _FallaUnaVezLLM())

    res = pdi.enrich_plan_display("plan-1", "user-1", _LOCALE)

    assert _FallaUnaVezLLM.invoke_count >= 2, (
        f"el lote no se reintento tras el fallo de invoke: {_FallaUnaVezLLM.invoke_count} "
        f"llamada(s). resultado={res!r} [{_MARKER}]"
    )
    assert res.get("enriched_meals", 0) >= 1, (
        f"un timeout transitorio siguio costando el tramo entero: {res!r} [{_MARKER}]"
    )
    assert "_display" in engine["plan_data"]["days"][0]["meals"][0], (
        f"no se persistio nada tras el reintento [{_MARKER}]"
    )


def test_el_reintento_no_hace_bucle_infinito(engine, monkeypatch) -> None:
    """El control de la mitad 1. Con el proveedor caido de verdad, el techo para."""
    _FallaSiempreLLM.invoke_count = 0
    engine["plan_data"] = _make_plan([_base_meal()])
    monkeypatch.setattr(pdi, "build_chat_llm", lambda model, **kw: _FallaSiempreLLM())

    res = pdi.enrich_plan_display("plan-1", "user-1", _LOCALE)

    assert _FallaSiempreLLM.invoke_count <= 8, (
        f"el reintento no tiene techo: {_FallaSiempreLLM.invoke_count} llamadas. El techo "
        f"`_max_invocaciones_por_ciclo` deberia pararlo. [{_MARKER}]"
    )
    assert res.get("enriched_meals", 0) == 0, res


def test_una_perdida_parcial_NO_se_reporta_como_exito(engine, monkeypatch) -> None:
    """LA mitad 2. Un lote escrito + un lote perdido no puede dar `skipped: None`."""
    engine["plan_data"] = {"days": [{"meals": [_base_meal()]}, {"meals": [_base_meal()]}]}
    monkeypatch.setenv("MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS", "1")

    respuestas = [
        _valid_response_for_base_meal(),
        _FakeResponse(content="esto no es JSON {{{"),
        _FakeResponse(content="esto tampoco {{{"),
        _FakeResponse(content="ni esto {{{"),
        _FakeResponse(content="ni esto otro {{{"),
        _FakeResponse(content="basta {{{"),
        _FakeResponse(content="basta {{{"),
        _FakeResponse(content="basta {{{"),
    ]

    class _Sec:
        def __init__(self, *a, **kw):
            pass

        def invoke(self, messages):
            return respuestas.pop(0) if respuestas else _FakeResponse(content="{{{")

    monkeypatch.setattr(pdi, "build_chat_llm", lambda model, **kw: _Sec())

    res = pdi.enrich_plan_display("plan-1", "user-1", _LOCALE)

    assert res.get("enriched_meals", 0) >= 1, f"nada escrito, el caso no es el que mide: {res!r}"
    assert res.get("skipped") == "partial_loss", (
        f"un ciclo con escrituras Y perdidas se reporta como exito limpio ({res!r}). El "
        f"usuario tiene medio plan traducido, permanentemente, y la fila de telemetria dice "
        f"lo mismo que un ciclo completo. [{_MARKER}]"
    )


def test_un_ciclo_completo_sigue_diciendo_exito(engine, monkeypatch) -> None:
    """El control de la mitad 2: sin perdidas, `skipped` sigue siendo None. Sin esto, el
    arreglo podria ser «decir partial_loss siempre», que no informa de nada."""
    _FakeLLM.NEXT_RESPONSE = _valid_response_for_base_meal()
    engine["plan_data"] = _make_plan([_base_meal()])

    res = pdi.enrich_plan_display("plan-1", "user-1", _LOCALE)

    assert res == {"enriched_meals": 1, "skipped": None}, res


def test_la_telemetria_distingue_la_perdida_parcial(engine, monkeypatch) -> None:
    """La fila que ve el operador. `reason` es lo unico que separa «medio plan en espanol
    para siempre» de «todo bien»."""
    capturado = {}

    def _spy(plan_id, user_id, locale, resumen):
        capturado.update(resumen)

    monkeypatch.setattr(pdi, "_emit_result_telemetry", _spy)
    engine["plan_data"] = {"days": [{"meals": [_base_meal()]}, {"meals": [_base_meal()]}]}
    monkeypatch.setenv("MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS", "1")

    respuestas = [_valid_response_for_base_meal()] + [
        _FakeResponse(content="no es JSON {{{") for _ in range(8)
    ]

    class _Sec:
        def __init__(self, *a, **kw):
            pass

        def invoke(self, messages):
            return respuestas.pop(0) if respuestas else _FakeResponse(content="{{{")

    monkeypatch.setattr(pdi, "build_chat_llm", lambda model, **kw: _Sec())

    pdi.enrich_plan_display("plan-1", "user-1", _LOCALE)

    assert capturado.get("targets_perdidos", 0) > 0, (
        f"la telemetria dice 0 perdidas con un lote perdido: {capturado!r} [{_MARKER}]"
    )
    assert capturado.get("reason") == "partial_loss", (
        f"`reason` colapsa a None en cuanto se escribe algo: {capturado!r} [{_MARKER}]"
    )
