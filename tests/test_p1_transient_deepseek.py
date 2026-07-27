"""[P1-TRANSIENT-DEEPSEEK · 2026-07-27] Un hipo de red desactivaba PRO para todos.

## La cadena, medida en los logs del VPS (6 horas)

    25 correcciones del self-critique intentadas
    12 perdidas:  6 pro_error:APIConnectionError  +  6 pro_cb_open
     2 regeneraciones COMPLETAS de plan

Los 6 `pro_cb_open` son consecuencia de los 6 primeros: los errores de conexión abrían el
circuit breaker de `deepseek-v4-pro`, y a partir de ahí el breaker rechazaba las siguientes
correcciones sin ni intentarlo.

Y no se queda en "calidad un poco peor". El caso vivo `corr=308f99c2`:

    self-critique detecta Diversidad 3/10 -> decide corregir días 2 y 3
    las DOS correcciones se pierden (PRO caído + CB abierto)
    el plan conserva el defecto
    REVISOR: "MISMA PROTEÍNA REPETIDA EL MISMO DÍA" -> rechazo severidad high
    regeneración COMPLETA del plan

O sea: el fallo se paga regenerando el plan entero — esqueleto + 3-4 días + revisión — no con
una llamada de más.

## La causa

`_is_transient_upstream_error` se escribió el 2026-05-21 para las firmas de **Google**
(`ServiceUnavailable`, `BadGateway`, 502/503/504, cadenas gRPC). El proyecto migró a DeepSeek el
2026-06-12 (P0-DEEPSEEK-MIGRATION) y el clasificador **nunca se actualizó** a la taxonomía del
cliente OpenAI-compatible. `APIConnectionError` —un fallo de red puro— se contaba como mala
salud del MODELO.

Es literalmente el bug que esta función existe para evitar, descrito en su propio docstring
("el CB contó esos 3 retries como fallas → abrió el modelo → los días cayeron aunque el modelo
estaba sano"), reaparecido con otro proveedor.

⚠️ El alcance es mayor que el corrector: el CB es per-modelo, así que abrirlo por un problema de
red deja sin PRO también al **revisor médico**, que va a PRO en TODOS los tiers.

Coste del fix: cero llamadas LLM extra. Solo deja de castigar al modelo por la red.

tooltip-anchor: P1-TRANSIENT-DEEPSEEK
"""
from __future__ import annotations

import pytest

import graph_orchestrator as g


class APIConnectionError(Exception):
    """Mismo NOMBRE que la del cliente OpenAI/DeepSeek — el clasificador matchea por nombre
    de tipo porque LangChain envuelve estos errores de forma inconsistente entre versiones."""


class APITimeoutError(Exception):
    pass


class RemoteProtocolError(Exception):
    pass


# ───────────── 1. lo que ANTES abría el breaker ─────────────

@pytest.mark.parametrize("exc", [
    APIConnectionError("Connection error."),
    APITimeoutError("Request timed out."),
    RemoteProtocolError("Server disconnected without sending a response."),
])
def test_los_fallos_de_RED_son_transitorios(exc):
    """Un error de transporte no dice NADA sobre la salud del modelo. Contarlo como falla abre
    el CB y deja sin PRO al corrector Y al revisor médico."""
    assert g._is_transient_upstream_error(exc) is True, (
        f"{type(exc).__name__} vuelve a contar como falla del modelo: un hipo de red "
        f"desactivará PRO para todos otra vez."
    )


def test_el_caso_exacto_de_los_logs():
    """`pro_error:APIConnectionError`, 6 veces en 6 h, y los 6 `pro_cb_open` que provocó."""
    assert g._is_transient_upstream_error(APIConnectionError("Connection error.")) is True


# ───────────── 2. no se ablandó la detección de fallo REAL ─────────────

class BadRequestError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class RateLimitError(Exception):
    pass


@pytest.mark.parametrize("exc", [
    BadRequestError("400 invalid request: prompt too long"),
    AuthenticationError("401 invalid api key"),
    ValueError("respuesta no parseable"),
    RuntimeError("algo se rompió de verdad"),
])
def test_los_fallos_REALES_siguen_contando(exc):
    """El CB tiene que seguir abriéndose cuando el modelo/credencial está mal de verdad. Si esto
    devolviera True, el breaker no protegería de nada."""
    assert g._is_transient_upstream_error(exc) is False, (
        f"{type(exc).__name__} se está tratando como transitorio: el CB dejaría de proteger."
    )


# ───────────── 3. lo de Google sigue cubierto ─────────────

class ServiceUnavailable(Exception):
    pass


class BadGateway(Exception):
    pass


@pytest.mark.parametrize("exc", [
    ServiceUnavailable("503"),
    BadGateway("502 Bad Gateway"),
    Exception("upstream returned 504 gateway timeout"),
])
def test_las_firmas_de_google_no_se_perdieron(exc):
    """P1-LLM-TRANSIENT-5XX sigue vigente: añadir un proveedor no quita el anterior."""
    assert g._is_transient_upstream_error(exc) is True


# ───────────── 4. ancla ─────────────

def test_el_clasificador_nombra_al_cliente_actual():
    """Ancla de la CLASE: si mañana se cambia de proveedor otra vez, que este test recuerde que
    ESTA función hay que actualizarla — es el fallo que costó 2 regeneraciones de plan."""
    import inspect
    src = inspect.getsource(g._is_transient_upstream_error)
    assert "APIConnectionError" in src, (
        "el clasificador perdió la taxonomía del cliente OpenAI-compatible (DeepSeek)"
    )
    assert "P1-TRANSIENT-DEEPSEEK" in src


def test_el_helper_del_cb_lo_consulta():
    """El clasificador solo sirve si `_record_cb_failure_unless_transient` lo llama."""
    import inspect
    src = inspect.getsource(g._record_cb_failure_unless_transient)
    assert "_is_transient_upstream_error" in src
