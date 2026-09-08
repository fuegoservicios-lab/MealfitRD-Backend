# -*- coding: utf-8 -*-
"""[P1-I18N-BUDGET-DICE-LA-CAUSA · 2026-09-08] El techo agotado es el síntoma, no el diagnóstico.

El plan `3957a669` —del ÚNICO usuario fr-FR— se sirve en español. La alerta y la telemetría decían
`invocation_budget_exhausted`, que manda al operador a subir el techo. Los números decían otra cosa:

    batches: 2 · retries: 6 · targets: 16 · mismatch: 0 · meals_written: 0
    tokens_estimated: 0 · duration_ms: 977.646   (tres ciclos el 08-sep, casi idénticos)

`tokens_estimated: 0` y `mismatch: 0` juntos significan que **ninguna llamada volvió**: no es que el
modelo devolviera basura (eso daría mismatch) ni que el techo fuera corto (eso daría tokens). Y
977 s / 8 invocaciones ≈ 122 s con un timeout de 60 s = el cliente reintenta una vez y las dos
mueren. La causa está fuera: el proveedor iba degradado ese día — el mismo que devolvió 429
«service temporarily overloaded» en el experimento de recetas de la misma tarde.

Subir el techo con ese cuadro compra ocho timeouts más.

## Qué se arregla aquí, y qué NO

Se arregla el DIAGNÓSTICO: agotar el techo por ocho excepciones y agotarlo porque el modelo devuelve
basura reportaban la misma razón, y sólo una de las dos se arregla subiendo el techo.

NO se toca el timeout ni el tamaño del lote: son la respuesta a una condición externa medida UN día.
Cambiarlos por eso sería ajustar el sistema a la meteorología de una tarde.

Es el mismo defecto de forma que `P2-ALERT-MESSAGE-REFRESH` y `P2-I18N-YA-TRADUCIDO` cerraron hoy:
*un diagnóstico que nombra el síntoma dirige la investigación al sitio equivocado con toda la
confianza.*
"""
import inspect

import plan_display_i18n as p


def test_el_contador_de_excepciones_existe_y_se_incrementa():
    src = inspect.getsource(p)
    assert "_excepciones_llm = 0" in src, "desapareció el contador"
    assert "_excepciones_llm += 1" in src, "el contador ya no se incrementa en el handler"


def test_la_razon_distingue_techo_de_proveedor_caido():
    src = inspect.getsource(p)
    assert "invocation_budget_exhausted_llm_errors" in src, (
        "volvió a colapsar las dos causas en una sola razón: un operador que la lea subirá el "
        "techo, que es lo que NO arregla el caso del proveedor caído")
    i_gen = src.index('"invocation_budget_exhausted"')
    i_esp = src.index('"invocation_budget_exhausted_llm_errors"')
    assert i_esp < i_gen, (
        "la rama específica debe evaluarse ANTES de la genérica; al revés nunca se alcanza")


def test_el_contador_viaja_en_la_telemetria_SIEMPRE():
    """No sólo cuando el techo se agota: un ciclo que acaba bien tras dos timeouts también lo
    cuenta, y esa serie es la que avisa antes de que el usuario vea su plan en español."""
    src = inspect.getsource(p)
    assert '"llm_exceptions": _excepciones_llm,' in src, (
        "el contador no llega a `pipeline_metrics`: sin la serie no hay aviso temprano")


def test_la_razon_nueva_NO_es_benigna():
    """Si entrara en la lista de benignas, el fix sería un silenciador con otro nombre."""
    assert "invocation_budget_exhausted_llm_errors" not in p._RAZONES_BENIGNAS
    assert "invocation_budget_exhausted" not in p._RAZONES_BENIGNAS


def test_no_se_toco_el_timeout_ni_el_lote():
    """Anclaje de lo que se decidió NO hacer. Los dos son la respuesta tentadora al síntoma y los
    dos ajustarían el sistema a la meteorología de una tarde."""
    assert p._plan_display_i18n_timeout_s() == 60.0, (
        "cambió el timeout: si es deliberado, mide primero varios días de proveedor, no uno")
    src = inspect.getsource(p._max_invocaciones_por_ciclo)
    assert "* 3 + 2" in src, "cambió el techo de invocaciones sin medir la causa"
