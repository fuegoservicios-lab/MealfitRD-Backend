# -*- coding: utf-8 -*-
"""[P2-I18N-YA-TRADUCIDO-NO-ES-DEGRADACION · 2026-09-08] «Ya está traducido» se reportaba como caída.

`enrich_plan_display` inicializa `last_skip_reason = "no_meals"` y, cuando no queda nada pendiente
—porque el plan **ya está enriquecido**—, el bucle no llega a correr y ese valor inicial sale por el
`return` como si fuera un diagnóstico. `no_meals` no está en `_RAZONES_BENIGNAS`, así que levantaba
`plan_display_i18n_degraded:<locale>` afirmando «el plan se sirve en español canónico».

## Medido sobre el usuario real, que es lo que lo destapó

El único perfil con `locale='fr-FR'` tenía esa alerta viva desde el 05-sep. Su plan `3957a669`:

- las **8 comidas** con `_display['fr-FR']` completo — `name`, `description`, `ingredients`, `recipe`
  («Avoine crémeuse avec œuf, raisins et cannelle»);
- el nombre del plan y los insights, también en francés.

O sea: **completamente traducido**, y la alerta llevaba tres días diciendo lo contrario. Era la única
warning de aspecto real en la vista del operador —las otras 13 apuntan a planes borrados— y no
describía nada.

*Un valor inicial que sale por un `return` no es un diagnóstico: es lo que quedaba en la variable.*

## Lo que NO se hizo, a propósito

`no_meals` **sigue siendo no benigno**. También lo devuelve el caso de índices de día vacíos
(`_normalize_day_indices` → lista vacía), que sí es un problema real. Meter `no_meals` entero en la
lista de benignos habría silenciado esa rama. Lo que se separa es el desenlace bueno, con nombre
propio.
"""
import inspect

import pytest

import plan_display_i18n as pi


def test_already_enriched_es_benigno():
    assert "already_enriched" in pi._RAZONES_BENIGNAS


def test_no_meals_SIGUE_sin_ser_benigno():
    """La rama de índices de día vacíos también devuelve `no_meals` y ésa sí es un problema.

    Si alguien «simplifica» metiendo `no_meals` en la lista de benignos, silencia un fallo real —
    que es exactamente el error opuesto al que este P-fix corrige.
    """
    assert "no_meals" not in pi._RAZONES_BENIGNAS


def test_el_corte_ocurre_ANTES_de_entrar_al_bucle():
    """El `return` va tras el encolado del lote vacío (nombre/insights pendientes) y antes del
    `while`: si se colocara antes, un plan al que sólo le falta el NOMBRE saldría como ya
    enriquecido y nunca se traduciría."""
    src = inspect.getsource(pi.enrich_plan_display)
    i_encola = src.index("_pendientes.append([])")
    i_corte = src.index('return {"enriched_meals": 0, "skipped": "already_enriched"}')
    i_bucle = src.index("while _pendientes:")
    assert i_encola < i_corte < i_bucle, (
        "el corte de `already_enriched` cambió de sitio: antes del encolado se traga los planes a "
        "los que sólo les falta nombre o insights")


def test_el_motivo_inicial_sigue_siendo_no_meals():
    """No se toca el valor inicial: hay ramas que dependen de él (bucle que corre y no consigue

    nada). Lo que cambia es que el camino «no había nada que hacer» ya no lo hereda.
    """
    src = inspect.getsource(pi.enrich_plan_display)
    assert 'last_skip_reason = "no_meals"' in src


@pytest.mark.parametrize("razon,alerta", [
    ("already_enriched", False),
    ("ok", False),
    ("dedupe_locked", False),
    ("no_meals", True),
    ("llm_exception", True),
    ("json_parse_error", True),
])
def test_solo_los_motivos_REALES_alertan(razon, alerta, monkeypatch):
    """El emisor filtra por `_RAZONES_BENIGNAS`; se comprueba el filtro completo, no sólo el caso
    nuevo — un test que sólo mira lo que acaba de cambiar no ve lo que el cambio rompió."""
    escrituras = []
    monkeypatch.setattr(pi, "execute_sql_write", lambda *a, **k: escrituras.append(a))
    pi._emit_degraded_alert("plan-x", "user-x", "fr-FR", razon)
    assert bool(escrituras) is alerta, f"{razon}: alerta={bool(escrituras)}, esperado {alerta}"
