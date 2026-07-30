"""[P1-COHERENCE-INF-NOT-SEVERE · 2026-07-30] Un plan entero bloqueado por sal y pimienta negra.

Caso vivo, medido en el journal de producción (plan 4d2c1111, semana 2, 2026-07-30 02:01/02:16/02:31):

    [COH-GUARD/warn] 2 divergencias (presence=0, magnitude=2) Hipótesis {'recipe_unquantified': 2}
                     Sample: Pimienta negra [magnitude]; Sal [magnitude]
    [P2-COHERENCE-1] T2 coherence block_severe_only escaló warn→block (week=2, divergences=2)
    [CHUNK/GAP2] Shopping list fallo 3 veces
    → el usuario se quedó SIN lista de compras de la semana 2.

La cadena: la receta dice "sal al gusto" sin gramos ⇒ `expected_qty = 0` ⇒ el guard escribe
`delta_pct = float("inf")` (es su única rama con expected 0) ⇒ `_has_severe_divergence` evalúa
`abs(inf) > 0.50`, que es verdadero trivialmente ⇒ escala warn→block ⇒ retry ⇒ 3 fallos.

**Un delta infinito no es una magnitud: es un denominador que falta.** "La receta no dijo cuánto" se
estaba leyendo como la magnitud más severa posible. Y por construcción esa rama es SOBRE-oferta (la
receta no pide nada, la lista tiene algo), que es justo lo que P1-COHERENCE-SEVERE-NO-NOISE
(2026-07-07) ya declaró no-severo para `unknown`/`pantry_overdeduct` — a `recipe_unquantified` no
llegó.

Y es la clase de P1-TRANSFORM-GATE-PARITY otra vez: **dos tests de severidad para el mismo concepto y
el endurecimiento aterrizó en uno solo.** La ruta de review pasa por
`graph_orchestrator._coherence_finite_abs_delta`, cuyo nombre lleva "finite" y mapea inf/NaN → 0.0;
la de T2 leía `delta_pct` en crudo. Estos tests anclan la PARIDAD entre las dos.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

import shopping_calculator as sc

_BACKEND = Path(__file__).resolve().parents[1]
_SC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


def _div(food, hyp, delta, magnitude=True, expected=0.0, actual=5.0):
    return {"food": food, "hypothesis": hyp, "magnitude": magnitude, "delta_pct": delta,
            "expected_qty": expected, "actual_qty": actual}


# --------------------------------------------------- el caso vivo

def test_el_plan_de_sal_y_pimienta_ya_no_bloquea():
    """Reproducción literal del plan 4d2c1111 semana 2."""
    vivas = [_div("Pimienta negra", "recipe_unquantified", float("inf")),
             _div("Sal", "recipe_unquantified", float("inf"))]
    assert not sc._has_severe_divergence(vivas), (
        "sal y pimienta sin cuantificar NO pueden bloquear la lista de compras de una semana")


@pytest.mark.parametrize("hyp", ["recipe_unquantified", "unit_mismatch", "yield_uncovered", "unknown"])
def test_ningun_delta_infinito_es_severo(hyp):
    """`inf` viene SIEMPRE de expected_qty=0, o sea sobre-oferta. Ninguna hipótesis lo convierte en
    severo por magnitud."""
    assert not sc._has_severe_divergence([_div("Sal", hyp, float("inf"))])


def test_nan_tampoco():
    assert not sc._has_severe_divergence([_div("Sal", "recipe_unquantified", float("nan"))])


# --------------------------------------------------- lo que NO se debilita

def test_cap_swallowed_sigue_bloqueando_aunque_su_delta_sea_infinito():
    """El caso que de verdad rompe un plan: la receta menciona pollo y la lista NO lo tiene. Sale
    por nombre de hipótesis, sin mirar el delta — si esto se rompe, el fix fue demasiado lejos."""
    assert sc._has_severe_divergence(
        [_div("Pollo", "cap_swallowed_modifier", float("inf"), magnitude=False)])


def test_magnitud_finita_severa_sigue_bloqueando():
    """Una lista con el doble/la mitad de lo que la receta pide sigue siendo severa."""
    assert sc._has_severe_divergence(
        [_div("Arroz blanco", "yield_uncovered", 1.4, expected=500.0, actual=1200.0)])
    assert sc._has_severe_divergence(
        [_div("Arroz blanco", "unit_mismatch", -0.75, expected=800.0, actual=200.0)])


def test_magnitud_finita_leve_no_bloquea():
    assert not sc._has_severe_divergence(
        [_div("Arroz blanco", "yield_uncovered", 0.20, expected=500.0, actual=600.0)])


def test_mezcla_realista_bloquea_por_lo_que_debe():
    """Con sal (inf) + un cap_swallowed real, debe bloquear POR el cap_swallowed."""
    mezcla = [_div("Sal", "recipe_unquantified", float("inf")),
              _div("Pollo", "cap_swallowed_modifier", float("inf"), magnitude=False)]
    assert sc._has_severe_divergence(mezcla)
    # …y sin el cap_swallowed, no bloquea nada
    assert not sc._has_severe_divergence([mezcla[0]])


# --------------------------------------------------- paridad entre las dos rutas

def test_paridad_con_la_ruta_de_review():
    """La ruta de review (graph_orchestrator) y la de T2 (aquí) deben coincidir en que un delta
    infinito no es severo. Que UNA de las dos estuviera endurecida es el bug de fondo."""
    import graph_orchestrator as go
    d = _div("Sal", "recipe_unquantified", float("inf"))
    assert go._coherence_finite_abs_delta(d) == 0.0, (
        "la ruta de review dejó de neutralizar inf — se rompió por el otro lado")
    assert not sc._has_severe_divergence([d]), "la ruta de T2 volvió a leer inf como severo"


def test_el_guard_de_inf_esta_anclado_en_el_fuente():
    """Si alguien quita el guard, este test cae antes de que un plan vuelva a bloquearse por sal."""
    i = _SC.index("def _has_severe_divergence")
    body = _SC[i:_SC.index("\ndef ", i + 10)]
    assert 'abs(delta) == float("inf")' in body, (
        "volvió el `abs(delta) > threshold` sin filtrar el infinito ⇒ 'la receta no dijo cuánto' "
        "vuelve a leerse como la magnitud más severa posible")
    assert "delta != delta" in body, "falta el guard de NaN"


def test_el_umbral_finito_sigue_siendo_el_de_siempre():
    assert sc._COHERENCE_SEVERE_MAGNITUDE_THRESHOLD == 0.50
