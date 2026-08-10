"""[P1-BUDGET-BANDS-RECALIBRATE · 2026-08-09] La tarjeta del paso 11 sobreestimaba
un 39 % el costo real, y el mínimo aceptaba un número que luego se reprochaba.

MEDIDO contra los planes vivos (21: 30 días, precios completos, ≥20 ítems; se
excluyeron 2 con `items_priced=1/1` — planes ROTOS, no baratos, que arrastraban
la mediana hacia abajo y contaminaron una primera lectura):

    piso de metas ........ RD$ 13.650 / 30 días
    costo real típico .... RD$ 15.747   → 1,15 × piso   (15 de 21 por ENCIMA del piso)
    referencia `medium` .. RD$ 21.840   → sobreestimaba 39 %

EL HALLAZGO QUE ORDENA TODO: el factor real medido (1,15) era EXACTAMENTE el
factor que tenía `low`. Cuando el usuario elegía Moderado, su plan costaba lo
que la banda Económico predecía — la escalera estaba corrida un escalón entera.

Y no era cosmético: con la referencia 39 % alta, `reconcile_budget_with_cost`
decía «dentro» casi siempre (22 de 26). **Un veredicto que no puede fallar no
informa.**

EL MÍNIMO NO SE TOCA — está bien y por la razón correcta: es un PISO DE
VIABILIDAD, no una estimación de gasto, y queda ~15 % por debajo del costo
típico. Un piso por ENCIMA del costo típico bloquearía presupuestos viables.

Tooltip-anchor: P1-BUDGET-BANDS-RECALIBRATE
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parent.parent
_NUTRI = _BACKEND / "nutrition_calculator.py"
_QBUDGET = (_BACKEND.parent / "frontend" / "src" / "components" / "assessment"
            / "questions" / "QBudget.jsx")

if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Medición del 2026-08-09 (ver docstring). El costo típico es 1,15 × piso.
_MEASURED_TYPICAL_OVER_FLOOR = 1.15


def test_the_medium_band_tracks_the_measured_cost():
    """`medium` es la única banda con evidencia: 21 planes, todos de ese tier.
    Debe quedar cerca del costo real medido — ni por debajo (acusaría a planes
    normales) ni muy por encima (el veredicto vuelve a ser un sello de goma)."""
    from nutrition_calculator import _budget_tier_band_factor

    medium = _budget_tier_band_factor("medium")
    assert medium is not None, "P1-BUDGET-BANDS-RECALIBRATE: `medium` sin factor"
    assert _MEASURED_TYPICAL_OVER_FLOOR <= medium <= 1.45, (
        f"P1-BUDGET-BANDS-RECALIBRATE: factor `medium`={medium}. El costo real medido "
        f"es {_MEASURED_TYPICAL_OVER_FLOOR}× el piso; por debajo de eso la referencia "
        "acusa a planes normales, y por encima de ~1.45 volvemos al sello de goma que "
        "este P-fix corrigió (22 de 26 planes daban «dentro» con el factor 1.6)."
    )


def test_the_ladder_stays_monotonic():
    """Económico < Moderado < Alto. Si dos bandas se cruzan, elegir «Alto» podría
    dar una referencia menor que «Económico» — el control dejaría de significar."""
    from nutrition_calculator import _budget_tier_band_factor

    low, medium, high = (_budget_tier_band_factor(t) for t in ("low", "medium", "high"))
    assert low < medium < high, (
        f"P1-BUDGET-BANDS-RECALIBRATE: escalera no monótona: low={low}, "
        f"medium={medium}, high={high}."
    )
    assert low >= 1.0, (
        f"P1-BUDGET-BANDS-RECALIBRATE: `low`={low} < 1.0 — quedaría POR DEBAJO del "
        "piso de viabilidad, que es justo lo que el piso existe para impedir."
    )


def test_the_evidence_limit_is_written_where_someone_would_change_it():
    """`low` y `high` se movieron por COHERENCIA, no por medición: los 21 planes
    son todos `medium`. Quien los retoque tiene que leer eso antes, no después."""
    src = _NUTRI.read_text(encoding="utf-8")
    i = src.index("_BUDGET_TIER_BAND_DEFAULTS")
    bloque = src[max(0, i - 2200):i]
    assert "MEDIDO" in bloque, (
        "P1-BUDGET-BANDS-RECALIBRATE: el bloque de bandas perdió la medición que lo "
        "justifica. Sin ella los números parecen arbitrarios y el siguiente los mueve a ojo."
    )
    assert "todos" in bloque.lower() and "medium" in bloque, (
        "P1-BUDGET-BANDS-RECALIBRATE: falta el LÍMITE DE LA EVIDENCIA (los 21 planes "
        "medidos son todos de tier `medium`; low/high son juicio). Omitirlo convierte "
        "una estimación en un hecho aparente."
    )


def test_the_floor_is_not_touched():
    """El piso NO era el problema — está 15 % por debajo del costo típico, que es
    la dirección correcta para un piso de viabilidad.

    Se ancla la BASE por ciclo, no la salida de `min_budget_for_goals`: esa se
    escala por calorías objetivo y hogar (un perfil sin calorías da un número
    distinto, cosa que este test aprendió fallando). La base es el número que ve
    el usuario en el aviso del mínimo y el que se auditó.
    """
    from nutrition_calculator import _BUDGET_CYCLE_FLOOR_DEFAULTS_DOP

    base_30 = float(_BUDGET_CYCLE_FLOOR_DEFAULTS_DOP[30])
    assert base_30 == 13000.0, (
        f"P1-BUDGET-BANDS-RECALIBRATE: la base del piso de 30 días es {base_30}, se "
        "esperaba 13000. Auditado el 2026-08-09 contra el costo real (típico RD$15.747): "
        "el piso queda ~15 % por debajo, que es la dirección CORRECTA para un piso de "
        "viabilidad. Este P-fix recalibró las BANDAS, no el piso."
    )


def test_the_fx_rate_cannot_age_in_silence():
    """USD se queda (visitantes de EE.UU. en RD). Lo que no puede quedarse es que
    la tasa caduque sin que nadie se entere: convierte mal y no falla nada."""
    src = _NUTRI.read_text(encoding="utf-8")
    fn = re.search(r"def _budget_usd_to_dop\(\).*?\n    return rate\n", src, re.DOTALL)
    assert fn, "P1-BUDGET-FX-STALENESS: no encuentro `_budget_usd_to_dop`"
    body = fn.group(0)
    assert "REVIEWED" in body and "logger.warning" in body, (
        "P1-BUDGET-FX-STALENESS: la tasa USD→DOP volvió a no tener control de edad. "
        "Es un número que caduca y su fallo es SILENCIOSO."
    )
    assert "return rate" in body, (
        "P1-BUDGET-FX-STALENESS: el aviso no debe poder impedir que se devuelva la "
        "tasa — vigilar no es bloquear."
    )


def test_the_minimum_message_also_states_the_expectation():
    """El piso queda 15 % por debajo del costo típico: quien ponía exactamente el
    mínimo acababa en «excedido». El sistema le aceptaba un número y luego lo
    regañaba por él."""
    jsx = _QBUDGET.read_text(encoding="utf-8")
    assert "typicalCost" in jsx, (
        "P1-BUDGET-BANDS-RECALIBRATE: el paso de presupuesto dejó de dar la "
        "expectativa junto al mínimo."
    )
    assert "tierReferences.medium" in jsx, (
        "P1-BUDGET-BANDS-RECALIBRATE: el costo típico debe salir de la referencia del "
        "backend, no de una constante repetida en el frontend — si se duplica, el día "
        "que se recalibren las bandas el formulario seguirá diciendo el número viejo."
    )
