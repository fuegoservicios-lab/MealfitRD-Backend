"""[P1-SLOT-FRACTIONS-KEY · 2026-07-26] El split fisiológico llevaba seis semanas inerte.

El owner reportó meriendas más grandes que el almuerzo. Medido sobre el plan vivo `cd08ea3c`:

    Día 1:  desayuno 499 · almuerzo 503 · MERIENDA 754 · cena 378
    Día 2:  desayuno 457 · almuerzo 483 · MERIENDA 911 · cena 233

Los totales del día son excelentes (2084-2143 contra 2100, banda 1.00): el motor clavaba la SUMA
redistribuyendo hacia la merienda, porque nadie restringía la FORMA del día.

## La causa

`_canonical_slot_fractions` leía `m.get("slot")` y esa clave **no existe**. Las comidas traen
`cals/carbs/desc/fats/ingredients/macros/`**`meal`**`/name/...`. Con `key=None` en todas, cada
comida caía a la rama "no mapeado" y recibía parte igual del remanente:

    devuelto:  0,25 / 0,25 / 0,25 / 0,25      ← plano
    canónico:  0,20 / 0,35 / 0,15 / 0,30      ← MEAL_SLOT_SPLITS

O sea que `P3-SLOT-DISTRIBUTION` (2026-06-13) nunca llegó a ejecutarse, y su propio docstring dice
qué venía a arreglar: *"el desayuno concentraba 48% de las kcal y 62% de la proteína del día"*.

⚠️ `_detect_slot_appropriateness`, en el mismo archivo, ya leía `meal` correctamente. **Dos
funciones hermanas leyendo la misma cosa por claves distintas** — tercera vez en la sesión que
aparece esa forma (`unit_mismatch` en el guard de coherencia, filler tokens en el matcher de
fidelidad).

Incidencia medida sobre 22 días de 8 planes ANTES del fix:

    merienda > almuerzo      27,3%
    merienda > 30% del día   13,6%     (mediana real 16%, máximo 43,7%)
"""
import pytest

import graph_orchestrator as go
from nutrition_calculator import MEAL_SLOT_SPLITS


def _dia(claves="meal"):
    """4 comidas con el slot en la clave indicada."""
    return [{claves: s, "name": f"Plato {s}"} for s in ("Desayuno", "Almuerzo", "Merienda", "Cena")]


# ───────────── 1. el split canónico se aplica ─────────────

def test_devuelve_el_split_fisiologico_no_el_plano():
    fr = go._canonical_slot_fractions(_dia("meal"))
    esperado = MEAL_SLOT_SPLITS[4]
    assert fr == pytest.approx(
        [esperado["desayuno"], esperado["almuerzo"], esperado["merienda"], esperado["cena"]])


def test_la_merienda_pesa_MENOS_que_el_almuerzo():
    """La inversión que reportó el owner: 15% vs 35%."""
    d, a, m, c = go._canonical_slot_fractions(_dia("meal"))
    assert m < a, f"merienda {m} debería pesar menos que almuerzo {a}"
    assert m < c and m < d, "la merienda es el slot más ligero del día"


def test_NO_es_el_reparto_plano():
    """Guard del bug exacto: 0.25 en las cuatro era la rama de fallback."""
    fr = go._canonical_slot_fractions(_dia("meal"))
    assert len(set(round(f, 4) for f in fr)) > 1, f"reparto plano = el bug: {fr}"


# ───────────── 2. compatibilidad y robustez ─────────────

def test_tambien_acepta_la_clave_slot():
    """Alguna superficie podría emitir `slot`; se leen las dos."""
    assert go._canonical_slot_fractions(_dia("slot")) == pytest.approx(
        go._canonical_slot_fractions(_dia("meal")))


def test_meal_gana_si_ambas_existen_y_discrepan():
    """Con 4 comidas (el split cambia según cuántas haya): si mandara `slot`, la primera sería
    merienda (0,15) y quedaría por DEBAJO de la última; mandando `meal` es almuerzo (0,35)."""
    ms = [{"meal": "Almuerzo", "slot": "Merienda", "name": "a"},
          {"meal": "Cena", "slot": "Desayuno", "name": "b"},
          {"meal": "Desayuno", "slot": "Cena", "name": "c"},
          {"meal": "Merienda", "slot": "Almuerzo", "name": "d"}]
    fr = go._canonical_slot_fractions(ms)
    esperado = MEAL_SLOT_SPLITS[4]
    assert fr[0] == pytest.approx(esperado["almuerzo"]), "debe mandar `meal`, no `slot`"
    assert fr[3] == pytest.approx(esperado["merienda"])


def test_slots_desconocidos_no_revientan():
    fr = go._canonical_slot_fractions([{"meal": "Brunch", "name": "x"}, {"meal": "Cena", "name": "y"}])
    assert len(fr) == 2 and all(f is not None and f > 0 for f in fr)


def test_siempre_suma_uno():
    """Preservar el total diario es la invariante: sólo se redistribuye la FORMA."""
    for n in (2, 3, 4, 5, 6):
        ms = [{"meal": s, "name": "x"} for s in
              (["Desayuno", "Almuerzo", "Merienda", "Cena", "Merienda 2", "Merienda 3"][:n])]
        assert sum(go._canonical_slot_fractions(ms)) == pytest.approx(1.0), n


def test_lista_vacia():
    assert go._canonical_slot_fractions([]) == []
