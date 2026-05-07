"""[P0-11] Tests para garantizar que `aggregate_and_deduct_shopping_list` no
produce entradas fantasma cuando `consumed > plan` en peso.

Bug original (audit P0-11):
  En `aggregate_and_deduct_shopping_list`, las cantidades se acumulan vía
      aggregated[name][unit] += plan_qty * multiplier
      aggregated[name][unit] -= consumed_qty
  Cuando consumed_qty (g/kg/oz/lb/ml/l) excede plan_qty para el mismo
  ingrediente, el net por unidad puede ser negativo, y `weight_in_lbs` neto
  queda negativo tras sumar todas las unidades de peso.

  El branch `if weight_in_lbs > 0.0001:` evitaba correctamente agregar la
  entrada de peso. PERO el `for u, q in list(units.items())` posterior
  iteraba sobre las unidades NO-peso (ej. 'unidad', 'paquete') aún
  positivas — agregando una entrada fantasma "1 Ud." aunque el peso
  planificado ya estuviera cubierto 100% por el consumed.

  Caso clásico: el LLM expresa el mismo aporte como peso + unidad
  ("1 cebolla mediana" + "200g cebolla"); cuando el usuario consume
  300g, el peso queda cubierto pero la "1 cebolla" residual produce
  un item fantasma en la lista de compras.

Fix:
  Si `has_weight and weight_in_lbs < 0`:
    1. Clamp `weight_in_lbs = 0.0` (defensa simétrica para futuros consumers).
    2. Reset `units = {}` para suprimir entradas residuales.
  Log INFO `[P0-11/CLAMP]` con detalles para SRE.

Cobertura:
  - test_phantom_unit_entry_suppressed_when_consumed_exceeds_plan_weight
  - test_no_clamp_when_net_weight_positive
  - test_no_clamp_when_no_weight_at_all
  - test_clamp_logs_marker_for_sre
  - test_aggregator_source_contains_p0_11_clamp
"""
import inspect
import logging

import pytest

import shopping_calculator
from shopping_calculator import aggregate_and_deduct_shopping_list


# ---------------------------------------------------------------------------
# 1. Bug repro: el ítem fantasma queda suprimido tras el clamp.
# ---------------------------------------------------------------------------
def test_phantom_unit_entry_suppressed_when_consumed_exceeds_plan_weight():
    """Plan: '1 cebolla mediana' + '100g cebolla'; consumed: '300g cebolla'.
    El usuario YA consumió 300g (más que el peso planificado, 100g).
    Antes del fix: la lista incluía '1 Ud. cebolla' (fantasma).
    Tras el fix: NO debe aparecer cebolla en la lista."""
    plan_ingredients = [
        "1 unidad de cebolla",
        "100 g de cebolla",
    ]
    consumed_ingredients = [
        "300 g de cebolla",
    ]
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients, consumed_ingredients,
        categorize=False, structured=True,
    )
    cebolla_items = [
        r for r in result
        if isinstance(r, dict) and "cebolla" in r.get("name", "").lower()
    ]
    assert cebolla_items == [], (
        f"P0-11 regression: cebolla NO debe aparecer en lista cuando "
        f"consumed (300g) excede plan (100g + 1 unidad). Got: {cebolla_items}"
    )


def test_phantom_paquete_entry_suppressed_when_weight_overconsumed():
    """Mismo patrón con paquete en lugar de unidad."""
    plan_ingredients = [
        "1 paquete de arroz",
        "200 g de arroz",
    ]
    consumed_ingredients = [
        "1500 g de arroz",  # mucho más que el plan
    ]
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients, consumed_ingredients,
        categorize=False, structured=True,
    )
    arroz_items = [
        r for r in result
        if isinstance(r, dict) and "arroz" in r.get("name", "").lower()
    ]
    assert arroz_items == [], (
        f"P0-11 regression: arroz NO debe aparecer cuando consumed (1500g) "
        f"excede plan total (200g + 1 paquete). Got: {arroz_items}"
    )


# ---------------------------------------------------------------------------
# 2. Casos donde el clamp NO debe activarse.
# ---------------------------------------------------------------------------
def test_no_clamp_when_net_weight_positive():
    """Plan: '500g pollo' + '1 unidad pollo'; consumed: '200g pollo'.
    Net: 300g + 1 unidad → debe aparecer en la lista (caso normal)."""
    plan_ingredients = [
        "500 g de pollo",
    ]
    consumed_ingredients = [
        "200 g de pollo",
    ]
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients, consumed_ingredients,
        categorize=False, structured=True,
    )
    pollo_items = [
        r for r in result
        if isinstance(r, dict) and "pollo" in r.get("name", "").lower()
    ]
    assert pollo_items, "P0-11: net positivo (300g) DEBE producir entrada"
    # Verifica que la cantidad numérica es positiva.
    mq = pollo_items[0].get("market_qty_numeric", pollo_items[0].get("market_qty"))
    assert isinstance(mq, (int, float)) and mq > 0, \
        f"market_qty_numeric debe ser positivo, got {mq}"


def test_no_clamp_when_no_weight_at_all():
    """Plan: '2 unidades de manzana' (sin peso); consumed vacío.
    No hay weight_in_lbs (has_weight=False), no entra al branch del clamp.
    Las 2 unidades se preservan."""
    plan_ingredients = [
        "2 unidades de manzana",
    ]
    consumed_ingredients = []
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients, consumed_ingredients,
        categorize=False, structured=True,
    )
    manzana_items = [
        r for r in result
        if isinstance(r, dict) and "manzana" in r.get("name", "").lower()
    ]
    assert manzana_items, "Plan sin peso debe preservar las 2 unidades"


def test_no_clamp_marker_when_net_weight_positive(caplog):
    """Verificación negativa: el marker `[P0-11/CLAMP]` NO debe aparecer
    cuando el net peso es positivo. Cualquier disparo accidental (e.g., el
    clamp activándose en casos no edge) lo detectamos vía caplog."""
    plan_ingredients = [
        "1000 g de pollo",  # net peso muy positivo, garantizado
    ]
    consumed_ingredients = [
        "200 g de pollo",
    ]
    with caplog.at_level(logging.INFO):
        aggregate_and_deduct_shopping_list(
            plan_ingredients, consumed_ingredients,
            categorize=False, structured=True,
        )
    clamp_logs = [r for r in caplog.records if "[P0-11/CLAMP]" in r.message]
    pollo_clamp = [r for r in clamp_logs if "pollo" in r.message.lower()]
    assert not pollo_clamp, (
        f"P0-11: clamp NO debe activarse para `pollo` con net peso positivo (800g). "
        f"Logs ofensores: {[r.message for r in pollo_clamp]}"
    )


# ---------------------------------------------------------------------------
# 3. Telemetría: log marker [P0-11/CLAMP] dispara para SRE.
# ---------------------------------------------------------------------------
def test_clamp_logs_marker_for_sre(caplog):
    """Cuando el clamp aplica, `[P0-11/CLAMP]` debe aparecer en logs INFO
    para que SRE detecte la frecuencia del caso edge."""
    plan_ingredients = [
        "1 unidad de zanahoria",
        "50 g de zanahoria",
    ]
    consumed_ingredients = [
        "200 g de zanahoria",
    ]
    with caplog.at_level(logging.INFO):
        aggregate_and_deduct_shopping_list(
            plan_ingredients, consumed_ingredients,
            categorize=False, structured=True,
        )
    matches = [r for r in caplog.records if "[P0-11/CLAMP]" in r.message]
    assert matches, \
        f"P0-11: log marker [P0-11/CLAMP] no apareció. Mensajes: {[r.message for r in caplog.records[-5:]]}"


# ---------------------------------------------------------------------------
# 4. Defensa textual contra reintroducción del bug.
# ---------------------------------------------------------------------------
def test_aggregator_source_contains_p0_11_clamp():
    """Defensa: el código fuente debe contener el clamp `if has_weight and
    weight_in_lbs < 0:` con reset de units. Si alguien lo borra, este test
    rompe."""
    src = inspect.getsource(aggregate_and_deduct_shopping_list)
    assert "[P0-11" in src or "P0-11" in src, \
        "P0-11: comentario de auditoría debe documentar el clamp"
    # El clamp asignación + reset units son las dos lineas críticas.
    import re
    assert re.search(r"weight_in_lbs\s*=\s*0\.0", src), \
        "P0-11: falta clamp `weight_in_lbs = 0.0`"
    # Reset units a dict vacío.
    assert re.search(r"units\s*=\s*\{\}", src), \
        "P0-11: falta reset `units = {}` tras clamp"
