"""[P1-2] Tests para garantizar que `aggregate_and_deduct_shopping_list` trata
plan e inventario simétricamente en peso literal, sin sesgar el delta hacia
OVER-BUYING vía `_calculate_yield_multiplier` aplicado asimétricamente.

Bug original (audit P1-2):
  El `_parse_quantity` aplicaba `_calculate_yield_multiplier` automáticamente
  cuando el texto contenía adjetivos como "cocido"/"asado"/"hervido" + un
  ingrediente proteico/granoso. Eso convertía peso-final-cocido a peso-crudo
  necesario (e.g., "1 lb pollo cocido" → 1.35 lb). PERO la conversión solo
  disparaba textualmente:
    - plan_ingredients del LLM frecuentemente: "1 lb pollo cocido" → 1.35 lb
    - physical_inventory tipeado por user: "5 lb pollo" → 5.0 lb (sin yield)
  La asimetría textual sesgaba el delta hacia OVER-BUYING — el plan se
  inflaba a peso crudo necesario, mientras el inventario quedaba en peso
  literal sin compensación.

Fix:
  - `_parse_quantity` acepta `apply_yield_multiplier: bool = True` (kw-only).
    Default mantiene el comportamiento existente para los call-sites que
    dependen de yield (tools.py, cron_tasks.py, db_inventory.py, etc).
  - `aggregate_and_deduct_shopping_list` lo invoca con
    `apply_yield_multiplier=False` para AMBOS plan_ingredients y
    consumed_ingredients → peso literal en ambos lados, comparable.

Cobertura:
  - test_parse_quantity_default_applies_yield_for_legacy_callers
  - test_parse_quantity_with_flag_false_skips_yield
  - test_aggregator_treats_plan_and_inventory_symmetrically
  - test_aggregator_uses_apply_yield_false_internally
  - test_no_overbuy_when_plan_says_cocido_inventory_says_raw
  - test_yield_multiplier_helper_unchanged
"""
import inspect

import pytest

import shopping_calculator
from shopping_calculator import (
    _parse_quantity,
    _calculate_yield_multiplier,
    aggregate_and_deduct_shopping_list,
)


# ---------------------------------------------------------------------------
# 1. Contrato de `_parse_quantity` con el nuevo flag.
# ---------------------------------------------------------------------------
def test_parse_quantity_default_applies_yield_for_legacy_callers():
    """Default `apply_yield_multiplier=True` preserva yield para los
    call-sites históricos que dependen del comportamiento existente
    (tools.py, cron_tasks.py, db_inventory.py)."""
    qty, unit, name = _parse_quantity("1 lb pollo cocido")
    # Yield 1.35 para proteína cocida (línea 343-345 de _calculate_yield_multiplier).
    assert abs(qty - 1.35) < 0.01, f"yield default debe aplicar: esperado ~1.35, got {qty}"


def test_parse_quantity_with_flag_false_skips_yield():
    """Con `apply_yield_multiplier=False`, el qty es literal."""
    qty, unit, name = _parse_quantity("1 lb pollo cocido", apply_yield_multiplier=False)
    assert qty == 1.0, f"con flag False, qty debe ser literal 1.0, got {qty}"


def test_parse_quantity_flag_no_op_for_strings_without_yield_keywords():
    """Si el texto no tiene "cocido"/"asado"/etc, yield_mult=1.0 y el flag
    no cambia comportamiento."""
    qty1, _, _ = _parse_quantity("5 lb pollo")
    qty2, _, _ = _parse_quantity("5 lb pollo", apply_yield_multiplier=False)
    assert qty1 == qty2 == 5.0


def test_parse_quantity_flag_is_keyword_only():
    """`apply_yield_multiplier` debe ser kw-only para evitar passing
    posicional accidental que cambie semántica silenciosamente."""
    sig = inspect.signature(_parse_quantity)
    param = sig.parameters.get("apply_yield_multiplier")
    assert param is not None
    assert param.kind == inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# 2. Helper de yield NO cambió (defensa contra regresión silenciosa).
# ---------------------------------------------------------------------------
def test_yield_multiplier_helper_unchanged():
    """El helper `_calculate_yield_multiplier` mantiene su contrato:
    solo aplica para textos específicos. Esto NO es un cambio P1-2 — es
    defensa contra que el fix se confunda y modifique el helper en lugar
    del flag."""
    assert _calculate_yield_multiplier("pollo cocido") == 1.35
    assert _calculate_yield_multiplier("arroz cocido") == 0.35
    assert _calculate_yield_multiplier("plátano pelado") == 1.30
    assert _calculate_yield_multiplier("pollo sin hueso") == 1.40
    # Sin keywords → yield neutro.
    assert _calculate_yield_multiplier("pollo") == 1.0
    assert _calculate_yield_multiplier("zanahoria") == 1.0


# ---------------------------------------------------------------------------
# 3. Aggregator: simetría real plan↔inventario.
# ---------------------------------------------------------------------------
def test_aggregator_uses_apply_yield_false_internally():
    """Defensa textual: el source de `aggregate_and_deduct_shopping_list`
    debe llamar `_parse_quantity` con `apply_yield_multiplier=False`."""
    src = inspect.getsource(aggregate_and_deduct_shopping_list)
    # Debe haber llamadas explícitas con el flag False (al menos 2: plan + consumed).
    occurrences = src.count("apply_yield_multiplier=False")
    assert occurrences >= 2, (
        f"P1-2 regression: el aggregator debe pasar apply_yield_multiplier=False "
        f"a `_parse_quantity` para ambos lados (plan + consumed). Got: {occurrences}"
    )


def test_aggregator_treats_plan_and_inventory_symmetrically():
    """Si plan e inventario tienen el MISMO string, el delta debe ser cero
    independiente de si menciona 'cocido'."""
    plan = ["1 lb pollo cocido"]
    consumed = ["1 lb pollo cocido"]
    result = aggregate_and_deduct_shopping_list(plan, consumed, structured=True)
    pollo_items = [r for r in result if isinstance(r, dict) and "pollo" in r.get("name", "").lower()]
    assert pollo_items == [], (
        f"P1-2: mismo string en plan e inventario → delta cero. Got: {pollo_items}"
    )


def test_no_overbuy_when_plan_says_cocido_inventory_says_raw():
    """Caso central del audit P1-2:
        plan: '1 lb pollo cocido'
        inventory: '1 lb pollo'
    Antes del fix: plan=1.35 lb (yield), inventory=1.0 lb (sin yield) →
        delta=0.35 lb a comprar (over-buying si LLM realmente describe
        peso comprable).
    Tras el fix: ambos en peso literal 1.0 lb → delta=0 (no compras).
    """
    plan = ["1 lb pollo cocido"]
    inventory_as_consumed = ["1 lb pollo"]
    result = aggregate_and_deduct_shopping_list(
        plan, inventory_as_consumed, structured=True,
    )
    pollo_items = [r for r in result if isinstance(r, dict) and "pollo" in r.get("name", "").lower()]
    assert pollo_items == [], (
        f"P1-2: 1 lb cocido en plan vs 1 lb en inventario → delta literal 0, "
        f"NO debe agregar pollo a la lista. Got: {pollo_items}"
    )


def test_aggregator_still_subtracts_partial_inventory():
    """Sanity: si plan=1 lb y inventory=0.5 lb, el delta es 0.5 lb a comprar
    (caso normal, no relacionado con yield)."""
    plan = ["1 lb pollo"]
    consumed = ["0.5 lb pollo"]
    result = aggregate_and_deduct_shopping_list(plan, consumed, structured=True)
    pollo_items = [r for r in result if isinstance(r, dict) and "pollo" in r.get("name", "").lower()]
    assert pollo_items, "delta positivo debe agregar item"
    mq = pollo_items[0].get("market_qty_numeric") or pollo_items[0].get("market_qty")
    assert isinstance(mq, (int, float))
    assert mq > 0


# ---------------------------------------------------------------------------
# 4. Defensa: yield sigue funcionando para callers que NO pasan el flag.
# ---------------------------------------------------------------------------
def test_legacy_callers_still_get_yield_via_default():
    """Callers que NO pasan el flag (legacy: tools.py, cron_tasks.py, etc.)
    siguen recibiendo yield aplicado — el contrato no cambia para ellos."""
    qty, _, _ = _parse_quantity("2 lb arroz cocido")
    # Yield 0.35 para granos cocidos: 2 lb cocido → 0.7 lb crudo.
    assert abs(qty - 0.7) < 0.01, f"legacy default debe aplicar yield, got {qty}"


def test_documentation_p1_2_present_in_aggregator():
    """Comentario `[P1-2]` documenta el rationale para futuros maintainers."""
    src = inspect.getsource(aggregate_and_deduct_shopping_list)
    assert "[P1-2]" in src, "P1-2: falta comentario de auditoría"


def test_documentation_p1_2_present_in_parse_quantity():
    """Comentario `[P1-2]` también en `_parse_quantity` documentando el flag."""
    src = inspect.getsource(_parse_quantity)
    assert "[P1-2]" in src or "P1-2" in src, "P1-2: falta documentación del flag"
