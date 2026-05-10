"""[P1-7] Tests para garantizar que `aggregate_and_deduct_shopping_list` y
`get_shopping_list_delta` defienden contra `multiplier` patológico
(NaN/Infinity/cero/negativo/no-numérico).

Bug original (audit P1-7):
  El multiplier escalaba `plan_ingredients` directamente vía
  `aggregated[name][unit] += float(qty) * float(multiplier)`. Sin guards:
    - `householdSize=0` por perfil corrupto → caller pasa `1.0 * 0 = 0` →
      todo plan_ingredients se anula → lista vacía falsa que el frontend
      muestra como "no hay nada que comprar".
    - `num_days=0` en plan vacío persistido a medias → div-zero al calcular
      `base_duration_scale = 7/num_days` (mitigado por max(1,..) pero
      callers terceros pueden pasar effective_multiplier directo).
    - Float overflow → `inf` → qty=`inf` en aggregated → cálculos posteriores
      revientan o producen strings "inf"/"nan" en el shopping list.

Fix:
  1. `aggregate_and_deduct_shopping_list` clampa `multiplier` a `[0.01, 50.0]`
     ante NaN/Inf/<=0/>50, con log warning específico.
  2. `get_shopping_list_delta` normaliza NaN/Inf desde el caller antes de
     calcular `effective_multiplier`, log warning específico.
  3. Lista vacía falsa eliminada: el sistema preserva una lista renderizable
     (proporcional al multiplier clampado) y SRE detecta el caso edge vía
     log warning para investigar.

Cobertura:
  - test_aggregator_clamps_zero_multiplier_to_safe_default
  - test_aggregator_clamps_negative_multiplier
  - test_aggregator_clamps_nan_multiplier
  - test_aggregator_clamps_infinity_multiplier
  - test_aggregator_clamps_excessive_multiplier
  - test_aggregator_handles_non_numeric_multiplier
  - test_delta_handles_nan_multiplier_from_caller
  - test_delta_handles_inf_multiplier_from_caller
  - test_normal_multiplier_is_not_modified
"""
import logging
import math

import pytest

import shopping_calculator
from shopping_calculator import aggregate_and_deduct_shopping_list, get_shopping_list_delta


# ---------------------------------------------------------------------------
# 1. Aggregator: NaN/Inf/zero/negative/excessive clamping.
# ---------------------------------------------------------------------------
def test_aggregator_clamps_zero_multiplier_to_safe_default(caplog):
    """multiplier=0 (perfil corrupto) NO debe anular el plan completo."""
    plan = ["500 g de pollo"]
    with caplog.at_level(logging.WARNING):
        result = aggregate_and_deduct_shopping_list(plan, [], structured=True, multiplier=0)
    # Debe haber un item de pollo (no lista vacía falsa).
    pollo_items = [r for r in result if isinstance(r, dict) and "pollo" in r.get("name", "").lower()]
    assert pollo_items, "P1-7: multiplier=0 NO debe anular el plan"
    # Y debe haber log de warning con marker P1-7.
    assert any("[P1-7" in r.message for r in caplog.records), \
        "P1-7: log warning debe disparar con marker [P1-7"


def test_aggregator_clamps_negative_multiplier(caplog):
    """multiplier negativo (jamás esperado) → clamp a default."""
    plan = ["500 g de pollo"]
    with caplog.at_level(logging.WARNING):
        result = aggregate_and_deduct_shopping_list(plan, [], structured=True, multiplier=-2.0)
    # No debe lanzar; debe producir lista renderizable.
    assert isinstance(result, list)
    assert any("[P1-7" in r.message for r in caplog.records)


def test_aggregator_clamps_nan_multiplier(caplog):
    """multiplier=NaN (overflow upstream) → clamp."""
    plan = ["500 g de pollo"]
    with caplog.at_level(logging.WARNING):
        result = aggregate_and_deduct_shopping_list(plan, [], structured=True, multiplier=float('nan'))
    assert isinstance(result, list)
    # Y NO debe haber NaN en los resultados (qty/cost serializables).
    for item in result:
        if isinstance(item, dict):
            mq = item.get("market_qty_numeric") or 0
            assert not math.isnan(mq), f"P1-7: market_qty_numeric NaN: {item}"


def test_aggregator_clamps_infinity_multiplier(caplog):
    """multiplier=Infinity → clamp a max safe."""
    plan = ["500 g de pollo"]
    with caplog.at_level(logging.WARNING):
        result = aggregate_and_deduct_shopping_list(plan, [], structured=True, multiplier=float('inf'))
    assert isinstance(result, list)
    for item in result:
        if isinstance(item, dict):
            mq = item.get("market_qty_numeric") or 0
            assert not math.isinf(mq), f"P1-7: market_qty_numeric Inf: {item}"


def test_aggregator_clamps_excessive_multiplier(caplog):
    """multiplier > 50 (cap del peor caso 12p × 4 ciclos) → clamp a 50."""
    plan = ["500 g de pollo"]
    with caplog.at_level(logging.WARNING):
        result = aggregate_and_deduct_shopping_list(plan, [], structured=True, multiplier=1000.0)
    assert isinstance(result, list)
    # Debe haber log de warning con marker excede cap.
    assert any("excede cap" in r.message or "[P1-7" in r.message for r in caplog.records), \
        "P1-7: clamp a 50 debe loggear warning"


def test_aggregator_handles_non_numeric_multiplier():
    """multiplier no-numérico (string, None, dict) → defaultea a 1.0 sin lanzar."""
    plan = ["500 g de pollo"]
    for bad in ["abc", None, {}, []]:
        # No debe lanzar.
        result = aggregate_and_deduct_shopping_list(plan, [], structured=True, multiplier=bad)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 2. get_shopping_list_delta: defensa upstream antes de effective_multiplier.
# ---------------------------------------------------------------------------
def test_delta_handles_nan_multiplier_from_caller(caplog):
    """Si el caller pasa NaN, el delta no debe propagar el NaN al aggregator."""
    plan = {"days": [{"meals": [{"ingredients": ["500 g de pollo"]}]}]}
    with caplog.at_level(logging.WARNING):
        result = get_shopping_list_delta(
            "guest", plan, is_new_plan=True, structured=True, multiplier=float('nan'),
        )
    assert isinstance(result, list)
    # Log con marker DELTA-MULT debe aparecer.
    assert any("[P1-7/DELTA-MULT" in r.message for r in caplog.records), \
        "P1-7: get_shopping_list_delta debe detectar y loguear NaN upstream"


def test_delta_handles_inf_multiplier_from_caller(caplog):
    """Si el caller pasa Inf, mismo defaulteo seguro."""
    plan = {"days": [{"meals": [{"ingredients": ["500 g de pollo"]}]}]}
    with caplog.at_level(logging.WARNING):
        result = get_shopping_list_delta(
            "guest", plan, is_new_plan=True, structured=True, multiplier=float('inf'),
        )
    assert isinstance(result, list)
    # Verifica que ningún item resultante tiene qty Inf.
    for item in result:
        if isinstance(item, dict):
            mq = item.get("market_qty_numeric") or 0
            assert not math.isinf(mq) and not math.isnan(mq), \
                f"P1-7: delta no debe propagar Inf/NaN: {item}"


# ---------------------------------------------------------------------------
# 3. Sanity: multiplier normal NO se modifica.
# ---------------------------------------------------------------------------
def test_normal_multiplier_is_not_modified(caplog):
    """multiplier=2.0 (válido) no debe disparar warnings ni modificarse."""
    plan = ["500 g de pollo"]
    with caplog.at_level(logging.WARNING):
        result = aggregate_and_deduct_shopping_list(plan, [], structured=True, multiplier=2.0)
    assert isinstance(result, list)
    # NO debe haber warnings con marker P1-7.
    p17_warnings = [r for r in caplog.records if "[P1-7" in r.message]
    assert not p17_warnings, \
        f"P1-7: multiplier=2.0 válido NO debe disparar warnings, got {[r.message for r in p17_warnings]}"


def test_documentation_p1_7_present():
    """Comentario `[P1-7]` documenta el rationale en aggregator + delta."""
    src = open(shopping_calculator.__file__, encoding="utf-8").read()
    assert "[P1-7" in src
    # Markers específicos para los dos sitios.
    assert "P1-7/MULTIPLIER" in src or "[P1-7]" in src
    assert "P1-7/DELTA-MULT" in src
