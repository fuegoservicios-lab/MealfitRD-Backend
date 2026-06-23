"""[P3-WEIGHT-DEFAULT-NO-UNITIZE · 2026-06-22] Un item DECLARADO por peso
(default_unit ∈ lb/kg/g) con densidad NULL en la DB ya NO recibe una densidad-unidad
fantasma desde constants.UNIT_WEIGHTS.

Bug: la sandía (owner la dio por libra, "L=30/lb") tenía density NULL pero caía a
UNIT_WEIGHTS["sandia"]=3000g → apply_smart_market_units la unitizaba a "1 sandía entera
(~6.6 lbs) RD$198" para CUALQUIER necesidad (incluso 200g). Ahora respeta default_unit='lb'
→ costea por libra. Items declarados por unidad (melón, default_unit='unidad' + densidad)
NO se ven afectados.
"""
from __future__ import annotations

import pytest

from shopping_calculator import apply_smart_market_units, _cost_from_market

# Sandía: por peso, sin densidad DB, nombre matchea UNIT_WEIGHTS['sandia'] (el caso del bug).
SANDIA = {"name": "Sandía", "default_unit": "lb", "density_g_per_unit": None,
          "price_per_lb": 30, "price_per_unit": 198.24, "category": "Frutas"}
# Melón: por unidad con densidad → debe seguir unitizando (control, NO afectado).
MELON = {"name": "Melón", "default_unit": "unidad", "density_g_per_unit": 1500,
         "price_per_lb": 21.79, "price_per_unit": 72, "category": "Frutas"}


@pytest.mark.parametrize("g,exp_cost", [(454, 30), (900, 60), (1800, 120)])
def test_weight_default_item_costs_by_pound(g, exp_cost):
    mo = apply_smart_market_units("Sandía", g / 453.592, "lb", 0.0, SANDIA)
    # NO debe unitizarse a "Ud." (sandía entera).
    assert "Ud." not in (mo.get("display_qty") or ""), f"sandía no debe ser unidad: {mo.get('display_qty')}"
    assert "lb" in (mo.get("display_qty") or "").lower()
    cost = _cost_from_market(mo, SANDIA, 30, 198.24)
    assert cost == pytest.approx(exp_cost, abs=1.0)


def test_unidad_default_item_still_unitizes():
    """Control: melón (default_unit='unidad' + densidad) sigue por unidad — el guard NO lo toca."""
    mo = apply_smart_market_units("Melón", 1500 / 453.592, "lb", 0.0, MELON)
    assert "Ud." in (mo.get("display_qty") or "")
    assert _cost_from_market(mo, MELON, 21.79, 72) == pytest.approx(72, abs=1.0)


def test_marker_present():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.apply_smart_market_units)
    assert "P3-WEIGHT-DEFAULT-NO-UNITIZE" in src
    assert "_du_weight" in src
