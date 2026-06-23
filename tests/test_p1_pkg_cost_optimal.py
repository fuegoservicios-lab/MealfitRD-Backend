"""[P1-PKG-COST-OPTIMAL + P1-AJO-4PACK · 2026-06-22] Selección de envases por COSTO mínimo
+ ajo en paquetes de 4 cabezas.

1) COST-OPTIMAL: `_select_market_package` elige el envase de MENOR COSTO total (con el mismo
   floor anti-desperdicio que _find_best_sku), no por desperdicio+conteo. Antes el yogurt
   griego compraba 2 four-packs (RD$730) para ~900g cuando 6 potes sueltos (RD$600, exacto)
   eran más baratos — el penalty de conteo de _find_best_sku ganaba. Verificado que arroz y
   habichuelas mantienen su selección (el floor anti-desperdicio evita over-comprar bulk).

2) AJO-4PACK: el ajo se vende en paquetes de 4 cabezas (RD$60). La consolidación redondea
   cabezas → 'paquete (4 uds.)' (ceil/4); el cost branch lee el precio real del 4-pack desde
   Ajo.market_packages [{units:4, price:60}]. 1-2 cabezas → 1 paquete RD$60.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from shopping_calculator import _select_market_package, _cost_from_market

_BACKEND = Path(__file__).resolve().parent.parent

YOG = [{"grams": 150, "price": 100, "label": "150 g", "unit": "pote"},
       {"grams": 600, "price": 365, "label": "4x150", "unit": "paquete"}]
ARROZ = [{"grams": 907, "price": 165, "label": "2 lb"},
         {"grams": 2268, "price": 235, "label": "5 lb"},
         {"grams": 4536, "price": 327, "label": "10 lb"}]
BEANS = [{"grams": 425, "price": 50, "label": "lata", "unit": "lata"},
         {"grams": 2000, "price": 115, "label": "seco", "unit": "paquete"}]


def _cost(sel):
    return sel["count"] * sel["price"]


def test_markers_present():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "P1-PKG-COST-OPTIMAL" in src
    assert "P1-AJO-4PACK" in src
    # La consolidación de ajo produce paquetes de 4.
    assert "paquete (4 uds.)" in src


# ─────────────────── COST-OPTIMAL: yogurt (el fix) ───────────────────

@pytest.mark.parametrize("g,exp_grams,exp_cost", [
    (150, 150, 100),    # 1 pote
    (300, 150, 200),    # 2 potes
    (600, 600, 365),    # 1 four-pack (más barato que 4 potes=400)
    (900, 150, 600),    # 6 potes RD$600 (NO 2 four-packs RD$730) ← EL FIX
    (1200, 600, 730),   # 2 four-packs
])
def test_yogurt_cost_optimal(g, exp_grams, exp_cost):
    sel = _select_market_package(g, YOG)
    assert sel["grams"] == exp_grams
    assert _cost(sel) == exp_cost


def test_yogurt_900_not_overshoot():
    """Caso reportado: 900g NO debe comprar 2 four-packs (RD$730) — 6 potes (RD$600) es más barato."""
    sel = _select_market_package(900, YOG)
    assert _cost(sel) == 600 and sel["unit"] == "pote"


# ─────────────────── COST-OPTIMAL: arroz/beans sin regresión ───────────────────

@pytest.mark.parametrize("g,exp_grams,exp_cost", [
    (1050, 907, 165),    # 7d → 2 lb
    (2250, 2268, 235),   # 15d → 5 lb
    (4500, 4536, 327),   # 30d → 10 lb
])
def test_arroz_unchanged(g, exp_grams, exp_cost):
    sel = _select_market_package(g, ARROZ)
    assert sel["grams"] == exp_grams
    assert _cost(sel) == exp_cost


@pytest.mark.parametrize("g,exp_grams,exp_cost", [
    (300, 425, 50),      # lata
    (2048, 2000, 115),   # bolsa seca
])
def test_beans_unchanged(g, exp_grams, exp_cost):
    sel = _select_market_package(g, BEANS)
    assert sel["grams"] == exp_grams
    assert _cost(sel) == exp_cost


# ─────────────────── AJO 4-PACK costeo ───────────────────

AJO = {"name": "Ajo", "market_packages": [{"units": 4, "price": 60, "label": "4 cabezas"}]}
HUEVO = {"name": "Huevo", "market_packages": [{"units": 20, "price": 200}, {"units": 30, "price": 295}]}


@pytest.mark.parametrize("count,exp", [(1, 60), (2, 120)])
def test_ajo_paquete_cost(count, exp):
    mo = {"market_qty": count, "market_qty_numeric": count, "market_unit": "paquete (4 uds.)"}
    assert _cost_from_market(mo, AJO, 119, 15) == pytest.approx(exp)


@pytest.mark.parametrize("n,count,exp", [(20, 1, 200), (30, 1, 295), (30, 2, 590)])
def test_egg_carton_still_works(n, count, exp):
    """El branch generalizado (units-based) no rompe huevos."""
    mo = {"market_qty": count, "market_qty_numeric": count, "market_unit": f"cartón ({n} uds.)"}
    assert _cost_from_market(mo, HUEVO, 0, 295) == pytest.approx(exp)


def test_units_branch_falls_through_without_market_packages():
    """Un '(N uds.)' sin market_packages.units y sin 'cart' → no cobra por este branch (fall-through)."""
    mo = {"market_qty": 1, "market_qty_numeric": 1, "market_unit": "paquete (6 uds.)"}
    # Sin market_packages units → el branch no retorna; cae a otros (aquí 0 por unit no costeable).
    cost = _cost_from_market(mo, {"name": "X"}, 0, 0)
    assert cost == 0.0
