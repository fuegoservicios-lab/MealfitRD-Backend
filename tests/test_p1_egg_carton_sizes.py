"""[P1-EGG-CARTON-SIZES · 2026-06-22] Cartón de huevos 20 vs 30 + unidad por-envase real.

Dos mejoras de nivel producción sobre la lista de compras (datos verificados in-store por owner):

1) HUEVOS — el mercado DR tiene DOS cartones reales: 20 uds (RD$200, mejor para planes de 7 días) y
   30 uds (RD$295, mejor valor/huevo para 15-30 días). Antes se forzaba cartón de 30 siempre y se
   costeaba derivando price_per_unit/30 (sobrecobro en planes chicos: 14 huevos → RD$295 en vez de
   RD$200). `_choose_egg_carton` elige el cartón cost-óptimo; el egg branch de `_cost_from_market`
   costea con el precio REAL del cartón elegido (no escala lineal: 20→10/huevo, 30→9.83/huevo).

2) UNIDAD POR-ENVASE — un mismo ítem puede venderse en formas distintas (habichuelas: lata 15oz Y
   bolsa 800g seco; orégano: sobre Y pote). Antes el display usaba SIEMPRE el market_container
   genérico → "1 lata (800 g seco)" (la palabra 'lata' falsa para la bolsa). Ahora cada market_package
   lleva su `unit` y el display la usa → "1 paquete (800 g seco)".

Tests puros (sin DB): master_item inline = espejo de master_ingredients en Neon.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from shopping_calculator import (
    _choose_egg_carton,
    _select_market_package,
    _cost_from_market,
    apply_smart_market_units,
    get_plural_unit,
)

_BACKEND = Path(__file__).resolve().parent.parent

EGG_PKGS = [
    {"units": 20, "price": 200, "label": "cartón 20 uds"},
    {"units": 30, "price": 295, "label": "cartón 30 uds"},
]
HUEVO = {"name": "Huevo", "container_weight_g": 900, "density_g_per_unit": 50, "market_packages": EGG_PKGS}

BEANS = {
    "name": "Habichuelas blancas", "default_unit": "lb", "market_container": "lata",
    "container_weight_g": 425, "available_sizes_g": [425, 2000],
    "market_packages": [
        {"grams": 425, "price": 50, "label": "15 oz", "unit": "lata"},
        {"grams": 2000, "price": 115, "label": "800 g seco", "unit": "paquete"},
    ],
    "price_per_lb": 0, "price_per_unit": 50, "density_g_per_unit": 0, "shelf_life_days": 14,
}


def test_marker_present_in_source():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "P1-EGG-CARTON-SIZES" in src
    assert "def _choose_egg_carton" in src


# ─────────────────────────── huevos: selección de cartón ───────────────────────────

@pytest.mark.parametrize("eggs,exp_units,exp_count,exp_cost", [
    (6, 20, 1, 200),    # plan chico → cartón 20 (ahorra RD$95 vs forzar 30)
    (14, 20, 1, 200),
    (20, 20, 1, 200),
    (22, 30, 1, 295),   # >20 → cartón 30 más barato que 2×20
    (30, 30, 1, 295),
    (35, 20, 2, 400),   # 2×20 (400) < 2×30 (590)
    (56, 30, 2, 590),   # bulk: 2×30 (590) < 3×20 (600)
    (60, 30, 2, 590),
])
def test_choose_egg_carton_cost_optimal(eggs, exp_units, exp_count, exp_cost):
    sel = _choose_egg_carton(eggs, EGG_PKGS)
    assert sel is not None
    assert sel["units"] == exp_units
    assert sel["count"] == exp_count
    assert sel["count"] * sel["price"] == exp_cost


def test_choose_egg_carton_none_without_data():
    assert _choose_egg_carton(14, None) is None
    assert _choose_egg_carton(14, []) is None
    assert _choose_egg_carton(14, [{"label": "x"}]) is None  # sin units/price


# ─────────────────────────── huevos: costeo con precio real ───────────────────────────

@pytest.mark.parametrize("carton_n,count,exp_cost", [
    (20, 1, 200),   # precio REAL del cartón de 20 (NO price_per_unit/30 × 20 = 196.67)
    (30, 1, 295),
    (30, 2, 590),
])
def test_egg_cost_uses_real_carton_price(carton_n, count, exp_cost):
    mo = {"market_qty": count, "market_qty_numeric": count, "market_unit": f"cartón ({carton_n} uds.)"}
    cost = _cost_from_market(mo, HUEVO, 0, 295)
    assert cost == pytest.approx(exp_cost)


def test_egg_cost_fallback_legacy_without_market_packages():
    """Sin market_packages → fallback al cálculo legacy (price_per_unit por cartón de 30)."""
    huevo_legacy = {"name": "Huevo", "container_weight_g": 900, "density_g_per_unit": 50}
    mo = {"market_qty": 1, "market_qty_numeric": 1, "market_unit": "cartón (30 uds.)"}
    cost = _cost_from_market(mo, huevo_legacy, 0, 295)
    assert cost == pytest.approx(295)  # 1 × 30 × (295/30)


# ─────────────────────────── unidad por-envase (habichuelas/especias) ───────────────────────────

def test_bean_dry_bag_shows_paquete_not_lata():
    """La bolsa seca (necesidad grande) debe mostrarse como 'paquete', NO 'lata'."""
    mo = apply_smart_market_units("Habichuelas blancas", 2048 / 453.592, "lb", 0.0, BEANS)
    assert mo["market_unit"] == "paquete"
    assert "paquete" in mo["display_qty"]
    assert "lata" not in mo["display_qty"]
    cost = _cost_from_market(mo, BEANS, 0, 50)
    assert cost == pytest.approx(115)


def test_bean_can_shows_lata():
    """Necesidad chica → lata (forma correcta del envase chico)."""
    mo = apply_smart_market_units("Habichuelas blancas", 300 / 453.592, "lb", 0.0, BEANS)
    assert mo["market_unit"] == "lata"
    assert "lata" in mo["display_qty"]
    cost = _cost_from_market(mo, BEANS, 0, 50)
    assert cost == pytest.approx(50)


def test_select_market_package_returns_unit():
    sel = _select_market_package(2048, BEANS["market_packages"])
    assert sel["unit"] == "paquete"
    sel2 = _select_market_package(300, BEANS["market_packages"])
    assert sel2["unit"] == "lata"


def test_plural_unit_with_parenthetical_suffix():
    """'cartón (30 uds.)' x2 → 'cartones (30 uds.)' (pluraliza la cabeza, re-anexa sufijo)."""
    assert get_plural_unit(2, "cartón (30 uds.)") == "cartones (30 uds.)"
    assert get_plural_unit(2, "cartón (20 uds.)") == "cartones (20 uds.)"
    assert get_plural_unit(1, "cartón (30 uds.)") == "cartón (30 uds.)"  # singular intacto
    # Unidades simples sin paréntesis: comportamiento idéntico al previo (sin regresión).
    assert get_plural_unit(3, "paquete") == "paquetes"
    assert get_plural_unit(2, "Pote") == "Potes"
    assert get_plural_unit(2, "lb") == "lbs"
    assert get_plural_unit(2, "lata") == "latas"


def test_egg_display_pluralized_for_multiple_cartons():
    huevo = {"name": "Huevo", "container_weight_g": 900, "density_g_per_unit": 50,
             "market_container": "cartón", "default_unit": "cartón", "market_packages": EGG_PKGS}
    mo = apply_smart_market_units("Huevo", 0.0, "cartón (30 uds.)", 2, huevo)
    assert mo["display_qty"] == "2 cartones (30 uds.)"


def test_unit_falls_back_to_container_when_absent():
    """market_packages sin `unit` (ej. arroz, batch previo) → usa market_container (sin regresión)."""
    arroz = {
        "name": "Arroz blanco", "default_unit": "lb", "market_container": "paquete",
        "container_weight_g": 907, "available_sizes_g": [907, 2268, 4536],
        "market_packages": [
            {"grams": 907, "price": 165, "label": "2 lb"},
            {"grams": 2268, "price": 235, "label": "5 lb"},
        ],
        "price_per_lb": 55, "price_per_unit": 55, "density_g_per_unit": 0, "shelf_life_days": 14,
    }
    mo = apply_smart_market_units("Arroz blanco", 1050 / 453.592, "lb", 0.0, arroz)
    assert mo["market_unit"] == "paquete"  # fallback a db_container, sin campo unit
