"""[P1-PKG-DURATION-PRICING · 2026-06-22] Precio por TAMAÑO de envase real (no precio plano).

La lista de compras ya elegía bien el envase por duración (`_find_best_sku` con
`available_sizes_g`: 2 lb para 7 días, 10 lb para 30 días), pero el COSTO se calculaba con un
único `price_per_lb`/`price_per_unit` → ignoraba el descuento por volumen. Ej. arroz blanco
plan 30 días = 10 lb × 55 = RD$550, cuando el paquete real de 10 lb cuesta RD$327 (datos
verificados in-store por el owner).

Fix: columna `market_packages` = [{grams, price, label}] + `_select_market_package` que
arrastra el precio del tamaño elegido a `market_obj['market_pkg_price_rd']`, consumido por
`_cost_from_market` (count × precio_del_paquete). NULL → fallback al precio plano (sin cambio).

Tests puros (sin DB): construyen el `master_item` inline (espejo de los datos en Neon).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from shopping_calculator import (
    apply_smart_market_units,
    _cost_from_market,
    _select_market_package,
)

_BACKEND = Path(__file__).resolve().parent.parent


# ── master_item espejo de master_ingredients (Neon) para arroz blanco ──
ARROZ = {
    "name": "Arroz blanco", "default_unit": "lb", "market_container": "paquete",
    "container_weight_g": 907, "available_sizes_g": [907, 2268, 4536],
    "market_packages": [
        {"grams": 907, "price": 165, "label": "2 lb"},
        {"grams": 2268, "price": 235, "label": "5 lb"},
        {"grams": 4536, "price": 327, "label": "10 lb"},
    ],
    "price_per_lb": 55, "price_per_unit": 55, "density_g_per_unit": 0, "shelf_life_days": 14,
}

SAL = {
    "name": "Sal", "default_unit": "paquete", "market_container": "paquete",
    "container_weight_g": 454, "available_sizes_g": [454, 907],
    "market_packages": [
        {"grams": 454, "price": 17, "label": "1 lb"},
        {"grams": 907, "price": 40, "label": "2 lb"},
    ],
    "price_per_lb": 0, "price_per_unit": 17, "density_g_per_unit": 0, "shelf_life_days": 14,
}


def _cost_for_grams(master, g_total):
    wlbs = g_total / 453.592
    mo = apply_smart_market_units(master["name"], wlbs, "lb", 0.0, master)
    return mo, _cost_from_market(mo, master, master.get("price_per_lb") or 0, master.get("price_per_unit") or 0)


# ───────────────────────── tooltip-anchor / marker ─────────────────────────

def test_marker_present_in_source():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "P1-PKG-DURATION-PRICING" in src
    assert "def _select_market_package" in src
    assert "market_pkg_price_rd" in src


# ───────────────────────── selección por duración ─────────────────────────

@pytest.mark.parametrize("g_total,exp_grams,exp_price", [
    (1050, 907, 165),    # ~7 días → 2 lb
    (2250, 2268, 235),   # ~15 días → 5 lb
    (4500, 4536, 327),   # ~30 días → 10 lb
])
def test_select_market_package_tier_by_grams(g_total, exp_grams, exp_price):
    sel = _select_market_package(g_total, ARROZ["market_packages"])
    assert sel is not None
    assert sel["count"] == 1
    assert sel["grams"] == exp_grams
    assert sel["price"] == exp_price


def test_select_market_package_none_without_data():
    assert _select_market_package(1000, None) is None
    assert _select_market_package(1000, []) is None
    assert _select_market_package(1000, [{"label": "x"}]) is None  # sin grams/price usables


# ───────────────── costo end-to-end (apply_smart_market_units + _cost_from_market) ─────────────────

@pytest.mark.parametrize("g_total,exp_label,exp_cost", [
    (1050, "2 lb", 165),
    (2250, "5 lb", 235),
    (4500, "10 lb", 327),
])
def test_arroz_cost_by_duration(g_total, exp_label, exp_cost):
    mo, cost = _cost_for_grams(ARROZ, g_total)
    assert mo.get("market_pkg_price_rd") == exp_cost  # 1 paquete → price == cost
    assert exp_label in (mo.get("display_qty") or "")
    assert cost == pytest.approx(exp_cost)


def test_arroz_30d_no_longer_overcharges_flat():
    """Antes: 30 días = ~10 lb × 55 = RD$550 (precio plano). Ahora: 1 paquete 10 lb = RD$327."""
    _, cost = _cost_for_grams(ARROZ, 4500)
    flat = (4500 / 453.592) * 55  # ~546
    assert cost < flat * 0.7      # el descuento por volumen es material
    assert cost == pytest.approx(327)


def test_sal_tiers():
    _, c7 = _cost_for_grams(SAL, 300)
    _, c30 = _cost_for_grams(SAL, 800)
    assert c7 == pytest.approx(17)   # 1 lb
    assert c30 == pytest.approx(40)  # 2 lb


# ───────────────────────── backward-compat (sin market_packages) ─────────────────────────

def test_fallback_without_market_packages_unchanged():
    generico = {
        "name": "Generico", "default_unit": "lb", "market_container": "paquete",
        "container_weight_g": 453, "available_sizes_g": None, "market_packages": None,
        "price_per_lb": 50, "price_per_unit": 0, "density_g_per_unit": 0, "shelf_life_days": 14,
    }
    mo, cost = _cost_for_grams(generico, 3 * 453.592)  # 3 lb
    assert mo.get("market_pkg_price_rd") is None       # no se inyecta el campo → path legacy
    assert cost == pytest.approx(150, rel=0.02)        # ~3 lb × 50 (precio plano legacy, sin cambio)
