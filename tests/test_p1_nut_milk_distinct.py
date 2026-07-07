"""[P1-NUT-MILK-DISTINCT · 2026-07-07] "leche de almendras" → "Almendras fileteadas"
× 19 paquetes = RD$5,491 (reventaba el presupuesto del plan de 30 días).

Forense (plan 5610de53): receta "1¾ tazas de leche de almendras sin azúcar (420ml)".
El master tiene AMBOS ["Leche de almendras" (Lácteos, cartón 946ml, RD$260)] y
["Almendras fileteadas" (Despensa, 170g, RD$289)]. Dos canonicalizers fallaban:
  - canonicalize_frutos_secos("leche de almendras") → "Almendras" (sin guard de "leche de X";
    solo tenía guard de mantequilla/crema/pasta) → consolidada a "Almendras fileteadas".
  - canonicalize_lacteo → "Leche" (leche de vaca): el guard `almendra\b` era SINGULAR y la
    receta dice "almendraS" (plural) → no matcheaba → caía a leche normal.

Fix: ambos devuelven None para "leche de <vegetal>" (bebida distinta) → el master resuelve
"Leche de almendras" exacto (cartón). Las almendras/mantequilla siguen correctas.
tooltip-anchor: P1-NUT-MILK-DISTINCT
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc


# --- Canonicalizers: leche vegetal → None en AMBOS ---
@pytest.mark.parametrize("name", [
    "leche de almendras", "leche de almendras sin azúcar", "leche de almendra",
    "leche de coco", "leche de maní", "leche de soya", "leche de avena",
])
def test_plant_milk_not_frutos_secos(name):
    assert sc.canonicalize_frutos_secos(name) is None, f"{name!r} no debe ser fruto seco"


@pytest.mark.parametrize("name", [
    "leche de almendras", "leche de almendras sin azúcar", "leche de almendra",
    "leche de coco", "leche de maní", "leche de soya", "leche de avena",
])
def test_plant_milk_not_dairy_leche(name):
    assert sc.canonicalize_lacteo(name) is None, f"{name!r} no debe ser 'Leche' (vaca)"


# --- Regresión: frutos secos REALES y mantequilla intactos ---
@pytest.mark.parametrize("name,expected", [
    ("almendras fileteadas", "Almendras"),
    ("almendras tostadas", "Almendras"),
    ("almendra laminada", "Almendras"),
    ("maní tostado", "Maní"),
])
def test_real_nuts_still_canonicalize(name, expected):
    assert sc.canonicalize_frutos_secos(name) == expected


def test_nut_butter_still_none():
    assert sc.canonicalize_frutos_secos("mantequilla de almendras") is None
    assert sc.canonicalize_frutos_secos("mantequilla de maní") is None


# --- Regresión: leche de vaca real sigue → "Leche" ---
@pytest.mark.parametrize("name", ["leche descremada", "leche entera", "leche deslactosada"])
def test_cow_milk_still_leche(name):
    assert sc.canonicalize_lacteo(name) == "Leche"


# --- E2E: la leche de almendras se compra como CARTÓN, no 19 paquetes de almendras ---
def test_almond_milk_priced_as_carton(monkeypatch):
    stub = [
        {"name": "Leche de almendras", "category": "Lácteos", "default_unit": "litro",
         "market_container": "carton", "container_weight_g": 946.0, "density_g_per_unit": None,
         "price_per_lb": 124.67, "price_per_unit": 260.0, "aliases": []},
        {"name": "Almendras fileteadas", "category": "Despensa", "default_unit": "paquete",
         "market_container": "paquete", "container_weight_g": 170.0, "density_g_per_unit": None,
         "price_per_lb": 771.11, "price_per_unit": 289.0, "aliases": []},
    ]
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(stub))
    sc.invalidate_master_cache()
    try:
        r = sc.aggregate_and_deduct_shopping_list(
            ["1¾ tazas de leche de almendras sin azúcar (420ml)"], [], structured=True, multiplier=1.0,
        )
        alm = [x for x in r if "almendras fileteadas" in str(x.get("name", "")).lower()]
        assert not alm, f"la leche de almendras NO debe comprarse como almendras fileteadas: {alm}"
        leche = [x for x in r if "leche de almendras" in str(x.get("name", "")).lower()]
        # Debe resolver a la leche (cartón) — y si aparece, su costo es de cartón, no ~RD$5k.
        for it in leche:
            assert float(it.get("estimated_cost_rd") or 0) < 1000.0, (
                f"leche de almendras debe costar como cartón, no explotar: {it}"
            )
    finally:
        sc.invalidate_master_cache()
