"""[P1-BRAND-GENERIC-LABEL · 2026-07-06] La lista SIEMPRE dice la marca que usa.

Feedback del owner viendo el PDF sin marcas: los productos MÁS BARATOS del
catálogo suelen ser los brand=NULL ("Genérico" en el picker) — para esos, el
label del envase era solo-tamaño ("2 lb"), indistinguible del costeo sin marcas.
Ahora `_pkg_from_product_row` etiqueta brand NULL como "Genérico" (mismo
fallback que el picker) → display "1 paquete (2 lb · Genérico)" en lista y PDF.
"""
from pathlib import Path

import shopping_calculator as sc

BACKEND = Path(__file__).resolve().parents[1]
SRC = (BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


def test_brandless_product_labeled_generico():
    pkg = sc._pkg_from_product_row({
        "presentation": "Paquete 2L", "brand": None, "price_rd": 165.0, "size_grams": None,
    })
    assert pkg is not None
    assert pkg["label"] == "2L · Genérico", f"brand NULL debe decir Genérico: {pkg['label']}"


def test_real_brand_unaffected():
    pkg = sc._pkg_from_product_row({
        "presentation": "Funda Selecto 1 Lb", "brand": "Wala", "price_rd": 42.0, "size_grams": None,
    })
    assert pkg is not None and pkg["label"].endswith("· Wala")


def test_generico_flows_to_display():
    defaults = {"arroz blanco": [
        {"grams": 907.184, "price": 165.0, "label": "2L · Genérico", "unit": "paquete"},
    ]}
    result = sc.aggregate_and_deduct_shopping_list(
        ["800g de arroz blanco"], [], structured=True, brand_defaults=defaults,
    )
    item = next((r for r in result if isinstance(r, dict) and "arroz" in str(r.get("name", "")).lower()), None)
    assert item is not None
    assert "Genérico" in str(item.get("display_qty", "")), (
        f"la lista debe decir la marca que usa (Genérico incluido): {item.get('display_qty')}"
    )


def test_anchor_in_source():
    assert "P1-BRAND-GENERIC-LABEL" in SRC
    assert 'or "Genérico"' in SRC, "el fallback Genérico vive en _pkg_from_product_row (SSOT prefs+defaults)"
