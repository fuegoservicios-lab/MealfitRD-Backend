"""[P1-BRAND-SIZE-FILTER · 2026-07-06] Picker de marcas filtrado al tamaño de la LISTA.

Feedback del owner sobre el picker post P1-BRAND-LIST-VISIBILITY: para "Arroz
blanco — 1 paquete (2 lb)" el panel enseñaba Genérico 2L/5L/10L + Wala 1 Lb (cap
4 comido por tamaños redundantes) y mandaba al catálogo por más. Debe enseñar las
MARCAS del tamaño que la lista ya eligió (2 lb), más económica primero, inline.

Tres piezas:
1. Parser: "L" suelta tras envase SÓLIDO (paquete/funda/saco/sobre/caja) = LIBRA
   (los líquidos del catálogo usan "Lt"/"Ml" explícitos) — los genéricos del sync
   ("Paquete 2L") recuperan tamaño → overlay de costeo + filtro los ven.
2. `package_grams` en cada ítem estructurado (envase elegido por el costeo).
3. `/match` expone `size_g` efectivo por variante; SupermarketBrands filtra ±15%.
"""
import re
from pathlib import Path

import pytest

import shopping_calculator as sc

BACKEND = Path(__file__).resolve().parents[1]
SRC = (BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
SUP = (BACKEND / "routers" / "supermarket.py").read_text(encoding="utf-8")
BRANDS_JSX = (BACKEND.parent / "frontend" / "src" / "components" / "dashboard"
              / "SupermarketBrands.jsx").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Parser: "L" tras envase sólido = libra; resto sigue fail-open
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pres,expected", [
    ("Paquete 2L", 2 * 453.592),
    ("Paquete 2 L", 2 * 453.592),
    ("Funda 5L", 5 * 453.592),
    ("Saco 10L", 10 * 453.592),
    ("Caja 1.5L", 1.5 * 453.592),
])
def test_solid_container_L_is_libra(pres, expected):
    got = sc._parse_presentation_grams(pres)
    assert got is not None and abs(got - expected) < 0.01, f"{pres} → {got} ≠ {expected}"


@pytest.mark.parametrize("pres", [
    "Botella 2L",   # líquido probable → litro ambiguo → fail-open (como siempre)
    "Cartón L",     # sin número
    "Malla 3L",     # envase no-sólido-staple → conservador
])
def test_ambiguous_L_still_fail_open(pres):
    assert sc._parse_presentation_grams(pres) is None


def test_meat_ratio_notation_unaffected():
    # "80/20 Lb" (ratio magro/grasa) sigue cayendo a venta-por-libra, jamás "20 lb".
    assert abs(sc._parse_presentation_grams("80/20 Lb") - 453.592) < 0.01
    assert abs(sc._parse_presentation_grams("Estelar 1.47 Lb") - 1.47 * 453.592) < 0.01


# ---------------------------------------------------------------------------
# 2. package_grams en el ítem estructurado
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "arroz blanco": [
        {"grams": 907.184, "price": 145.0, "label": "2 Lb · Cariño", "unit": "paquete"},
        {"grams": 2267.96, "price": 235.0, "label": "5 Lb · Cariño", "unit": "paquete"},
    ],
}


def test_item_carries_package_grams():
    result = sc.aggregate_and_deduct_shopping_list(
        ["800g de arroz blanco"], [], structured=True, brand_defaults=_DEFAULTS,
    )
    item = next((r for r in result if isinstance(r, dict) and "arroz" in str(r.get("name", "")).lower()), None)
    assert item is not None, f"arroz ausente: {result}"
    assert abs(float(item.get("package_grams") or 0) - 907.184) < 0.5, (
        f"el ítem debe exponer el tamaño del envase elegido (2 lb): {item.get('package_grams')}"
    )


def test_package_grams_anchor_in_result_dict():
    i = SRC.index('"display_qty": display_qty_final,')
    win = SRC[i:i + 1200]
    assert 'result["package_grams"]' in win, (
        "apply_smart_market_units debe exponer package_grams junto al resto del ítem"
    )


# ---------------------------------------------------------------------------
# 3. /match expone size_g efectivo por variante
# ---------------------------------------------------------------------------
def test_match_selects_size_grams_and_exposes_size_g():
    start = SUP.index("def api_supermarket_match(")
    body = SUP[start:start + 4000]
    assert "size_grams::float8 AS size_grams" in body, "/match debe traer size_grams del catálogo"
    assert '"size_g": _size_g(r)' in body, "cada variante debe exponer size_g efectivo"
    assert "_parse_presentation_grams" in body, (
        "el fallback de tamaño debe ser el parser SSOT del costeo (no duplicar lógica)"
    )


# ---------------------------------------------------------------------------
# 4. Frontend: filtro por tamaño (anchors sobre el JSX)
# ---------------------------------------------------------------------------
def test_frontend_filters_by_list_package_size():
    assert "package_grams" in BRANDS_JSX, "el picker debe leer package_grams del ítem de la lista"
    assert "SIZE_TOLERANCE" in BRANDS_JSX and "MAX_SIZED_SHOWN" in BRANDS_JSX
    assert "sizeFilteredVariants" in BRANDS_JSX
    assert re.search(r"v\.id === chosenId", BRANDS_JSX), (
        "la variante YA elegida debe mostrarse aunque sea de otro tamaño (para des-seleccionar)"
    )
    assert "price_rd ?? Infinity" in BRANDS_JSX, "orden por precio: la más económica primero"


def test_frontend_fallback_when_no_size():
    # Planes viejos sin package_grams → comportamiento previo (todas, cap 4).
    assert re.search(r"if \(!targetG \|\| !Array\.isArray\(variants\)\) return null;", BRANDS_JSX)
    assert "MAX_VARIANTS_SHOWN" in BRANDS_JSX
