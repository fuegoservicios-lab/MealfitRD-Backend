"""[P1-BRAND-DEFAULT-PRESELECTED · 2026-07-06] El picker pre-selecciona la marca
que la LISTA está usando.

Pedido del owner: "en la lista está Wala por defecto en arroz blanco, así que
debe verse seleccionado Wala en el menú de marcas; la yuca también — así los
usuarios no se confunden". Diseño: el default se muestra con check verde HUECO
(punteado) — distinto de la preferencia manual (check sólido) — y tocarlo lo FIJA
como preferencia permanente. Fallback: ítem con UNA sola variante en el catálogo
(yuca/laurel) se muestra como default aunque el costeo fuera a granel.

Plumbing: `_pkg_from_product_row` lleva `id` del producto → `_select_market_package`
lo arrastra al envase elegido → `apply_smart_market_units` lo expone como
`brand_product_id` del ítem → el picker matchea contra `variant.id` del /match.
"""
import pytest

import shopping_calculator as sc

BRANDS_JSX = None


def _jsx():
    global BRANDS_JSX
    if BRANDS_JSX is None:
        from pathlib import Path
        BRANDS_JSX = (Path(sc.__file__).resolve().parent.parent / "frontend" / "src"
                      / "components" / "dashboard" / "SupermarketBrands.jsx").read_text(encoding="utf-8")
    return BRANDS_JSX


_MASTER = [
    {"name": "Arroz blanco", "category": "Despensa", "market_container": "paquete",
     "container_weight_g": 907.0, "price_per_lb": 40.0, "default_unit": "paquete",
     "shelf_life_days": 365, "aliases": []},
]


@pytest.fixture()
def master_stub(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_MASTER))
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


# ───────────── backend: el id fluye hasta el ítem ─────────────

def test_pkg_row_carries_product_id():
    pkg = sc._pkg_from_product_row({
        "id": "abc-123", "presentation": "Funda Selecto 1 Lb", "brand": "Wala",
        "price_rd": 42.0, "size_grams": None,
    })
    assert pkg is not None and pkg["id"] == "abc-123"


def test_select_market_package_carries_id():
    sel = sc._select_market_package(900.0, [
        {"grams": 453.6, "price": 42.0, "label": "1 Lb · Wala", "unit": "funda", "id": "wala-1"},
        {"grams": 907.2, "price": 145.0, "label": "2 Lb · Cariño", "unit": "paquete", "id": "car-2"},
    ])
    assert sel is not None and sel.get("id") in ("wala-1", "car-2"), "el envase elegido conserva su id"


def test_item_exposes_brand_product_id(master_stub):
    defaults = {"arroz blanco": [
        {"grams": 907.184, "price": 145.0, "label": "2 Lb · Cariño", "unit": "paquete", "id": "car-2lb"},
    ]}
    result = sc.aggregate_and_deduct_shopping_list(
        ["800g de arroz blanco"], [], structured=True, brand_defaults=defaults,
    )
    item = next((r for r in result if isinstance(r, dict) and "arroz" in str(r.get("name", "")).lower()), None)
    assert item is not None
    assert item.get("brand_product_id") == "car-2lb", (
        f"el ítem debe decir QUÉ producto usa el costeo: {item.get('brand_product_id')}"
    )


def test_fetchers_select_product_id():
    from pathlib import Path
    src = (Path(sc.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")
    i = src.index("def fetch_brand_pref_packages(")
    assert "sp.id::text AS id" in src[i:i + 2500], "prefs fetch debe traer el id del producto"
    j = src.index("def fetch_brand_default_packages(")
    assert "sp.id::text AS id" in src[j:j + 2500], "defaults fetch debe traer el id del producto"


# ───────────── frontend: pre-selección visual ─────────────

def test_picker_preselects_list_brand():
    jsx = _jsx()
    assert "P1-BRAND-DEFAULT-PRESELECTED" in jsx
    assert "defaultIdByKey" in jsx and "brand_product_id" in jsx
    assert "isDefault" in jsx, "la fila default se marca visualmente (check hueco)"
    assert "dashed" in jsx, "estilo DISTINTO de la preferencia manual (no confundir elección vs default)"


def test_single_variant_fallback():
    jsx = _jsx()
    assert "all.length === 1" in jsx, (
        "ítem con UNA variante (yuca/laurel) se muestra como default aunque el costeo sea a granel"
    )


def test_tapping_default_pins_it():
    jsx = _jsx()
    assert "tócala para fijarla como tu preferida" in jsx, (
        "tocar el default lo convierte en preferencia permanente (persistPref con su id)"
    )
