"""[P1-BRAND-LIST-VISIBILITY · 2026-07-06] Marcas del súper visibles en la lista/PDF.

Pedido del owner: (a) la lista de compras debe ENSEÑAR la marca de cada alimento;
(b) por default la IA usa la marca MÁS BARATA del súper; (c) la marca elegida a mano
en "Marcas del súper" gana sobre el default, sube el total, y queda como
predeterminado permanente para futuros planes (user_brand_preferences ya era
user-scoped — todas las superficies pasan por get_shopping_list_delta).

Implementación: overlay de `market_packages` con los productos reales de
`supermarket_products` (marca+precio vivo) para ítems SIN preferencia manual;
`_select_market_package` (costo-óptimo, P1-PKG-COST-OPTIMAL) elige la marca más
barata y su label "2 lb · La Garza" fluye a display_qty → panel por pasillo y PDF.
"""
import re
from pathlib import Path

import pytest

import shopping_calculator as sc

BACKEND = Path(__file__).resolve().parents[1]
SRC = (BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _fresh_defaults_cache():
    sc._brand_defaults_cache["data"] = None
    sc._brand_defaults_cache["at"] = 0.0
    yield
    sc._brand_defaults_cache["data"] = None
    sc._brand_defaults_cache["at"] = 0.0


# ---------------------------------------------------------------------------
# 1. _pkg_from_product_row — SSOT compartido prefs/defaults
# ---------------------------------------------------------------------------
def test_pkg_row_size_grams_authoritative():
    pkg = sc._pkg_from_product_row({
        "presentation": "Funda 999 gr", "brand": "La Garza",
        "price_rd": 165.0, "size_grams": 907.0,
    })
    assert pkg is not None and pkg["grams"] == 907.0, "size_grams explícito debe ganar al parser"


def test_pkg_row_presentation_fallback_and_label():
    pkg = sc._pkg_from_product_row({
        "presentation": "Funda 800 gr", "brand": "La Sanjuanera",
        "price_rd": 99.0, "size_grams": None,
    })
    assert pkg is not None
    assert abs(pkg["grams"] - 800.0) < 0.01
    assert pkg["label"] == "800 gr · La Sanjuanera", "label = tamaño · Marca (la MARCA visible)"
    assert pkg["unit"] == "funda"


def test_pkg_row_brandless_label():
    pkg = sc._pkg_from_product_row({
        "presentation": "Paquete 2 Lb", "brand": "", "price_rd": 165.0, "size_grams": None,
    })
    assert pkg is not None and pkg["label"] == "2 Lb", "sin marca → label solo tamaño (sin ' ·' colgando)"


@pytest.mark.parametrize("row", [
    {"presentation": "Paquete 2L", "brand": "X", "price_rd": 100.0, "size_grams": None},  # L ambigua
    {"presentation": "Funda 800 gr", "brand": "X", "price_rd": 0, "size_grams": None},     # sin precio
    {"presentation": "Funda 800 gr", "brand": "X", "price_rd": None, "size_grams": None},
    {"presentation": "", "brand": "X", "price_rd": 100.0, "size_grams": None},             # sin tamaño
])
def test_pkg_row_fail_open(row):
    assert sc._pkg_from_product_row(row) is None


# ---------------------------------------------------------------------------
# 2. fetch_brand_default_packages — agrupación, orden barata-primero, cache TTL
# ---------------------------------------------------------------------------
def _rows_super():
    return [
        {"food_name": "Arroz blanco", "brand": "Cariño", "presentation": "Paquete 2 Lb",
         "price_rd": 145.0, "size_grams": 907.0},
        {"food_name": "Arroz blanco", "brand": "La Garza", "presentation": "Paquete 2 Lb",
         "price_rd": 165.0, "size_grams": 907.0},
        {"food_name": "Avena", "brand": "Quaker", "presentation": "Tarro 400 gr",
         "price_rd": 175.0, "size_grams": None},
        {"food_name": "Avena", "brand": "Selecta", "presentation": "Funda 400 gr",
         "price_rd": 98.0, "size_grams": None},
        # inservible (sin tamaño) → excluido sin romper el resto
        {"food_name": "Avena", "brand": "Rota", "presentation": "Paquete", "price_rd": 50.0,
         "size_grams": None},
    ]


def test_fetch_defaults_groups_and_sorts_cheapest_first(monkeypatch):
    monkeypatch.setattr(sc, "execute_sql_query", lambda *a, **k: _rows_super())
    out = sc.fetch_brand_default_packages()
    assert set(out.keys()) == {"arroz blanco", "avena"}
    assert [p["price"] for p in out["arroz blanco"]] == [145.0, 165.0], "más barata primero"
    assert out["avena"][0]["label"].endswith("Selecta"), "la marca más barata encabeza"
    assert len(out["avena"]) == 2, "producto sin tamaño usable queda fuera (fail-open del ítem)"


def test_fetch_defaults_cached_within_ttl(monkeypatch):
    calls = {"n": 0}

    def _q(*a, **k):
        calls["n"] += 1
        return _rows_super()

    monkeypatch.setattr(sc, "execute_sql_query", _q)
    sc.fetch_brand_default_packages()
    sc.fetch_brand_default_packages()
    assert calls["n"] == 1, "segunda llamada dentro del TTL debe salir del cache global (crons/recalcs)"


def test_fetch_defaults_fail_open(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("db caída")

    monkeypatch.setattr(sc, "execute_sql_query", _boom)
    assert sc.fetch_brand_default_packages() == {}, "error → {} (el costeo JAMÁS se rompe por marcas)"


def test_fetch_defaults_caps_variants_per_food(monkeypatch):
    rows = [
        {"food_name": "Arroz blanco", "brand": f"M{i}", "presentation": "Paquete 2 Lb",
         "price_rd": 100.0 + i, "size_grams": 907.0}
        for i in range(20)
    ]
    monkeypatch.setattr(sc, "execute_sql_query", lambda *a, **k: rows)
    out = sc.fetch_brand_default_packages()
    assert len(out["arroz blanco"]) == sc._BRAND_DEFAULTS_MAX_PER_FOOD


# ---------------------------------------------------------------------------
# 3. Overlay en el aggregator: default con marca → display; pref manual GANA
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "arroz blanco": [
        {"grams": 907.0, "price": 145.0, "label": "2 Lb · Cariño", "unit": "paquete"},
        {"grams": 907.0, "price": 165.0, "label": "2 Lb · La Garza", "unit": "paquete"},
    ],
}


def _find_item(result, needle):
    for r in result:
        if isinstance(r, dict) and needle in str(r.get("name", "")).lower():
            return r
    return None


def test_default_brand_shows_in_display():
    result = sc.aggregate_and_deduct_shopping_list(
        ["800g de arroz blanco"], [], structured=True, brand_defaults=_DEFAULTS,
    )
    item = _find_item(result, "arroz")
    assert item is not None, f"arroz ausente del agregado: {result}"
    assert "Cariño" in str(item.get("display_qty", "")), (
        f"la marca MÁS BARATA debe enseñarse en el display (lista+PDF): {item.get('display_qty')}"
    )
    assert float(item.get("estimated_cost_rd") or item.get("estimated_cost") or 0) == 145.0


def test_manual_pref_wins_over_default_and_raises_cost():
    pref = {"arroz blanco": {"grams": 907.0, "price": 210.0, "label": "2 Lb · Premium", "unit": "paquete"}}
    result = sc.aggregate_and_deduct_shopping_list(
        ["800g de arroz blanco"], [], structured=True,
        brand_prefs=pref, brand_defaults=_DEFAULTS,
    )
    item = _find_item(result, "arroz")
    assert item is not None
    disp = str(item.get("display_qty", ""))
    assert "Premium" in disp and "Cariño" not in disp, (
        f"la preferencia MANUAL debe ganar al default más barato: {disp}"
    )
    assert float(item.get("estimated_cost_rd") or item.get("estimated_cost") or 0) == 210.0, (
        "el total debe subir al precio de la marca elegida (el usuario paga su gusto)"
    )


def test_no_defaults_keeps_previous_behavior():
    """brand_defaults=None (knob off / fail-open) → display idéntico al previo (sin marca)."""
    result = sc.aggregate_and_deduct_shopping_list(
        ["800g de arroz blanco"], [], structured=True, brand_defaults=None,
    )
    item = _find_item(result, "arroz")
    assert item is not None
    disp = str(item.get("display_qty", ""))
    assert "Cariño" not in disp and "La Garza" not in disp


# ---------------------------------------------------------------------------
# 4. Anchors de wiring (parser-based)
# ---------------------------------------------------------------------------
def test_knob_default_on_and_reversible():
    assert re.search(r'_knob_env_bool\("MEALFIT_BRAND_DEFAULT_PACKAGES",\s*True\)', SRC), (
        "knob MEALFIT_BRAND_DEFAULT_PACKAGES debe existir con default True "
        "(rollback sin redeploy: =false)."
    )


def test_aggregate_accepts_brand_defaults_kwarg():
    assert re.search(
        r"def aggregate_and_deduct_shopping_list\([^)]*brand_defaults:\s*dict\s*\|\s*None\s*=\s*None",
        SRC,
    ), "aggregate_and_deduct_shopping_list perdió el kwarg brand_defaults=None (default = cero cambio)."


def test_overlay_pref_wins_ordering_in_source():
    m = re.search(r"if _pref_pkg is None and brand_defaults:", SRC)
    assert m, "el overlay de defaults debe estar gated por ausencia de preferencia manual"
    window = SRC[m.start():m.start() + 900]
    assert "master_item = dict(master_item)" in window, (
        "el overlay de defaults debe COPIAR master_item — mutarlo corrompería el cache global."
    )
    pref_block = SRC.index("_pref_pkg = _resolve_brand_pref(name, brand_prefs)")
    assert pref_block < m.start(), "la preferencia manual se resuelve ANTES que el default"


def test_delta_fetches_defaults_fail_open():
    start = SRC.index("def get_shopping_list_delta(")
    body = SRC[start:start + 5000]
    assert "_brand_default_packages_enabled()" in body, "el fetch debe estar gated por el knob"
    assert re.search(r"try:\s*\n\s*brand_defaults = fetch_brand_default_packages\(\)", body), (
        "fetch de defaults sin try/except — el costeo JAMÁS puede romperse por marcas (fail-open)."
    )
    assert "brand_defaults=brand_defaults" in SRC, "brand_defaults no se propaga al aggregate"
