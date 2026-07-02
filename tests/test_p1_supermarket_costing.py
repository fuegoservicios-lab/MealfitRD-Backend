"""[P1-SUPERMARKET-COSTING · 2026-07-02] Marca preferida del súper → costeo real
de la lista de compras (fase 3 de la conexión Supermercado RD).

Unit tests PUROS (parser de presentaciones, resolución de preferencias,
selección de envase único) + anchors parser-based del wiring en
shopping_calculator.py (kwarg, overlay, fetch fail-open, knob).
"""
import math
import re
from pathlib import Path

import pytest

import shopping_calculator as sc

BACKEND = Path(__file__).resolve().parents[1]
SRC = (BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Parser de presentaciones → gramos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pres,expected", [
    ("Funda 800 gr", 800.0),
    ("Lata 15 Oz", 15 * 28.3495),
    ("Tarro 14.5 Oz", 14.5 * 28.3495),
    ("Estelar 1.47 Lb", 1.47 * 453.592),
    ("Brik 290 Ml", 290.0),
    ("Botella 0.25 Lt", 250.0),
    ("Funda 1 Kg", 1000.0),
    ("Criolla Lb", 453.592),          # venta por libra sin número
    ("Importado Congelado Lb", 453.592),
    # Notación carnicería magro/grasa: el ratio NO es tamaño (bug real del
    # dry-run 2026-07-02: "80/20 Lb" parseaba 20 lb ≈ 9 kg) → venta por libra.
    ("80/20 Lb", 453.592),
    ("Cadera 96/4 Lb", 453.592),
    ("Bola 96/4 Lb", 453.592),
])
def test_parse_presentation_grams_ok(pres, expected):
    got = sc._parse_presentation_grams(pres)
    assert got is not None and abs(got - expected) < 0.01, f"{pres} → {got} ≠ {expected}"


@pytest.mark.parametrize("pres", [
    "Paquete 2L",        # "L" suelta AMBIGUA (libra en produce / litro en leche) → skip
    "Cartón L",
    "Caja 20 unid",      # conteos → path propio (huevos), no gramos
    "Malla Mini 5 unid",
    "Paquete",           # sin tamaño
    "",
    None,
])
def test_parse_presentation_grams_fail_open(pres):
    assert sc._parse_presentation_grams(pres) is None, (
        f"'{pres}' NO debe parsear — presentación ambigua/sin tamaño = costeo estándar (fail-open)."
    )


def test_parse_presentation_sanity_bounds():
    assert sc._parse_presentation_grams("Saco 500 Lb") is None, "fuera de rango (>50kg) debe fail-open"


# ---------------------------------------------------------------------------
# 2. Normalización + resolución de preferencias (simetría con /match)
# ---------------------------------------------------------------------------
def test_norm_pref_food_symmetry():
    assert sc._norm_pref_food("Kéfir") == sc._norm_pref_food("Kefir") == "kefir"
    assert sc._norm_pref_food("  Plátano   Verde ") == "platano verde"


def test_resolve_brand_pref_ladder():
    pkg_arroz = {"grams": 907.0, "price": 165.0, "label": "2 lb", "unit": "paquete"}
    pkg_pechuga = {"grams": 453.6, "price": 175.0, "label": "Lb", "unit": "libra"}
    prefs = {"arroz blanco": pkg_arroz, "filete pechuga de pollo": pkg_pechuga}
    # exacto (con acentos/case del lado del plan)
    assert sc._resolve_brand_pref("Arroz Blanco", prefs) is pkg_arroz
    # plural del plan → singular de la pref
    assert sc._resolve_brand_pref("Arroces Blancos", prefs) is None  # heurística no fuerza doble-singular
    # contención word-boundary: nombre del plan ⊂ food de la pref
    assert sc._resolve_brand_pref("Pechuga de pollo", prefs) is pkg_pechuga
    # 'sal' NO matchea 'salsa...' (padding con espacios)
    prefs2 = {"salsa de tomate": pkg_arroz}
    assert sc._resolve_brand_pref("Sal", prefs2) is None


# ---------------------------------------------------------------------------
# 3. Envase único preferido → compra ceil(g_total/grams) × precio real
# ---------------------------------------------------------------------------
def test_select_market_package_single_preferred():
    pref = [{"grams": 425.24, "price": 88.0, "label": "15 Oz · Goya", "unit": "lata"}]
    sel = sc._select_market_package(900.0, pref)
    assert sel is not None
    assert sel["count"] == math.ceil(900.0 / 425.24) == 3
    assert sel["price"] == 88.0
    assert sel["unit"] == "lata"


# ---------------------------------------------------------------------------
# 4. Anchors de wiring (parser-based)
# ---------------------------------------------------------------------------
def test_aggregate_accepts_brand_prefs_kwarg():
    assert re.search(
        r"def aggregate_and_deduct_shopping_list\([^)]*brand_prefs:\s*dict\s*\|\s*None\s*=\s*None",
        SRC,
    ), "aggregate_and_deduct_shopping_list perdió el kwarg brand_prefs=None (default = cero cambio)."


def test_overlay_lives_in_aggregate_loop():
    assert "_resolve_brand_pref(name, brand_prefs)" in SRC, "falta el overlay en el loop del aggregator"
    m = re.search(r"if brand_prefs:\s*\n\s*_pref_pkg = _resolve_brand_pref", SRC)
    assert m, "el overlay debe estar gated por `if brand_prefs:` (None = comportamiento idéntico)"
    window = SRC[m.start():m.start() + 900]
    assert "master_item = dict(master_item)" in window, (
        "el overlay debe COPIAR master_item — mutarlo corrompería el cache global de master_map."
    )


def test_delta_fetches_prefs_fail_open():
    start = SRC.index("def get_shopping_list_delta(")
    body = SRC[start:start + 4000]
    assert "_brand_pref_costing_enabled()" in body, "el fetch debe estar gated por el knob"
    assert re.search(r"try:\s*\n\s*brand_prefs = fetch_brand_pref_packages\(user_id\)", body), (
        "fetch de prefs sin try/except — el costeo JAMÁS puede romperse por prefs (fail-open)."
    )
    assert "brand_prefs=brand_prefs" in SRC, "brand_prefs no se propaga al aggregate"


def test_knob_default_on_and_reversible():
    assert re.search(r'_knob_env_bool\("MEALFIT_BRAND_PREF_COSTING",\s*True\)', SRC), (
        "knob MEALFIT_BRAND_PREF_COSTING debe existir con default True "
        "(rollback sin redeploy: =false)."
    )


def test_fetch_only_active_products_with_price():
    start = SRC.index("def fetch_brand_pref_packages(")
    body = SRC[start:start + 2500]
    assert "sp.active" in body and "price_rd IS NOT NULL" in body, (
        "el fetch debe filtrar productos activos con precio — una preferencia a un "
        "producto oculto/sin precio no puede costear la lista."
    )
    assert "WHERE p.user_id = %s" in body, "invariante I2: fetch anclado al user_id"
