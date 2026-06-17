"""[P2-PRICES-ENGINE-1 · 2026-06-16] Tests de la lógica pura del scraper de Nacional.

Lo crítico/frágil es `to_per_lb` (normaliza precio-de-paquete → RD$/libra) y el parser
Magento. Estos casos vienen de nombres REALES observados en vivo (2026-06-16). No tocan
la red — importan el módulo del script y prueban las funciones puras.
"""
import importlib.util
import os

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "fetch_nacional_prices.py")
_spec = importlib.util.spec_from_file_location("fetch_nacional_prices", _SCRIPT)
fnp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fnp)


@pytest.mark.parametrize("name,price,expected", [
    # Paquete: "N Lb" en el nombre → divide por el peso.
    ("Arroz Enriquecido La Garza 10 Lb", 426.95, 42.695),
    ("Arroz Premium Campos 5 Lb", 234.95, 46.99),
    # Malla con rango → promedio del rango (3-5 → 4).
    ("Papas Selectas Malla (Aprox. 3 a 5 Libras Por Paquete )", 224.0, 56.0),
    # Per-lb marker (", Lb" / "Por Libra") → el precio YA es por libra (NO dividir).
    ("Yuca Selecta Parafinada, Lb (Aprox. 1- 2 Unidades Por Libra)", 44.95, 44.95),
    ("Pollo Entero Fresco Empacado, Lb (Peso Aprox. 5 Libras)", 84.0, 84.0),
    # Marca per-lb tiene prioridad sobre el "Peso Aprox" descriptivo.
    ("Batata Roja, Lb (Peso Aprox. 2 Libra)", 29.0, 29.0),
])
def test_to_per_lb_real_world_names(name, price, expected):
    assert fnp.to_per_lb(name, price) == pytest.approx(expected, rel=1e-3)


def test_matches_include_and_exclude():
    assert fnp._matches("Arroz Premium Campos 5 Lb", ["arroz"], ["integral"]) is True
    assert fnp._matches("Arroz Integral Bisono 5 Lb", ["arroz"], ["integral"]) is False
    # accent-insensitive
    assert fnp._matches("Plátano Verde, Und", ["platano verde"], []) is True


def test_norm_strips_accents():
    assert fnp._norm("Plátano Ñame ÁÉÍ") == "platano name aei"


def test_parser_regex_extracts_name_and_price():
    html = (
        '<span id="product-price-26" data-price-amount="84.00" data-price-type="finalPrice" '
        'class="price-wrapper"><span class="price">$84.00</span></span>'
        '<strong class="product name product-item-name"><a href="/x">Pollo Entero, Lb</a></strong>'
    )
    prices = fnp._PRICE_RE.findall(html)
    names = fnp._NAME_RE.findall(html)
    assert prices == ["84.00"]
    assert names == ["Pollo Entero, Lb"]


def test_categories_and_match_are_consistent():
    # Toda categoría referenciada en MATCH debe existir en CATEGORIES.
    referenced = {c for rule in fnp.MATCH.values() for c in rule[0]}
    missing = referenced - set(fnp.CATEGORIES)
    assert not missing, f"MATCH referencia categorías sin URL: {missing}"


def test_to_per_lb_handles_grams_and_onzas():
    # "800 Gr" (no "g" exacto) → divide por 1.764 lb. Regresión del bug gr/onz.
    assert fnp.to_per_lb("Habichuela Roja Larga 800 Gr", 88.0) == pytest.approx(49.9, rel=0.02)
    assert fnp.to_per_lb("Garbanzo Goya 14 Onz", 90.0) == pytest.approx(102.9, rel=0.03)


def test_estimate_prices_grams_to_per_lb():
    per_lb, per_unit = fnp.estimate_prices([("Habichuela Roja Larga 800 Gr", 88.0)])
    assert per_lb == pytest.approx(49.9, rel=0.02)
    assert per_unit is None


def test_estimate_prices_count_unit_from_per_lb_and_und():
    products = [
        ("Manzana Gala, Lb (Aprox. 2-3 unidades Por Libra)", 90.0),  # 90/lb ÷ 2.5 ≈ 36/u
        ("Manzana Red Delicious Bebe, Und", 18.0),                    # 18/u
    ]
    per_lb, per_unit = fnp.estimate_prices(products)
    assert per_lb == pytest.approx(90.0, rel=0.01)       # per-lb del producto vendido por libra
    assert per_unit is not None and 15 <= per_unit <= 40  # pieza derivada


def test_estimate_prices_pack_with_count():
    # "Paq. (Aprox. 4 Unidades)" @ 60 → 15/pieza
    per_lb, per_unit = fnp.estimate_prices([("Ajo Selecto Premium Paq. (Aprox. 4 Unidades)", 60.0)])
    assert per_unit == pytest.approx(15.0, rel=0.01)
