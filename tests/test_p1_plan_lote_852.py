"""[P1-PLAN-LOTE-852 · 2026-09-29] La lista de compras de un país beta no lleva productos del súper de RD.

LO QUE SE MIDIÓ (validación beta G24 del 29-sep, 6 planes reales con P1-PLAN-LOTE-815): entre 14 y 20
alimentos de cada lista beta (ES/US/MX/PR/CO) llevaban `brand_product_id` de un producto de
`supermarket_products` —el súper DOMINICANO— con su envase («1 funda (1 Lb) de Sal», «Selecto 1 Lb»,
«12 Oz», «Queso ricotta · Sosua» en CO: la marca se colaba porque el rótulo lleva paréntesis anidados y
el saneador de marcas de P1-BETA-PRICE-LEAKS no la ve), y entre 17 y 22 el campo `market_pkg_price_rd`
(el precio RD del envase elegido; `beta_no_prices` sólo anulaba `estimated_cost_rd`).

CAUSA: `get_shopping_list_delta` pedía a `supermarket_products` las marcas default (y las preferencias del
usuario) para TODA lista, y el agregador las ponía encima del envase del catálogo. El país de la lista ya
viajaba por contexto (`envase_pais.con_pais_del_plan`, lote 790) pero nadie lo miraba ahí.

ARREGLO (`lista_sin_super_rd.py`): con el país de la lista distinto de RD, ni marcas default ni
preferencias (no se consultan), y al final del agregador se quitan `brand_product_id` y
`market_pkg_price_rd` de cada ítem. El envase sale del catálogo (lotes 790/791). Sin país en contexto
(chat, swap) o con RD: idéntico. Knob `MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS` (default on).
"""
from __future__ import annotations

import copy
import re
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_BETA = ("ES", "US", "MX", "PR", "CO")


def _fila(nombre, cat, du, **kw):
    base = {"name": nombre, "category": cat, "aliases": [], "default_unit": du, "price_per_lb": 0.0,
            "price_per_unit": 0.0, "market_container": None, "container_weight_g": None,
            "available_sizes_g": None, "market_packages": None, "density_g_per_unit": None,
            "density_g_per_cup": None, "shelf_life_days": 180, "name_en": None}
    base.update(kw)
    return base


# Filas con la FORMA de las de producción (SELECT del 29-sep): Arroz blanco y Yogurt llevan
# `market_packages` del catálogo con precio RD; Sal va en funda.
_CATALOGO = [
    _fila("Arroz blanco", "Despensa", "lb", aliases=["arroz"], price_per_lb=40.0, name_en="White rice",
          market_container="funda", container_weight_g=453.6, available_sizes_g=[453.6, 2268],
          market_packages=[{"unit": "funda", "grams": 453.6, "label": "1 lb", "price": 45},
                           {"unit": "funda", "grams": 2268, "label": "5 lb", "price": 210}]),
    _fila("Yogurt", "Lácteos", "pote", aliases=["yogurt regular"], price_per_lb=164.0, price_per_unit=54.0,
          name_en="Yogurt", market_container="pote", container_weight_g=150, available_sizes_g=[150, 1960],
          market_packages=[{"grams": 150, "label": "150 g", "price": 100},
                           {"grams": 1960, "label": "1.96 kg", "price": 220}], shelf_life_days=14),
    _fila("Sal", "Despensa", "funda", aliases=["sal de mesa"], price_per_unit=17.0, name_en="Salt",
          market_container="funda", container_weight_g=453.6, available_sizes_g=[453.6]),
    # Legumbre en base SECA con la lata y la funda del súper (forma de producción: «1 lb seco» pesa 1135).
    _fila("Garbanzos", "Despensa", "lb", aliases=["garbanzo"], price_per_lb=70.0, price_per_unit=83.0,
          name_en="Chickpeas", kcal_per_100g=364.0, market_container="lata", container_weight_g=425,
          available_sizes_g=[425, 1135],
          market_packages=[{"unit": "lata", "grams": 425, "label": "lata 15 oz", "price": 83},
                           {"unit": "paquete", "grams": 1135, "label": "1 lb seco", "price": 70}]),
    _fila("Leche de soya", "Lácteos", "carton", aliases=["leche de soja"], price_per_unit=150.0,
          name_en="Soy milk", market_container="carton", container_weight_g=946, available_sizes_g=[946],
          market_packages=[{"unit": "carton", "grams": 946, "label": "32 oz", "price": 150}], shelf_life_days=10),
    _fila("Manzana", "Frutas", "funda", aliases=["manzanas"], price_per_lb=90.0, name_en="Apple",
          market_container="funda", container_weight_g=1360.78, shelf_life_days=21,
          market_packages=[{"unit": "funda", "grams": 1360.78, "label": "Amarilla 3 Lb", "price": 265.0}]),
]

# Productos del súper dominicano, con la forma de `_pkg_from_product_row`.
_DEFAULTS_RD = {
    "arroz blanco": [{"grams": 453.6, "price": 38.0, "label": "Selecto 1 Lb · Wala", "unit": "funda",
                      "per_lb": False, "id": "sp-arroz-selecto"}],
    "sal": [{"grams": 453.6, "price": 17.0, "label": "1 Lb · Wala", "unit": "funda", "per_lb": False,
             "id": "sp-sal-wala"}],
}
_PREFS_RD = {"yogurt": {"grams": 907.0, "price": 150.0, "label": "32 Oz · Sosua", "unit": "pote",
                        "per_lb": False, "id": "sp-yogurt-sosua"}}
_MARCAS_RD = ("Wala", "Selecto", "Sosua")


@pytest.fixture
def sc(monkeypatch):
    import shopping_calculator as _sc
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.delenv("MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS", raising=False)
    monkeypatch.setattr(_sc, "_master_cache", [dict(r) for r in _CATALOGO])
    monkeypatch.setattr(_sc, "_master_cache_ts", time.time() + 10 ** 6)
    monkeypatch.setattr(_sc, "_VERIFIED_SHOPPING_NAMES", None, raising=False)
    llamadas = {"defaults": 0, "prefs": 0}

    def _defaults():
        llamadas["defaults"] += 1
        return copy.deepcopy(_DEFAULTS_RD)

    def _prefs(user_id):
        llamadas["prefs"] += 1
        return copy.deepcopy(_PREFS_RD)

    monkeypatch.setattr(_sc, "fetch_brand_default_packages", _defaults)
    monkeypatch.setattr(_sc, "fetch_brand_pref_packages", _prefs)
    _sc._llamadas_852 = llamadas
    return _sc


def _plan(pais):
    plan = {"_country": pais, "days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Arroz con yogur", "ingredients": [
            "300 g de arroz blanco", "400 g de yogurt", "1 cdta de sal"]}]}]}
    if pais != "DO":
        plan["_pricing_mode"] = "beta_no_prices"
    return plan


def _lista(sc, pais, uid="u-852"):
    res = sc.get_shopping_list_delta(uid, _plan(pais), True, False, True, 1.0,
                                     inventory_override=[], consumed_override=[])
    return {it["name"]: it for it in res if isinstance(it, dict)}


# ── A. Beta: ni producto, ni envase, ni precio del súper dominicano ─────────────────────────────

@pytest.mark.parametrize("pais", _BETA)
def test_la_lista_beta_no_lleva_productos_del_super_rd(sc, pais):
    items = _lista(sc, pais)
    assert {"Arroz blanco", "Yogurt", "Sal"} <= set(items), sorted(items)
    for nombre, it in items.items():
        assert "brand_product_id" not in it, (pais, nombre, it.get("brand_product_id"))
        assert "market_pkg_price_rd" not in it, (pais, nombre, it.get("market_pkg_price_rd"))
        texto = f"{it.get('display_string')} {it.get('display_qty')} {it.get('sku_size_label')}"
        assert not any(m in texto for m in _MARCAS_RD), (pais, nombre, texto)
    # El envase lo pone el catálogo (lotes 790/791), no el súper: el arroz sigue siendo comprable.
    assert items["Arroz blanco"].get("market_unit") == "funda", items["Arroz blanco"]


@pytest.mark.parametrize("pais", _BETA)
def test_la_lista_beta_ni_siquiera_consulta_el_super_rd(sc, pais):
    _lista(sc, pais)
    assert sc._llamadas_852 == {"defaults": 0, "prefs": 0}, sc._llamadas_852


# ── B. RD: idéntica (control) ───────────────────────────────────────────────────────────────────

def test_la_lista_dominicana_sigue_con_su_super(sc):
    items = _lista(sc, "DO")
    assert items["Arroz blanco"].get("brand_product_id") == "sp-arroz-selecto", items["Arroz blanco"]
    assert items["Yogurt"].get("brand_product_id") == "sp-yogurt-sosua", items["Yogurt"]
    assert items["Arroz blanco"].get("market_pkg_price_rd") == 38.0
    assert "Wala" in items["Sal"]["display_string"], items["Sal"]
    assert sc._llamadas_852 == {"defaults": 1, "prefs": 1}, sc._llamadas_852


def test_la_lista_dominicana_es_la_misma_con_el_knob_apagado(sc, monkeypatch):
    con = _lista(sc, "DO")
    monkeypatch.setenv("MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS", "false")
    sin = _lista(sc, "DO")
    assert con == sin


# ── C. Palanca y superficies sin país ───────────────────────────────────────────────────────────

def test_el_knob_apagado_devuelve_la_conducta_anterior(sc, monkeypatch):
    monkeypatch.setenv("MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS", "false")
    items = _lista(sc, "ES")
    assert items["Arroz blanco"].get("brand_product_id") == "sp-arroz-selecto", items["Arroz blanco"]
    assert "market_pkg_price_rd" in items["Arroz blanco"]


def test_sin_pais_en_contexto_el_agregador_no_cambia(sc):
    """Chat, swap y scripts llaman al agregador sin plan: el súper se sigue aplicando si se lo pasan."""
    res = sc.aggregate_and_deduct_shopping_list(["300 g de arroz blanco"], structured=True,
                                                brand_defaults=copy.deepcopy(_DEFAULTS_RD))
    it = next(i for i in res if i.get("name") == "Arroz blanco")
    assert it.get("brand_product_id") == "sp-arroz-selecto", it
    assert it.get("market_pkg_price_rd") == 38.0, it


def test_el_precio_rd_del_envase_del_catalogo_tampoco_viaja_en_beta(sc):
    """El `market_pkg_price_rd` no sólo venía del súper: el `market_packages` del catálogo lleva precio RD."""
    import envase_pais
    with envase_pais.lista_de_pais("MX"):
        res = sc.aggregate_and_deduct_shopping_list(["400 g de yogurt"], structured=True)
    it = next(i for i in res if i.get("name") == "Yogurt")
    assert "market_pkg_price_rd" not in it, it
    with envase_pais.lista_de_pais("DO"):
        res = sc.aggregate_and_deduct_shopping_list(["400 g de yogurt"], structured=True)
    it = next(i for i in res if i.get("name") == "Yogurt")
    assert it.get("market_pkg_price_rd") in (100, 220), it


# ── D. La talla del envase del catálogo, en el sistema del país (ES/MX/CO) ──────────────────────
# Los `market_packages` del catálogo son envases del Supermercado Nacional (`price_source='nacional_tienda'`):
# 109 de sus rótulos hablan en libras/onzas. En una lista métrica la talla se reescribe en g/kg/ml; el
# resto del rótulo (variedad, «seco», «lata») y la CUENTA de envases no cambian.

_LINEAS_CATALOGO = ["300 g de arroz blanco", "700 ml de leche de soya", "250 g de garbanzos secos",
                    "600 g de manzana"]


def _por_nombre(sc, pais, lineas=_LINEAS_CATALOGO):
    import envase_pais
    with envase_pais.lista_de_pais(pais):
        res = sc.aggregate_and_deduct_shopping_list(list(lineas), structured=True)
    return {i["name"]: i for i in res if isinstance(i, dict)}


@pytest.mark.parametrize("pais", ("ES", "MX", "CO"))
def test_la_talla_del_envase_del_catalogo_habla_el_sistema_metrico(sc, pais):
    items = _por_nombre(sc, pais)
    rd = _por_nombre(sc, "DO")
    assert "(2 lb)" in rd["Arroz blanco"]["display_string"] or "lb" in rd["Arroz blanco"]["display_string"], rd
    for nombre in ("Arroz blanco", "Leche de soya", "Garbanzos", "Manzana"):
        it = items[nombre]
        texto = f"{it.get('display_string')} | {it.get('sku_size_label')}"
        assert not re.search(r"\d\s*(?:oz|lbs?)\b", texto, re.IGNORECASE), (pais, nombre, texto)
        # Sólo el rótulo: la cuenta de envases es la misma que en RD.
        assert it["market_qty"] == rd[nombre]["market_qty"], (pais, nombre, it["market_qty"], rd[nombre]["market_qty"])
        assert it["market_unit"] == rd[nombre]["market_unit"], (pais, nombre)
    assert "(946 ml)" in items["Leche de soya"]["display_string"], items["Leche de soya"]
    assert "seco" in items["Garbanzos"]["display_string"], items["Garbanzos"]
    assert "Amarilla 1,4 kg" in items["Manzana"]["display_string"], items["Manzana"]


@pytest.mark.parametrize("pais", ("DO", "US", "PR", None))
def test_los_paises_imperiales_conservan_la_talla_del_catalogo(sc, pais):
    items = _por_nombre(sc, pais)
    assert "(32 oz)" in items["Leche de soya"]["display_string"], items["Leche de soya"]
    assert "Amarilla 3 Lb" in items["Manzana"]["display_string"], items["Manzana"]


def test_la_talla_metrica_tiene_su_propia_palanca(sc, monkeypatch):
    monkeypatch.setenv("MEALFIT_BETA_METRIC_PACKAGE_LABELS", "false")
    items = _por_nombre(sc, "ES")
    assert "(32 oz)" in items["Leche de soya"]["display_string"], items["Leche de soya"]


def test_la_talla_se_convierte_token_a_token():
    import lista_sin_super_rd as l
    assert l.talla_en_metrico("1 lb seco", "paquete") == "454 g seco"
    assert l.talla_en_metrico("lata 15 oz", "lata") == "lata 425 g"
    assert l.talla_en_metrico("32 oz", "carton") == "946 ml"
    assert l.talla_en_metrico("16 Oz", "botella") == "473 ml"
    assert l.talla_en_metrico("35.2 oz", "paquete") == "998 g"
    assert l.talla_en_metrico("Amarilla 3 Lb", "funda") == "Amarilla 1,4 kg"
    assert l.talla_en_metrico("80/20 Lb", "paquete") == "80/20 Lb", "proporción magro/grasa, no un peso"
    assert l.talla_en_metrico("4×150 g", "paquete") == "4×150 g"
    assert l.talla_en_metrico("", "paquete") == ""


# ── E. Anclas ───────────────────────────────────────────────────────────────────────────────────

def test_anclas():
    mod = (_BACKEND / "lista_sin_super_rd.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-852" in mod
    assert "MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS" in mod
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert src.count("[P1-PLAN-LOTE-852]") >= 3, "gate de prefs, gate de defaults y el saneo final"
    knobs = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "`MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS`" in knobs and "P1-PLAN-LOTE-852" in knobs
    assert "`MEALFIT_BETA_METRIC_PACKAGE_LABELS`" in knobs
    assert "MEALFIT_BETA_METRIC_PACKAGE_LABELS" in mod
    doc = (_BACKEND / "docs" / "envases_y_catalogo_por_pais.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-852" in doc
