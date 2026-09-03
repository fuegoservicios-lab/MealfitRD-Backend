"""[P1-COUNTRY-KEEP-COUNT-UNITS · 2026-08-23]

Las filas beta sin precio deben conservar los conteos de la receta. El guard
mide la salida comprable y su escalamiento; no exige un texto de display fijo.
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc


@pytest.fixture(autouse=True)
def verified_only(monkeypatch):
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    monkeypatch.setenv("MEALFIT_COUNTRY_KEEP_RESPECT_RECIPE_QTY", "true")
    # G07 prueba la rama estática de catálogo-país. El master vivo puede
    # canonicalizar una fila beta hacia su gemela DO (G08) y cambiar la rama
    # observada según el estado de producción, algo ajeno a este contrato.
    monkeypatch.setattr(sc, "_build_shopping_master_map", lambda: {})


def _items(lines):
    result = sc.aggregate_and_deduct_shopping_list(lines, structured=True)
    return result.get("items") if isinstance(result, dict) else result


def _item_containing(lines, name_fragment):
    needle = name_fragment.casefold()
    return next(i for i in _items(lines) if needle in str(i.get("name") or "").casefold())


@pytest.mark.parametrize("line,name_fragment,expected_count", [
    ("3 unidades de Membrillo", "Membrillo", 3.0),
    ("2 unidades de Higo", "Higo", 2.0),
    ("2 chiles poblanos", "poblano", 2.0),
])
def test_conteo_beta_llega_como_conteo_al_motor_de_compra(line, name_fragment, expected_count):
    item = _item_containing([line], name_fragment)
    assert item["base_unit"] == "unidad"
    assert float(item["base_qty"]) == pytest.approx(expected_count)
    assert float(item["market_qty_numeric"]) == pytest.approx(expected_count)


def test_qty_de_lista_escala_con_numero_de_demandas():
    one = _item_containing(["3 unidades de Membrillo"], "Membrillo")
    three = _item_containing(["3 unidades de Membrillo"] * 3, "Membrillo")
    assert float(three["base_qty"]) == pytest.approx(3 * float(one["base_qty"]))
    assert float(three["market_qty_numeric"]) == pytest.approx(
        3 * float(one["market_qty_numeric"])
    )


def test_espejo_do_y_beta_comparten_el_contrato_de_conteo():
    beta = _item_containing(["3 unidades de Membrillo"], "Membrillo")
    do = sc.apply_smart_market_units("Manzana", 0.0, "unidad", 3.0, {})
    for key in ("base_qty", "base_unit", "market_qty_numeric"):
        assert beta[key] == do[key]


def test_nominal_no_se_disfraza_de_conteo_y_knob_restaura_default(monkeypatch):
    assert sc._country_keep_has_recipe_qty({"pizca": 1.0}) is False
    assert sc._country_keep_has_recipe_qty({"unidad": 2.0}) is True
    monkeypatch.setenv("MEALFIT_COUNTRY_KEEP_RESPECT_RECIPE_QTY", "false")
    assert sc._country_keep_has_recipe_qty({"unidad": 2.0}) is False
    item = _item_containing(["3 unidades de Membrillo"], "Membrillo")
    assert item["base_unit"] == "g"
    assert float(item["base_qty"]) == pytest.approx(sc._COUNTRY_CATALOG_UNPRICED_DEFAULT_G)
