# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-285 · 2026-09-25] La receta dice «de lata»: la lista compra la legumbre LISTA, en su cantidad.

Con «Nada» de tiempo la receta pide «½ taza de habichuelas negras de lata, escurridas». La lista la contaba como
legumbre SECA (el 0,35× de lo cocido no miraba «de lata»: 3× de más) y elegía por precio la funda seca — la que hay que
remojar y hervir una hora."""
from __future__ import annotations

import pathlib

import pytest

import envase_legumbre as el
import shopping_calculator as sc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]

_MASTER = [
    {"name": "Habichuelas negras", "category": "Despensa", "kcal_per_100g": 348.8, "density_g_per_cup": 180,
     "market_container": "lata", "container_weight_g": 425, "default_unit": "lb", "price_per_lb": 0,
     "price_per_unit": 88, "shelf_life_days": 180, "aliases": ["habichuela negra", "frijoles negros"],
     "market_packages": [{"unit": "lata", "grams": 425, "label": "15 oz", "price": 88},
                         {"unit": "paquete", "grams": 2000, "label": "800 g seco", "price": 105}]},
]


@pytest.fixture()
def master_stub(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: [dict(r) for r in _MASTER])
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


def _item(result, needle="habichuela"):
    return next((r for r in result if isinstance(r, dict) and needle in str(r.get("name", "")).lower()), None)


def test_de_lata_es_cocida_para_la_lista():
    y = lambda s: sc._calculate_yield_multiplier(s, only_legumbres_grains=True)   # noqa: E731
    assert y("habichuelas negras de lata, escurridas") == 0.35
    assert y("garbanzos en lata") == 0.35 and y("lentejas enlatadas") == 0.35
    assert y("frijoles horneados de lata") == 1.0          # su fila ya está en cocido
    assert y("atún en lata") == 1.0 and y("habichuelas negras secas") == 1.0


def test_la_forma_que_pide_cada_linea():
    assert el.linea_lista("½ taza de habichuelas negras de lata, escurridas") is True
    assert el.linea_lista("85 g de garbanzos cocidos y escurridos") is True
    assert el.linea_lista("40 g de habichuelas negras secas") is False
    assert el.linea_lista("½ taza de habichuelas rojas cocidas") is None      # «cocidas» no dice cómo se compra
    assert el.linea_lista("1 taza de arroz") is None
    formas: dict = {}
    el.anotar_forma(formas, "Habichuelas negras", "½ taza de habichuelas negras de lata")
    assert el.prefiere_listo(formas, "Habichuelas negras")
    el.anotar_forma(formas, "Habichuelas negras", "40 g de habichuelas negras secas")
    assert not el.prefiere_listo(formas, "Habichuelas negras")                 # una línea seca: decide el precio


def test_solo_listos_cae_al_catalogo_si_el_super_no_trae_latas():
    catalogo = _MASTER[0]
    super_sin_latas = dict(catalogo, market_packages=[{"unit": "funda", "grams": 800, "label": "800 gr · Wala", "price": 105}])
    out = el.solo_listos("Habichuelas negras", super_sin_latas, catalogo)
    assert [p["unit"] for p in out["market_packages"]] == ["lata"]
    tetra = dict(catalogo, market_packages=[{"unit": "carton", "grams": 400, "label": "Tetra 400 gr · Rica", "price": 78},
                                            {"unit": "funda", "grams": 400, "label": "400 gr · Giselle", "price": 65}])
    assert [p["unit"] for p in el.solo_listos("Habichuelas negras", tetra, catalogo)["market_packages"]] == ["carton"]
    seca = dict(catalogo, market_packages=[{"unit": "funda", "grams": 800, "label": "800 gr", "price": 105}])
    assert el.solo_listos("Habichuelas negras", seca, seca) is seca


def test_la_lista_compra_latas_para_la_receta_de_lata(master_stub):
    """6 × ½ taza escurrida = 3 tazas ≈ 2,1 latas de 15 oz (se compran 2-3). Antes: 540 g «secos» → la funda seca."""
    result = sc.aggregate_and_deduct_shopping_list(
        ["½ taza de habichuelas negras de lata, escurridas"] * 6, [], structured=True, brand_defaults=None)
    it = _item(result)
    assert it is not None, result
    assert it.get("market_unit") == "lata" and 2 <= float(it.get("market_qty") or 0) <= 3, it


def test_con_una_linea_seca_decide_el_precio(master_stub):
    result = sc.aggregate_and_deduct_shopping_list(
        ["½ taza de habichuelas negras de lata, escurridas"] * 3 + ["60 g de habichuelas negras secas"] * 3, [],
        structured=True, brand_defaults=None)
    it = _item(result)
    assert it is not None and it.get("market_unit") == "paquete", it


def test_la_marca_elegida_manda(master_stub):
    pref = {"habichuelas negras": {"grams": 800.0, "price": 105.0, "label": "Funda 800 gr · Wala", "unit": "funda"}}
    result = sc.aggregate_and_deduct_shopping_list(
        ["½ taza de habichuelas negras de lata, escurridas"] * 6, [], structured=True, brand_prefs=pref,
        brand_defaults=None)
    it = _item(result)
    assert it is not None and "Wala" in str(it.get("display_qty")), it


def test_ganchos():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert '__import__("envase_legumbre").anotar_forma(_formas_legumbre, name, item)  # [P1-PLAN-LOTE-285]' in src
    assert "if _pref_pkg is None and __import__(\"envase_legumbre\").prefiere_listo(_formas_legumbre, name):" in src
    assert "tooltip-anchor: P1-PLAN-LOTE-285-LISTA-LISTA" in (_BACKEND / "envase_legumbre.py").read_text(encoding="utf-8")
