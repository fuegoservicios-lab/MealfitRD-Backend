# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-282 · 2026-09-25] Una lata de habichuelas no son 425 g de habichuelas SECAS.

La necesidad de legumbres llega al selector de envases en gramos SECOS (el catálogo mide en seco y el motor convierte
lo cocido); los envases listos (lata, cartón Tetra, frasco de cocidas) se contaban por lo que pesan con su líquido y la
funda «800 g seco» del catálogo por su equivalente cocido (2000). Baterías del 25-sep: 1 lata o 1 cartón para 150-900 g
secos (hasta ~4,5× corta) y 1 funda para 1.814 g secos (2,3× corta), invisible al guard de coherencia."""
from __future__ import annotations

import pathlib

import envase_legumbre as el
import shopping_calculator as sc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]

_HABICHUELAS_NEGRAS = {
    "name": "Habichuelas negras", "category": "Despensa", "kcal_per_100g": 348.8,
    "market_container": "lata", "container_weight_g": 425, "default_unit": "lb",
    "price_per_lb": 0, "price_per_unit": 88,
    "market_packages": [
        {"unit": "lata", "grams": 425, "label": "15 oz", "price": 88},
        {"unit": "paquete", "grams": 2000, "label": "800 g seco", "price": 105},
    ],
}


def test_lata_carton_y_frasco_cuentan_su_legumbre_seca():
    f = el.factor_listo_a_seco()
    assert 0.18 <= f <= 0.24, f                                   # ~57 % escurrido × 1/2,7 de hidratación
    lata = el.envase_en_seco({"unit": "lata", "grams": 425, "label": "15 oz", "price": 50})
    assert lata["grams"] == round(425 * f, 1) and lata["label"] == "15 oz" and lata["price"] == 50
    tetra = el.envase_en_seco({"unit": "carton", "grams": 400, "label": "Tetra 400 gr · Rica", "price": 78})
    assert tetra["grams"] == round(400 * f, 1)
    frasco = el.envase_en_seco({"unit": "frasco", "grams": 540, "label": "Cocidas Extra 540 gr · La Cochura", "price": 175})
    assert frasco["grams"] == round(540 * f, 1)


def test_lo_seco_se_mide_por_su_etiqueta():
    funda = {"unit": "funda", "grams": 800, "label": "Secos 800 gr · Wala", "price": 123}
    assert el.envase_en_seco(funda) is funda                      # ya estaba en seco: ni se copia
    assert el.envase_en_seco({"unit": "paquete", "grams": 2000, "label": "800 g seco", "price": 129})["grams"] == 800
    assert round(el.envase_en_seco({"unit": "paquete", "grams": 1135, "label": "1 lb seco", "price": 70})["grams"]) == 454
    giselle = {"unit": "funda", "grams": 400, "label": "400 gr · Giselle", "price": 65}
    assert el.envase_en_seco(giselle) is giselle


def test_solo_legumbres_en_base_seca():
    atun = {"name": "Atún en lata", "kcal_per_100g": 116, "market_packages": [{"unit": "lata", "grams": 142, "label": "5 oz", "price": 60}]}
    assert el.en_base_del_catalogo("Atún en lata", atun) is atun
    horneados = {"name": "Frijoles horneados", "kcal_per_100g": 94, "market_packages": [{"unit": "lata", "grams": 454, "label": "16 oz", "price": 95}]}
    assert el.en_base_del_catalogo("Frijoles horneados", horneados) is horneados   # base cocida: la lata ES la base
    sin_kcal = {"name": "Garbanzos", "market_packages": [{"unit": "lata", "grams": 425, "label": "15 oz", "price": 83}]}
    assert el.en_base_del_catalogo("Garbanzos", sin_kcal) is sin_kcal              # sin dato, no se inventa


def test_el_master_del_cache_no_se_muta():
    copia = el.en_base_del_catalogo("Habichuelas negras", _HABICHUELAS_NEGRAS)
    assert copia is not _HABICHUELAS_NEGRAS
    assert _HABICHUELAS_NEGRAS["market_packages"][0]["grams"] == 425
    assert _HABICHUELAS_NEGRAS["market_packages"][1]["grams"] == 2000
    assert [p["grams"] for p in copia["market_packages"]] == [round(425 * el.factor_listo_a_seco(), 1), 800]


def test_la_lista_ya_no_compra_una_lata_para_450_g_secos():
    """El caso de la batería (mes sin pescado): 40 g secos por cena × el ciclo = 450 g. Antes «1 lata (15 oz)» —
    ~90 g de legumbre—; ahora la funda de 800 g secos, que cuesta menos que las 5-6 latas que harían falta."""
    obj = sc.apply_smart_market_units("Habichuelas negras", 450 / 453.592, "lb", 0.0, dict(_HABICHUELAS_NEGRAS), cycle_days=30)
    assert obj["market_unit"] == "paquete" and obj["market_qty"] == 1, obj
    assert "800 g seco" in obj["display_qty"], obj


def test_si_solo_hay_latas_se_compran_las_que_hacen_falta():
    solo_latas = dict(_HABICHUELAS_NEGRAS, market_packages=[{"unit": "lata", "grams": 425, "label": "15 oz", "price": 88}])
    obj = sc.apply_smart_market_units("Habichuelas negras", 450 / 453.592, "lb", 0.0, solo_latas, cycle_days=30)
    assert obj["market_unit"] == "lata" and obj["market_qty"] >= 5, obj          # 450 / ~89 g secos por lata
    assert "15 oz" in obj["display_qty"], obj                                    # la etiqueta del estante no cambia


def test_el_gancho_esta_en_la_entrada_del_selector():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    i = src.index("def apply_smart_market_units(")
    cuerpo = src[i:i + 4000]
    assert 'master_item = __import__("envase_legumbre").en_base_del_catalogo(name, master_item)' in cuerpo
    assert cuerpo.index("en_base_del_catalogo") < cuerpo.index('cat = (master_item.get("category") or "").lower()')
    assert "tooltip-anchor: P1-PLAN-LOTE-282-ENVASE-EN-SECO" in (_BACKEND / "envase_legumbre.py").read_text(encoding="utf-8")
