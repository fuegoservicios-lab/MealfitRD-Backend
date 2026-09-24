# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-219 · 2026-09-24] Techo servible del maíz dulce: una lata escurrida (250 g) por comida.

Batería real del 24-sep (perfil sin tiempo para cocinar, 30 días): «365 g de maíz dulce en granos» en una cena, «370 g»
en otra, «310 g» en un almuerzo; en las baterías guardadas 10 de 31 líneas pasaban de 250 g (hasta 415 g). El solver lo
infla como base de carbohidrato de los menús sin cocción y ningún techo lo veía: no es un vegetal acuoso.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def _dia(lineas):
    return [{"day": 1, "meals": [{"meal": "Cena", "name": "Bowl de maíz y pollo", "ingredients": list(lineas),
                                  "ingredients_raw": list(lineas)}]}]


def _gramos(linea):
    m = re.match(r"^\s*([\d.,]+)\s*g\b", linea)
    return float(m.group(1).replace(",", ".")) if m else None


def test_el_maiz_dulce_no_pasa_de_una_lata_por_comida(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _dia(["370 g de maíz dulce en granos", "135 g de pechuga de pollo"])
    go._cap_unrealistic_portions(days, db=_NoopDB())
    maiz = next(x for x in days[0]["meals"][0]["ingredients"] if "maíz" in x)
    assert _gramos(maiz) is not None and _gramos(maiz) <= go.REALISM_SWEET_CORN_CAP_G, maiz


def test_lo_que_no_es_maiz_dulce_o_ya_cabe_no_se_toca(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    lineas = ["90 g de maíz dulce en granos", "300 g de harina de maíz precocida"]
    days = _dia(lineas)
    go._cap_unrealistic_portions(days, db=_NoopDB())
    assert days[0]["meals"][0]["ingredients"][0] == "90 g de maíz dulce en granos"
    assert "harina de maíz" in days[0]["meals"][0]["ingredients"][1]


def test_knob_y_ancla():
    assert go.REALISM_SWEET_CORN_CAP_G == 250
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'MEALFIT_REALISM_SWEET_CORN_CAP_G", 250' in src and "[P1-PLAN-LOTE-219] 1.65)" in src


def test_la_auyama_entra_al_techo_de_volumen(monkeypatch):
    """«460 g de auyama en cubos» y «470 g» en dos comidas del mismo día (batería del 24-sep): víver acuoso (30,9 kcal)
    que el techo de vegetales de volumen no veía por su categoría."""
    import shopping_calculator as sc
    filas = [{"name": "Auyama", "category": "Víveres", "kcal_per_100g": 30.9, "aliases": ["zapallo"]},
             {"name": "Batata", "category": "Víveres", "kcal_per_100g": 86.0, "aliases": []}]
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: filas)
    monkeypatch.setattr(go, "_WATERY_VEG_TOKENS_CACHE", None)
    monkeypatch.setattr(go, "_catalog_index_should_rebuild", lambda *a, **k: True)
    toks = go._watery_veg_tokens()
    assert "auyama" in toks and "zapallo" in toks and "batata" not in toks


def test_las_dos_migajas_de_la_bateria():
    import pulido_lineas as pl
    assert pl.pulir_linea("⅔ g de semillas de girasol") == "1 pizca de semillas de girasol"
    assert pl.pulir_linea("1.53 g de yogurt natural sin azúcar") == "1 cdta de yogurt natural sin azúcar"
    for igual in ("1 cdta de yogurt natural sin azúcar", "1.5 g de aguacate", "0.4 g de aguacate", "⅔ taza de yogurt"):
        assert pl.pulir_linea(igual) == igual, igual

