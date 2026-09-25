# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-310 · 2026-09-25] El peso entre paréntesis de un paso sigue al de la MISMA línea de la lista.

Baterías del 25-sep: «lava y corta ¾ manzana (≈120 g)» con «¾ manzana (≈65 g)» en la lista; «pela ½ plátano maduro grande
(200 g)» con «(101 g)»; «corta ½ pechuga de pollo (porción) (120 g)» con «(≈139 g)». El sincronizador de pesos sólo veía
menciones con unidad (taza, cda, g…), no piezas."""
from __future__ import annotations

import copy
import pathlib

import pasos_cantidades as pc
import recipe_contract as rc
from culinary_coherence import build_culinary_index

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _meal(ings, rec):
    return {"name": "Plato", "meal": "Cena", "ingredients": list(ings), "ingredients_raw": list(ings), "recipe": list(rec)}


def test_la_pieza_del_paso_toma_el_peso_de_su_linea():
    m = _meal(["¾ manzana (≈65 g)", "1 cda de mantequilla de maní (16 g)"],
              ["Mise en place: lava y corta ¾ manzana (≈120 g) en 8 gajos dejándole la cáscara."])
    assert pc.pesos_de_la_lista(m) == 1
    assert m["recipe"][0] == "Mise en place: lava y corta ¾ manzana (≈65 g) en 8 gajos dejándole la cáscara."
    p = _meal(["½ plátano maduro grande (101 g)", "30 g de queso blanco fresco"],
              ["Mise en place: pela ½ plátano maduro grande (200 g) y córtalo en rodajas de 1 cm."])
    pc.pesos_de_la_lista(p)
    assert "pela ½ plátano maduro grande (101 g) y córtalo" in p["recipe"][0], p["recipe"][0]


def test_la_porcion_que_la_lista_ya_respondio():
    m = _meal(["1¼ pechuga de pollo (≈224 g)", "85 g de harina de maíz precocida"],
              ["Mise en place: Mide 1¼ pechugas de pollo (porción) (190 g), pica 1 diente de ajo.",
               "Montaje: sirve ½ pechuga de pollo (porción) con las arepitas."])
    pc.pesos_de_la_lista(m)
    assert m["recipe"][0] == "Mise en place: Mide 1¼ pechugas de pollo (≈224 g), pica 1 diente de ajo.", m["recipe"][0]
    assert m["recipe"][1] == "Montaje: sirve ½ pechuga de pollo (porción) con las arepitas.", "otra cantidad: no es esa línea"


def test_lo_que_coincide_o_no_es_la_misma_linea_no_se_toca():
    m = _meal(["¾ manzana (≈65 g)", "2 mandarinas (80 g)"],
              ["lava y corta ¾ manzana (≈66 g).", "pela 1 mandarina (40 g).", "⚕️ Nota clínica: ¾ manzana (≈120 g) basta."])
    antes = copy.deepcopy(m)
    assert pc.pesos_de_la_lista(m) == 0 and m == antes
    assert pc.pesos_de_la_lista({"ingredients": None, "recipe": "x"}) == 0


def test_corre_en_el_contrato_final():
    m = _meal(["¾ manzana (≈65 g)"], ["Mise en place: lava y corta ¾ manzana (≈120 g)."])
    rc._aplicar_meal(m, build_culinary_index([{"name": "Manzana", "aliases": ["manzanas"], "category": "Frutas"}]),
                     "repair")
    assert m["recipe"][0] == "Mise en place: lava y corta ¾ manzana (≈65 g).", m["recipe"][0]


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").pesos_de_la_lista(meal)    # [P1-PLAN-LOTE-310]' in src
