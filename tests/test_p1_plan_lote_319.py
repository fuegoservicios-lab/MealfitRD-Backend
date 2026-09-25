# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-319 · 2026-09-25] Una traza no es un ingrediente.

Baterías del 25-sep: «0.95 ml de leche descremada» (diabetes con insulina) y «Bate 1 clara de huevo con 0.07 ml de leche»
(familia de 4): el motor de macros deja un alimento en menos de 1 g/ml. Sale de la lista y el contrato final (V5, lote 30)
retira su mención de los pasos."""
from __future__ import annotations

import copy
import pathlib

import pasos_cantidades as pc
import recipe_contract as rc
from culinary_coherence import build_culinary_index

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_IDX = build_culinary_index([
    {"name": "Leche descremada", "aliases": ["leche"], "category": "Lácteos", "prep_methods": ["crudo"]},
    {"name": "Clara de huevo", "aliases": ["clara de huevo", "claras de huevo", "claras"], "category": "Proteínas",
     "prep_methods": ["cocido"]},
    {"name": "Canela", "aliases": ["canela en polvo"], "category": "Especias", "prep_methods": ["crudo"]},
    {"name": "Avena", "aliases": ["avena en hojuelas"], "category": "Granos", "prep_methods": ["cocido"]},
])


def _plato():
    ings = ["2 claras de huevo", "0.95 ml de leche descremada", "¼ cdta de canela en polvo", "30 g de avena"]
    return {"name": "Tortilla de claras con canela", "meal": "Desayuno", "ingredients": list(ings),
            "ingredients_raw": list(ings),
            "recipe": ["Mise en place: bate 2 claras de huevo con 0.95 ml de leche descremada y ¼ cdta de canela.",
                       "El Toque de Fuego: cuaja la tortilla en una sartén 3 min por lado.",
                       "Montaje: sirve con la avena."]}


def test_la_traza_sale_de_la_lista_y_de_los_pasos():
    m = _plato()
    rc._aplicar_meal(m, _IDX, "repair")
    assert "0.95 ml de leche descremada" not in m["ingredients"] and "0.95 ml de leche descremada" not in m["ingredients_raw"]
    assert "leche" not in m["recipe"][0].lower(), m["recipe"][0]
    assert m["_trazas_quitadas"] == ["0.95 ml de leche descremada"]


def test_lo_que_no_es_traza_o_es_su_dosis():
    m = {"ingredients": ["2 claras de huevo", "0.5 g de sal", "0.5 ml de aceite de oliva", "1 ml de leche", "30 g de avena"],
         "recipe": ["Bate las claras."]}
    antes = copy.deepcopy(m)
    assert pc.quitar_trazas(m) == 0 and m == antes
    casi_vacio = {"ingredients": ["0.5 ml de leche", "1 clara de huevo"], "recipe": ["Bate."]}
    assert pc.quitar_trazas(casi_vacio) == 0
    assert pc.quitar_trazas(None) == 0


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").quitar_trazas(meal)  # [P1-PLAN-LOTE-319]' in src
