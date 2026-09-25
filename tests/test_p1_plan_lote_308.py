# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-308 · 2026-09-25] El sincronizador exacto, también como primer paso del contrato final.

El contrato final de receta mide gramos con la tolerancia del medidor V4 (±25 %): un cambio tardío de la lista (piso de
porción, techo, sustitución) de 50 → 60 g de mango o de 25 → 30 g de aguacate quedaba en el paso («pela y corta 50 g de
mango» con 60 g en la lista). Baterías del 25-sep: 4 comidas así entre 15 % y 25 %."""
from __future__ import annotations

import copy
import pathlib

import recipe_contract as rc
from culinary_coherence import build_culinary_index

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_CAT = [
    {"name": "Mango", "aliases": ["mangos"], "category": "Frutas", "prep_methods": ["crudo"]},
    {"name": "Yogurt griego", "aliases": ["yogurt griego natural", "yogur griego"], "category": "Lácteos",
     "prep_methods": ["crudo"]},
    {"name": "Aguacate", "aliases": ["aguacates"], "category": "Frutas", "prep_methods": ["crudo"]},
]
_INDEX = build_culinary_index(_CAT)


def _meal():
    ings = ["⅓ taza de yogurt griego natural sin azúcar", "60 g de mango", "2 cdtas de mantequilla de almendras"]
    return {"name": "Yogurt con mango", "meal": "Merienda", "ingredients": list(ings), "ingredients_raw": list(ings),
            "recipe": ["Mise en place: pela y corta 50 g de mango en cubos; ten a mano el yogurt bien frío.",
                       "Montaje: sirve el yogurt en una copa y corona con el mango."]}


def test_el_contrato_final_deja_el_paso_igual_a_la_lista():
    m = _meal()
    rc._aplicar_meal(m, _INDEX, "repair")
    assert "pela y corta 60 g de mango en cubos" in m["recipe"][0], m["recipe"][0]


def test_en_sombra_no_se_toca_nada():
    m = _meal()
    antes = copy.deepcopy(m)
    rc._aplicar_meal(m, _INDEX, "shadow")
    assert m["recipe"] == antes["recipe"] and m["ingredients"] == antes["ingredients"]


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").sincronizar_exacto(meal)   # [P1-PLAN-LOTE-308]' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-308" in (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
