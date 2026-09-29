# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-885 · 2026-09-29] La fruta fantasma no se materializa si la lista ya la trae en singular.

Baterías de 6d (ES D1 desayuno, PR D1 merienda): «395 g de uva» + «¼ taza de uvas (43g)» — el guard de frutas fantasma
(P2-STEP-FRUIT-GHOST) buscaba en la lista el token de su tabla, «uvas», y la lista decía «uva». Corpus: 15 comidas, todas
con la línea fija «¼ taza de uvas (43g)».
"""
from __future__ import annotations

import fantasma_presente as fp
import graph_orchestrator as go


def test_el_singular_de_la_lista_cuenta():
    assert fp.presente("uvas", "395 g de uva")
    assert fp.presente("nueces", "15 g de nuez picada")
    assert fp.presente("fresas", "100 g de fresa")
    assert not fp.presente("uvas", "30 g de avena ; 1 guineo")
    assert not fp.presente("kiwi", "1 kiwi"[:0]), "un token en singular no se toca"


def test_el_guard_no_duplica_la_uva():
    days = [{"day": 1, "meals": [{"name": "Tostada integral con huevo, tomate y uvas",
                                  "ingredients": ["1 rebanada de pan integral", "2 huevos", "395 g de uva"],
                                  "recipe": ["Mise en place: lava y desgrana las uvas.", "Montaje: sirve las uvas al lado."]}]}]
    go._add_missing_recipe_step_carbs(days)
    assert days[0]["meals"][0]["ingredients"] == ["1 rebanada de pan integral", "2 huevos", "395 g de uva"]
