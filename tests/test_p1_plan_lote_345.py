# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-345 · 2026-09-26] Una sustitución no reescribe las notas-plantilla.

Batería real sobre el 331: «⚠️ Sodio (hipertensión/riñón): … enjuaga los enlatados (yogurt griego sin azúcar, granos)»
(adulto mayor con HTA) y «🌱 … esta receta usa solo pechuga de pollo — NO botes pechuga de pollo: guárdalas tapadas»
(suplementos + estatina): el reescritor de pasos tras una sustitución cambiaba también el texto de las notas."""
from __future__ import annotations

import graph_orchestrator as go

_SODIO = ("⚠️ Sodio (hipertensión/riñón): elige versiones bajas en sodio y enjuaga los enlatados (atún, granos) antes de "
          "usarlos; el queso, mejor fresco y bajo en sal.")
_CLARAS = ("🌱 Nota del Nutricionista AI: esta receta usa solo las claras — NO botes las yemas: guárdalas tapadas en la "
           "nevera (2-3 días) y úsalas en otra comida.")


def test_la_nota_de_sodio_se_queda_y_la_del_nutricionista_obsoleta_sale():
    meal = {"recipe": ["Mise en place: bate 6 claras de huevo y escurre el atún.", _CLARAS,
                       "Montaje: sirve el atún con las claras.", _SODIO]}
    assert go._rewrite_recipe_steps_after_subs(meal, [(["atún", "atun"], "Yogurt griego sin azúcar"),
                                                      (["claras de huevo", "claras"], "Pechuga de pollo")])
    assert _SODIO in meal["recipe"], meal["recipe"]
    assert not any(str(s).startswith("🌱") for s in meal["recipe"]), meal["recipe"]
    assert "yogurt griego sin azúcar" in meal["recipe"][0].lower()


def test_la_nota_del_nutricionista_de_otro_alimento_se_queda():
    nota = "🌱 Nota del Nutricionista AI: espolvorea semillas de girasol sobre el plato al servir."
    meal = {"recipe": ["Mise en place: escurre el atún.", nota, "Montaje: sirve."]}
    go._rewrite_recipe_steps_after_subs(meal, [(["atún", "atun"], "Sardinas")])
    assert nota in meal["recipe"]
