# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-588 · 2026-09-27] Una sustitución no reescribe las notas clínicas ⚕️.

Corpus de baterías reales: la nota «⚕️ Alergia a mariscos: el pescado de aleta (tilapia, mero, atún, sardina…) no es un
marisco» salió 5 de 55 veces con el ejemplo cambiado — «(tilapia, mero, pechuga de pavo, sardina…)»: la pechuga de pavo
como pescado de aleta. El reescritor de pasos tras una sustitución cambiaba el alimento también dentro de la nota.
"""
from __future__ import annotations

import graph_orchestrator as go

_MARISCOS = ("⚕️ Alergia a mariscos: el pescado de aleta (tilapia, mero, atún, sardina…) no es un marisco y tu perfil no "
             "lo excluye. Si también te hace reacción, márcalo en tus alergias y lo quitamos.")
_AJUSTE = ("⚕️ Ajuste clínico (condición médica): se sustituyó atún en aceite por una alternativa segura (sin azúcar "
           "añadida / baja en sodio) para tu condición.")


def test_la_nota_clinica_no_cambia_de_alimento():
    meal = {"recipe": ["Mise en place: escurre el atún y corta la cebolla.", "Montaje: sirve el atún con la cebolla.",
                       _MARISCOS, _AJUSTE]}
    assert go._rewrite_recipe_steps_after_subs(meal, [(["atún", "atun"], "Pechuga de pavo")])
    assert "pechuga de pavo" in meal["recipe"][0].lower(), meal["recipe"][0]
    assert _MARISCOS in meal["recipe"] and _AJUSTE in meal["recipe"], meal["recipe"]


def test_las_demas_plantillas_siguen_igual():
    import pasos_cantidades as pc
    assert pc.nota_plantilla("🌱 Nota del Nutricionista AI: espolvorea semillas.")
    assert pc.nota_plantilla(_MARISCOS)
    assert not pc.nota_plantilla("El Toque de Fuego: escurre el atún.")
