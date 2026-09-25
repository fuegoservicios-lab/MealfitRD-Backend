# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-229 · 2026-09-25] Una nota clínica (⚕️) no es un paso de cocción.

Batería real (perfil hipotiroidismo + levotiroxina): la nota «separa estos alimentos … al menos 4 horas de la dosis» salió
como «al menos 10-12 min a fuego medio de la dosis» / «2-3 min por lado» / «18-20 min a 180 °C» en 7 de 12 comidas — el
clamp de tiempo/temperatura la trató como una cocción de 240 min. Consejo médico falso entregado.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

NOTA = ("⚕️ Nota clínica: toma la levotiroxina en ayunas y separa estos alimentos (lácteos/soya/linaza/espinacas/café/"
        "toronja) al menos 4 horas de la dosis — interfieren su absorción.")


def test_la_nota_clinica_es_nota():
    assert go._is_recipe_safety_note_step(NOTA)
    assert go._is_recipe_safety_note_step("⚕️ Alergia a mariscos: el pescado de aleta no es un marisco.")
    assert not go._is_recipe_safety_note_step("El Toque de Fuego: hierve la yuca 15-18 min.")


def test_el_clamp_no_reescribe_las_horas_de_la_nota():
    meal = {"name": "Lechosa con maní y yogurt", "recipe": [
        "Mise en place: corta la lechosa en cubos.", "Montaje: sirve con el maní por encima.", NOTA]}
    go._clamp_recipe_time_temp_outliers(meal)
    assert meal["recipe"][-1] == NOTA, meal["recipe"]
    # y un paso de cocción de verdad con 240 min SÍ se sigue clampando
    meal2 = {"name": "Yuca hervida", "recipe": ["El Toque de Fuego: hierve la yuca 240 min a fuego medio."]}
    assert go._clamp_recipe_time_temp_outliers(meal2)
    assert "240" not in meal2["recipe"][0]


def test_la_nota_que_escribe_la_capa_clinica_llega_intacta():
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Merienda", "name": "Yogurt natural con lechosa y maní",
        "ingredients": ["150 g de yogurt natural", "100 g de lechosa", "15 g de maní"],
        "recipe": ["Mise en place: corta la lechosa.", "Montaje: sirve el yogurt con la lechosa y el maní."]}]}]}
    go._apply_condition_safety_annotations(plan, {"medicalConditions": ["Hipotiroidismo"],
                                                  "medications": ["Levotiroxina"]})
    meal = plan["days"][0]["meals"][0]
    notas = [s for s in meal["recipe"] if "levotiroxina" in s]
    assert notas and "4 horas" in notas[0], meal["recipe"]
    go._clamp_recipe_time_temp_outliers(meal)
    notas = [s for s in meal["recipe"] if "levotiroxina" in s]
    assert notas and "4 horas" in notas[0] and "fuego" not in notas[0], meal["recipe"]


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-229-NOTA-CLINICA" in src


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 229
