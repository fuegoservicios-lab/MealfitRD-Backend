"""[P1-1-QTYSYNC-STALE-EN-TOTAL · 2026-07-10] (recipe plausibility roadmap, item P1-1) Evidencia visual
(plan 564d6e4e): paso "Unta cada tortilla con 2 cdas de mantequilla de maní (en total)." con línea de
ingrediente "1.25 cdas de mantequilla de maní (jif)". `_sync_recipe_step_quantities` YA corre en TODAS
las superficies (assemble tail + expand endpoint, P1-RECIPE-QTY-SYNC/P1-EXPAND-QTY-SYNC) — pero su pase
CADA-TOTAL (P2-QTYSYNC-CADA-TOTAL) se ABSTIENE incondicionalmente cuando el paso YA contiene la
substring "(en total" (asume que un pase previo la anotó correctamente). El LLM (day_generator/Chef)
puede escribir "(en total)" DIRECTAMENTE con un número INVENTADO nunca validado contra la línea real —
el guard existente trata ese texto como "ya sincronizado" y lo deja intacto.

Fix: pase adicional (NO reemplaza el guard existente — es aditivo, corre DESPUÉS) que SÍ entra a los
pasos con "(en total" preexistente, valida el número contra `food_total_f` y corrige SOLO si diverge,
preservando la anotación "(en total)". Reusa el parsing YA EXISTENTE (`_STEP_QTY_MENTION_RE`,
`food_total_f`, `_qtysync_qty_to_float`/`_qtysync_unit_norm`) — cero regex nueva de cantidades.
"""
from __future__ import annotations

import graph_orchestrator as g


def _meal(ingredients, recipe):
    return {"name": "Tostadas con maní", "ingredients": ingredients, "recipe": recipe}


def test_marker_present():
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(os.path.dirname(_here), "graph_orchestrator.py"), encoding="utf-8") as f:
        assert "P1-1-QTYSYNC-STALE-EN-TOTAL" in f.read()


def test_corrects_stale_number_preserving_en_total_annotation():
    meal = _meal(
        ["1.25 cdas de mantequilla de maní (jif)", "2 tortillas de trigo"],
        ["Precalienta la sartén.",
         "Unta cada tortilla con 2 cdas de mantequilla de maní (en total). Rocía con miel al gusto.",
         "Sirve caliente."],
    )
    n = g._sync_recipe_step_quantities(meal)
    assert n >= 1
    step = meal["recipe"][1]
    assert "1.25 cdas de mantequilla de maní (en total)" in step
    assert "2 cdas de mantequilla de maní (en total)" not in step


def test_noop_when_en_total_already_correct():
    meal = _meal(
        ["1.25 cdas de mantequilla de maní (jif)", "2 tortillas de trigo"],
        ["Precalienta la sartén.",
         "Unta cada tortilla con 1.25 cdas de mantequilla de maní (en total). Rocía con miel al gusto.",
         "Sirve caliente."],
    )
    before = list(meal["recipe"])
    g._sync_recipe_step_quantities(meal)
    assert meal["recipe"] == before


def test_preexisting_cada_total_pass_still_covers_bare_mismatch():
    """SIN '(en total' preexistente, el pase CADA-TOTAL YA EXISTENTE (P2-QTYSYNC-CADA-TOTAL) ya
    corrige y anota correctamente — este test ancla que ese comportamiento previo NO regresiona con
    el pase nuevo (que solo cubre el caso complementario: anotación preexistente con número stale)."""
    meal = _meal(
        ["1.25 cdas de mantequilla de maní (jif)", "2 tortillas de trigo"],
        ["Precalienta la sartén.",
         "Unta cada tortilla con 2 cdas de mantequilla de maní. Rocía con miel al gusto.",
         "Sirve caliente."],
    )
    g._sync_recipe_step_quantities(meal)
    assert "1.25 cdas de mantequilla de maní (en total)" in meal["recipe"][1]
