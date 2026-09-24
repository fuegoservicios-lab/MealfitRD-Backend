# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-190 · 2026-09-24] Batería rd18 (código 189, alergia a lácteos y mariscos): plan real, cero críticos, pero
un intento quemado por una merienda que ESCRIBIMOS nosotros: «Aguacate con maní, semillas de calabaza, habichuelas rojas
guisadas a la plancha y Habichuelas rojas guisadas».

`_protein_repeat_autofix`, en una comida LIGERA (merienda/desayuno), sólo podía elegir queso o legumbres; al alérgico a
lácteos (y al vegano) le quedaban las legumbres, cuya forma es «habichuelas rojas guisadas» / «lentejas guisadas». La
regla de horario las rechaza SIEMPRE: en la merienda por «guisada», en el desayuno por «habichuela/lenteja». O sea, el
autofix arreglaba la variedad rompiendo el horario — y como la regla de horario es blanda en el último intento, ese
nombre podía llegar al usuario. Sin candidato limpio, el autofix ya no toca la comida ligera: decide la puerta de
variedad (que en el último intento es aviso, y una proteína repetida es un defecto mucho menor)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _meal(name, ingredients, slot):
    return {"name": name, "meal": slot, "ingredients": list(ingredients),
            "ingredients_raw": list(ingredients), "recipe": [f"Prepara {name.lower()}."]}


def _dia(merienda_slot="Merienda"):
    almuerzo = _meal("Pollo guisado con auyama", ["150 g de pechuga de pollo", "200 g de auyama"], "Almuerzo")
    ligera = _meal("Aguacate con maní y pollo a la plancha", ["60 g de aguacate", "15 g de maní",
                                                             "80 g de pechuga de pollo"], merienda_slot)
    cena = _meal("Pescado al horno con yuca", ["150 g de filete de pescado blanco", "150 g de yuca"], "Cena")
    return {"day": 1, "meals": [almuerzo, ligera, cena]}, ligera


def _viola_horario(meal):
    from constants import slot_violations_for_meal_name, SLOT_INAPPROPRIATE_FOODS
    return slot_violations_for_meal_name(meal["name"], meal["meal"].lower(), rules_table=SLOT_INAPPROPRIATE_FOODS)


def test_alergico_a_lacteos_no_recibe_legumbre_guisada_en_la_merienda():
    import graph_orchestrator as go
    dia, merienda = _dia()
    go._protein_repeat_autofix([dia], {"allergies": ["Lacteos", "Mariscos"], "mainGoal": "lose_fat"})
    blob = (merienda["name"] + " " + " ".join(merienda["ingredients"])).lower()
    assert "guisad" not in blob and "habichuela" not in blob and "lenteja" not in blob, merienda
    assert not _viola_horario(merienda), merienda["name"]


def test_desayuno_tampoco():
    import graph_orchestrator as go
    dia, desayuno = _dia("Desayuno")
    go._protein_repeat_autofix([dia], {"allergies": ["Lacteos"], "mainGoal": "lose_fat"})
    assert "habichuela" not in desayuno["name"].lower() and "lenteja" not in desayuno["name"].lower()
    assert not _viola_horario(desayuno), desayuno["name"]


def test_sin_alergia_la_merienda_sigue_reparandose_con_queso():
    import graph_orchestrator as go
    dia, merienda = _dia()
    assert go._protein_repeat_autofix([dia], {"mainGoal": "lose_fat"}) >= 1
    assert "queso" in (merienda["name"] + " " + " ".join(merienda["ingredients"])).lower(), merienda


def test_las_comidas_principales_conservan_el_respaldo_de_legumbres():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '_PROTEIN_REPEAT_NONGATED_FALLBACK = ("habichuelas", "lentejas", "queso")' in src
    assert '_allowed = ("queso",)  # [P1-PLAN-LOTE-190]' in src


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 190 and m.group(2) >= "2026-09-24"
