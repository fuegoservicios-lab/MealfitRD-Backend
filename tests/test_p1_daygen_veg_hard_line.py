# -*- coding: utf-8 -*-
"""[P1-DAYGEN-VEG-HARD-LINE · 2026-09-05] En cuatro planes vegetarianos seguidos (606e9017, 82d6f2a5, b40a3c48…) el
generador metió «pechuga de pollo»: la dieta viajaba en el pool y en un bloque de DIVERSIDAD, nunca como prohibición en el
prompt del día; el guard duro la cazaba DESPUÉS y quemaba un reintento completo (~90 s). Y el tope de huevos añadía su
clara sin fusionarla con la línea de claras existente («3 huevos + 3 claras + 1 clara»)."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import prompts.day_generator as dg  # noqa: E402

SKEL = {"assigned_technique": "Guiso", "protein_pool": ["Lentejas", "Huevo"], "carb_pool": ["Papa", "Arroz"]}


def test_hard_diet_line_per_diet():
    veg = dg.build_day_assignment_context(SKEL, 1, diet_type="vegetariana")
    assert "DIETA VEGETARIANA — PROHIBICIÓN ABSOLUTA" in veg and "CERO carne, pollo" in veg
    assert "Huevo, lácteos y legumbres SÍ" in veg
    vegan = dg.build_day_assignment_context(SKEL, 1, diet_type="vegana")
    assert "DIETA VEGANA — PROHIBICIÓN ABSOLUTA" in vegan and "huevo, lácteos" in vegan
    pesc = dg.build_day_assignment_context(SKEL, 1, diet_type="pescetariana")
    assert "DIETA PESCETARIANA" in pesc and "Pescado, mariscos, huevo" in pesc
    # sin dieta declarada: byte-idéntico al prompt de siempre (sin línea)
    assert "PROHIBICIÓN ABSOLUTA" not in dg.build_day_assignment_context(SKEL, 1)
    assert "PROHIBICIÓN ABSOLUTA" not in dg.build_day_assignment_context(SKEL, 1, diet_type="balanceada")


def test_hard_line_is_at_the_top_of_the_assignment():
    ctx = dg.build_day_assignment_context(SKEL, 1, diet_type="vegetariana")
    i_hdr = ctx.index("ASIGNACIÓN DEL PLANIFICADOR")
    i_line = ctx.index("PROHIBICIÓN ABSOLUTA")
    i_pool = ctx.index("Proteínas Asignadas")
    assert i_hdr < i_line < i_pool, "la prohibición va arriba, antes de los pools"


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def test_egg_cap_merges_with_existing_whites(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Tortilla española exprés",
                                  "ingredients": ["4 huevos", "3 claras de huevo", "1 papa"],
                                  "ingredients_raw": ["4 huevos", "3 claras de huevo", "1 papa"], "recipe": []}]}]
    assert go._cap_daily_whole_eggs(days, db=_NoopDB(), max_whole=3) == 1
    ings = days[0]["meals"][0]["ingredients"]
    assert ings[0] == "3 huevos" and "4 claras de huevo" in ings
    assert sum(1 for i in ings if "clara" in i) == 1, ings
    assert days[0]["meals"][0]["ingredients_raw"] == ings


def test_egg_cap_without_existing_whites_still_appends(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Tortilla", "ingredients": ["5 huevos"],
                                  "ingredients_raw": ["5 huevos"], "recipe": []}]}]
    go._cap_daily_whole_eggs(days, db=_NoopDB(), max_whole=3)
    assert days[0]["meals"][0]["ingredients"] == ["3 huevos", "2 claras de huevo"]
