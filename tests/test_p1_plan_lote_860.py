# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-860 · 2026-09-29] El agua no cuenta como el alimento que nombra, y el total fundido de algo cocido sigue
cocido.

Batería real DO de 6d (29-sep), «Wrap integral de habichuelas rojas majadas…» (D2 almuerzo): la línea «70 g de Agua para
calentar las habichuelas» resolvía a Habichuelas rojas (SECAS), el pase de legumbre a lata (545) la convirtió en «190 g
de agua para calentar las habichuelas cocidos (de lata…)», la fusión de duplicados la sumó a las habichuelas cocidas y
escribió «315 g de Habichuelas rojas» — en seco. Declarado 1.013 kcal; lo que se come, ~500: el día real −25 %.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import formas_de_base as fb
import legumbre_lista as ll
from shopping_calculator import resolve_preparation_distinct as rpd

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_agua_con_proposito_es_agua():
    for s in ("Agua para calentar las habichuelas", "70 g de Agua para calentar las habichuelas",
              "1 taza de agua para cocinar el arroz", "½ L de agua para hervir la yautía",
              "40 g de agua para hidratar la harina", "agua para la harina de trigo", "agua de cocción",
              "Agua, cantidad necesaria para hervir el plátano verde"):
        assert rpd(s) == (True, None), s
    for s in ("Agua de coco", "aguacate", "Aguacate hass", "Agua de jamaica"):
        assert rpd(s) == (False, None), s


def test_nutrition_db_no_da_macros_de_habichuela_al_agua():
    from nutrition_db import IngredientNutritionDB
    rows = [{"name": "Habichuelas rojas", "aliases": ["habichuelas"], "kcal_per_100g": 344.7, "protein_g_per_100g": 22.5,
             "carbs_g_per_100g": 61.3, "fats_g_per_100g": 1.1},
            {"name": "Arroz blanco", "aliases": ["arroz"], "kcal_per_100g": 358.6, "protein_g_per_100g": 6.6,
             "carbs_g_per_100g": 80.3, "fats_g_per_100g": 0.6}]
    db = IngredientNutritionDB(rows=rows)
    assert db.lookup("Agua para calentar las habichuelas") is None
    assert db.lookup("agua según las instrucciones del arroz") is None
    assert db.lookup("habichuelas rojas").name == "Habichuelas rojas"


class _DbQueConfunde:
    """El catálogo de antes: cualquier texto con «habichuela» es la fila seca."""

    def lookup(self, nombre):
        return SimpleNamespace(name="Habichuelas rojas", kcal=333.0) if "habichuela" in str(nombre).lower() else None

    def grams_from_ingredient_string(self, linea):
        import re
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(linea))
        return float(m.group(1)) if m else None


def test_la_legumbre_a_lata_solo_convierte_la_legumbre():
    agua = "70 g de Agua para calentar las habichuelas"
    days = [{"meals": [{"ingredients": ["1 tortilla integral", "35 g de habichuelas rojas secas", agua],
                        "recipe": ["Mise en place: ten las habichuelas."]}]}]
    assert ll.a_lata(days, {"cookingTime": "30min", "batchCooking": "never"}, _DbQueConfunde()) == 1
    ings = days[0]["meals"][0]["ingredients"]
    assert ings[2] == agua, "el agua no se convierte en habichuelas de lata"
    assert ings[1].startswith("90 g de habichuelas rojas cocidas (de lata"), ings[1]


def test_el_total_fundido_de_lo_cocido_sigue_cocido():
    assert fb.con_forma("315 g de Habichuelas rojas", "cocido") == "315 g de Habichuelas rojas cocidas"
    assert fb.con_forma("1½ tazas de Arroz blanco", "cocido") == "1½ tazas de Arroz blanco cocido"
    assert fb.con_forma("200 g de Garbanzos", "cocido") == "200 g de Garbanzos cocidos"
    assert fb.con_forma("120 g de Lentejas", "cocido") == "120 g de Lentejas cocidas"
    assert fb.con_forma("315 g de Habichuelas rojas", "") == "315 g de Habichuelas rojas", "lo seco no cambia"
    assert fb.con_forma("140 g de Huevo", "") == "140 g de Huevo"
    assert fb.con_forma("200 g de garbanzos cocidos", "cocido") == "200 g de garbanzos cocidos", "idempotente"


def test_la_fusion_usa_la_forma():
    import graph_orchestrator as go
    days = [{"day": 2, "meals": [{"name": "Wrap", "ingredients": ["125 g de habichuelas rojas cocidas",
                                                                  "90 g de habichuelas rojas cocidas"]}]}]
    tele = go._merge_duplicate_food_lines(days)
    lineas = days[0]["meals"][0]["ingredients"]
    assert len(lineas) == 1 and tele, (lineas, tele)
    assert "cocid" in lineas[0].lower(), lineas
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("formas_de_base").con_forma(_dup_merge_format(total, dom[2], dom[3]), _k[1])' in src
