# -*- coding: utf-8 -*-
"""[P1-SUBST-NAME-REWRITE · 2026-09-06] La sustitución cambiaba el ingrediente y dejaba la carne EN EL TÍTULO.

Medido en el plan vivo c5ba1681 (vegetariano, RD): la cena se llamaba «Pechuga de pollo…» con soya texturizada
dentro, el revisor médico la rechazó —con razón— y costó un INTENTO ENTERO de generación. Es el mismo hueco que
`P1-SUBST-ORPHAN-STEP` cerró para los pasos de la receta, un día antes, sin mirar el título.

Ya existía `_fix_phantom_protein_in_name`, pero **declina a propósito** cuando el reemplazo no es otra carne: su
caso es una incoherencia de generación, donde renombrar disimularía el fallo. Aquí es lo contrario — sabemos qué
se sustituyó y por qué, y dejar la carne en el título de un plan vegetariano es peor que un nombre desmañado."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import graph_orchestrator as go  # noqa: E402


def _plato(nombre, ingredientes):
    plan = {"days": [{"meals": [{"name": nombre, "ingredients": list(ingredientes),
                                 "ingredients_raw": list(ingredientes), "recipe": ["Cocina todo."]}]}]}
    go._apply_diet_substitutions(plan, {"dietType": "vegetariana"})
    return plan["days"][0]["meals"][0]


@pytest.mark.parametrize("nombre, ings, fuera", [
    ("Pechuga de pollo a la plancha con vegetales", ["150 g de pechuga de pollo", "Brócoli"], "pollo"),
    ("Guiso de res con auyama", ["180 g de carne de res", "Auyama"], "res"),
    ("Tilapia al horno con batata", ["160 g de tilapia", "Batata"], "tilapia"),
    ("Chuleta de cerdo con yuca", ["170 g de chuleta", "Yuca"], "chuleta"),
])
def test_la_carne_sale_tambien_del_titulo(nombre, ings, fuera):
    m = _plato(nombre, ings)
    assert fuera not in m["name"].lower(), f"«{nombre}» → «{m['name']}»"
    assert m["name"][0].isupper(), m["name"]


def test_el_titulo_nombra_lo_que_el_plato_lleva():
    m = _plato("Pechuga de pollo a la plancha con vegetales", ["150 g de pechuga de pollo", "Brócoli"])
    assert "soya texturizada" in m["name"].lower(), m["name"]
    assert "plancha" in m["name"].lower(), "el resto del título sobrevive"


@pytest.mark.parametrize("nombre, ings", [
    ("Queso fresco a la plancha con repollo", ["120 g de queso fresco", "Repollo"]),
    ("Ensalada de fresas con repollo morado", ["Fresas", "Repollo"]),
    ("Ensalada de garbanzos", ["Garbanzos", "Tomate"]),
])
def test_el_titulo_inocente_no_se_toca(nombre, ings):
    """El umbral del título baja a 3 caracteres para pescar «res»; la frontera de palabra es lo que lo hace
    seguro: «res» no casa dentro de «f-res-co» ni «pollo» dentro de «re-pollo»."""
    assert _plato(nombre, ings)["name"] == nombre


def test_es_idempotente():
    """Las sustituciones corren dos veces por plan: el título no puede acumular reemplazos."""
    ings = ["150 g de pechuga de pollo", "Brócoli"]
    plan = {"days": [{"meals": [{"name": "Pechuga de pollo a la plancha", "ingredients": list(ings),
                                 "ingredients_raw": list(ings), "recipe": ["Cocina todo."]}]}]}
    go._apply_diet_substitutions(plan, {"dietType": "vegetariana"})
    primero = plan["days"][0]["meals"][0]["name"]
    go._apply_diet_substitutions(plan, {"dietType": "vegetariana"})
    assert plan["days"][0]["meals"][0]["name"] == primero
    assert primero.lower().count("soya texturizada") == 1, primero


def test_tiene_su_propio_knob():
    """Reescribir el título es más visible que reescribir instrucciones: se revierte por separado."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'SUBST_NAME_REWRITE_ENABLED = _env_bool("MEALFIT_SUBST_NAME_REWRITE", True)' in src
    i = src.index("def _rewrite_meal_name_after_subs")
    assert "if not SUBST_NAME_REWRITE_ENABLED:" in src[i:i + 1800]


def test_el_titulo_se_reescribe_aunque_los_pasos_no(monkeypatch):
    """Dos knobs porque son dos riesgos: apagar el de los pasos no puede dejar la carne en el título."""
    monkeypatch.setattr(go, "SUBST_RECIPE_REWRITE_ENABLED", False)
    m = _plato("Pechuga de pollo a la plancha", ["150 g de pechuga de pollo"])
    assert "pollo" not in m["name"].lower(), m["name"]
