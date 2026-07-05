"""[P1-CURED-GHOST-STEPS + P2-STEP-FRUIT-GHOST + P2-NOCOOK-TDF-APLICA · 2026-07-05]

Review visual #5 (plan 55846e5e):
- Pasos guisando ARENQUE que NO está en la lista ("Desala el arenque… Incorpora el arenque
  desmenuzado" — la línea curada fue removida por otro pase y sus pasos sobrevivieron). Dirección
  INVERSA al ghost de frutas/nueces: materializar un curado regalaría el sodio de vuelta → se
  REESCRIBEN las menciones hacia la proteína presente + strip de frases de desalado.
- "Desala el filete de pescado blanco remojándolo 2 horas" — el swap curado→fresco del sodio
  reemplazaba el nombre pero dejaba la instrucción de desalado (nadie desala pescado fresco).
- "Lava las uvas… Distribuye las uvas" con CERO uvas en la lista → fruit-ghost (materializa).
- "El Toque de Fuego: No aplica (preparación en frío) (~10-12 min a fuego medio)" — la variante
  "No aplica" no estaba en el regex del placeholder y el backstop inyectaba tiempo falso.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


# ───────────────────── cured ghost → rewrite ─────────────────────

def test_arenque_ghost_rewritten_to_present_protein(go):
    days = [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Bollitos de Plátano con Arenque Guisado",
        "ingredients": ["½ plátanos verdes", "1 tomate", "50g de sardinas en lata"],
        "ingredients_raw": ["½ plátanos verdes", "1 tomate", "50g de sardinas en lata"],
        "recipe": [
            "Mise en place: Desala el arenque remojándolo en agua fría por 2 horas (cambiando el agua cada 30 minutos). Escurre y desmenuza. Pela los plátanos verdes.",
            "El Toque de Fuego: incorpora el arenque desmenuzado y cocina 5 minutos a fuego bajo.",
            "Montaje: sirve los bollitos.",
        ],
    }]}]
    assert go._rewrite_cured_ghost_protein_steps(days) == 1
    m = days[0]["meals"][0]
    blob = " ".join(str(s) for s in m["recipe"]).lower()
    assert "arenque" not in blob, "el fantasma se reescribe hacia la proteína PRESENTE"
    assert "sardinas" in blob
    assert "desala" not in blob and "remoj" not in blob, "el desalado desaparece"
    assert str(m["recipe"][0]).lower().startswith("mise en place"), "el prefijo del pilar se preserva"
    assert "arenque" not in str(m["name"]).lower(), f"nombre coherente: {m['name']}"
    assert m.get("_cured_ghost_rewritten") == "arenque"


def test_cured_in_list_untouched(go):
    """Si el curado SÍ está en la lista (usuario lo pidió/planner lo asignó), no es fantasma."""
    days = [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Bacalao Guisado",
        "ingredients": ["150 g de bacalao", "150 g de yuca"],
        "ingredients_raw": ["150 g de bacalao", "150 g de yuca"],
        "recipe": ["Mise en place: desala el bacalao 2 horas.",
                   "El Toque de Fuego: guisa 15 min.", "Montaje: sirve."],
    }]}]
    assert go._rewrite_cured_ghost_protein_steps(days) == 0
    assert "desala" in str(days[0]["meals"][0]["recipe"][0]).lower(), \
        "bacalao real → el desalado es la técnica correcta, intacta"


def test_desalt_stripped_on_sodium_saltcured_swap():
    i = _GO.index('"swap_saltcured" if _is_saltcured_swap else "swap_canned"')
    win = _GO[i:i + 700]
    assert "_strip_desalt_instructions(_bm)" in win, \
        "el swap curado→fresco del sodio también limpia el desalado"


# ───────────────────── fruit ghost ─────────────────────

def test_uvas_ghost_materialized(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_STEP_CARB_GUARD_ENABLED", True)
    monkeypatch.setattr(go, "RECIPE_STEP_FRUIT_GUARD_ENABLED", True)
    days = [{"day": 1, "meals": [{
        "meal": "Merienda", "name": "Yogurt con Uvas y Semillas",
        "ingredients": ["1¼ taza de yogurt natural descremado", "25 g de semillas de linaza"],
        "ingredients_raw": ["1¼ taza de yogurt natural descremado", "25 g de semillas de linaza"],
        "recipe": ["Mise en place: lava las uvas y córtalas por la mitad.",
                   "Montaje: distribuye las uvas por encima."],
    }]}]
    assert go._add_missing_recipe_step_carbs(days, db=None, allergies=None) >= 1
    assert any("uvas" in str(i).lower() for i in days[0]["meals"][0]["ingredients"]), \
        "las uvas de los pasos deben existir en la lista (si no, no se compran)"


# ───────────────────── 'No aplica' placeholder ─────────────────────

def test_no_aplica_variant_stripped(go):
    meal = {
        "name": "Yogurt Griego con Merey y Limón",
        "ingredients": ["1 taza de yogurt griego entero (201 g)", "5 g de merey tostado sin sal"],
        "recipe": ["Mise en place: exprime el jugo de medio limón. Pica el merey tostado.",
                   "El Toque de Fuego: No aplica (preparación en frío) (~10-12 min a fuego medio).",
                   "Montaje: coloca el yogurt, mezcla y sirve."],
    }
    assert go._inject_recipe_time_temp_defaults(meal) is True
    assert not any("toque de fuego" in str(s).lower() for s in meal["recipe"]), \
        "la variante 'No aplica' también se elimina en platos genuinamente fríos"
    assert go._recipe_step_contract_issues(meal) == []
