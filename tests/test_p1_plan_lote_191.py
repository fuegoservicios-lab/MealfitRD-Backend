# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-191 · 2026-09-24] Lo que un paso MIDE tiene que estar en la lista.

rd18: «mide 20 g de harina de Negrito… incorpora poco a poco la harina disuelta» y la lista sin harina. Medido sobre las
921 comidas guardadas: 11 comidas con un paso que mide un alimento ausente de la lista — dos harinas de Negrito, «30 g
de queso blanco fresco», «10 g de nueces», «30 g de pan rallado», «80 g de yogur» en un plato llamado «Yogur con…». La
guarda de alérgenos y el revisor leen la LISTA: un alérgeno que sólo está en un paso no lo veía nadie."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import lista_desde_pasos as lp  # noqa: E402

_GRAMOS = {"harina": 100.0, "queso": 100.0, "nueces": 100.0, "yogur": 100.0, "pan rallado": 100.0, "ajo": 100.0,
           "agua": 100.0}


class _DB:
    """Resuelve «N g/taza de X» si X empieza por un alimento conocido (el catálogo real exige red)."""

    def macros_from_ingredient_string(self, s):
        m = re.match(r"^\s*([\d.,½¼¾]+)\s*(g|taza|tazas|cda|cdas|cdta|cdtas|ml)\s+de\s+(.+)$", str(s), re.I)
        if not m:
            return None
        alimento = m.group(3).lower()
        if not any(alimento.startswith(k) for k in _GRAMOS):
            return None
        q = {"½": 0.5, "¼": 0.25, "¾": 0.75}.get(m.group(1)) or float(m.group(1).replace(",", "."))
        g = q * (120.0 if m.group(2).startswith("taza") else 1.0)
        return {"name": alimento, "grams": g, "kcal": g * 3.5, "protein": g * 0.1, "carbs": g * 0.7, "fats": g * 0.02}


def _cena(pasos, lista=None):
    lista = lista or ["2 filetes de pescado", "2 tomates medianos", "½ cebolla", "Sal al gusto"]
    return {"meal": "Cena", "name": "Pescado blanco guisado con tomate y harina de Negrito",
            "ingredients": list(lista), "ingredients_raw": list(lista), "recipe": list(pasos)}


def test_la_harina_del_paso_entra_a_la_lista():
    m = _cena(["Mise en place: corta el pescado; mide 20 g de harina de Negrito y pica el tomate.",
               "El Toque de Fuego: disuelve la harina en agua e incorpórala poco a poco."])
    assert lp.reconciliar_meal(m, _DB()) == ["20 g de harina de Negrito"]
    assert m["ingredients"][-1] == "20 g de harina de Negrito" and m["ingredients_raw"][-1] == "20 g de harina de Negrito"


def test_si_la_lista_ya_la_trae_no_se_duplica():
    m = _cena(["mide 20 g de harina de Negrito"], lista=["2 filetes de pescado", "30 g de harina de maíz"])
    assert lp.reconciliar_meal(m, _DB()) == []


@pytest.mark.parametrize("paso", [
    "⚠️ Seguridad alimentaria: sirve 30 g de queso pasteurizado.",
    "🛡️ Sustitución por alergia declarada: se reemplazó 80 g de yogur por yogur de coco.",
    "Hierve 250 ml de agua con sal.",
    "Añade 0.5 g de ajo en polvo.",
])
def test_notas_agua_y_migajas_no(paso):
    assert lp.reconciliar_meal(_cena([paso]), _DB()) == []


def test_tope_de_dos_por_comida():
    m = _cena(["mide 20 g de harina, 30 g de queso blanco, 10 g de nueces y 30 g de pan rallado"])
    assert len(lp.reconciliar_meal(m, _DB())) == 2


def test_el_alergeno_que_solo_esta_en_un_paso_ya_lo_ve_la_guarda():
    import graph_orchestrator as go
    import nutrition_db
    original = nutrition_db.IngredientNutritionDB
    nutrition_db.IngredientNutritionDB = _DB
    try:
        m = _cena(["Montaje: corona con 10 g de nueces picadas."])
        viol = go.clinical_backstop_for_meal(m, allergies=["Frutos secos"])
    finally:
        nutrition_db.IngredientNutritionDB = original
    assert viol and "nuec" in " ".join(viol).lower(), viol
    assert all("nuec" not in x for x in m["ingredients"]), "la guarda mira una COPIA: no muta la comida"


def test_el_escudo_de_la_generacion_lo_llama_antes_de_las_guardas():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('__import__("lista_desde_pasos").reconciliar(plan, _db)')
    assert i < src.index("# ── Guard 2 (FS1): food-safety") < src.index("# ── Guard 2.5 (FS-IgE)")
    assert '__import__("lista_desde_pasos").con_lo_que_miden_los_pasos(meal)' in src


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 191 and m.group(2) >= "2026-09-24"
