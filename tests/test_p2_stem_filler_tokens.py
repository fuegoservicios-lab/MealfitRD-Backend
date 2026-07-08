"""[P2-STEM-FILLER-TOKENS + P2-FRUIT-DEDUP-COMPOUND + P2-REPAIR-LIGHT-CAP · 2026-07-06]

Review visual #11 (plan ff673061):
- "55g de atún en agua" SIN paso otra vez — causa nueva: la masa de las arepitas usa "agua"
  REAL en los pasos → el token filler "agua" de "atún en agua" contaba como evidencia de uso
  (reverse-coherence Y closer-hygiene). Los fillers (agua/aceite/lata) salen de los stems.
- "½ taza de harina de Negrito": el fruit-dedup swapeó guineo→Negrito DENTRO de "harina de
  guineo" — fruta ESTRUCTURAL (base de masa) jamás se swapea.
- "140 g de queso" en una merienda (> cap 120): `_repair_protein_floor_post_caps` corre post-caps
  y sus adiciones no heredaban el cap LIGHT del closer.
- "½ guayabas fresco" / "1 filetes de pescado" → plurales.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


# ───────────── filler tokens ─────────────

def test_atun_en_agua_step_added_despite_masa_water(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_REVERSE_COHERENCE_ENABLED", True)
    meal = {
        "name": "Arepitas con Revoltillo",
        "ingredients": ["½ taza de harina de trigo (59 g)", "¼ taza de agua (60 ml)",
                        "2 huevos", "55g de atún en agua"],
        "recipe": ["Mise en place: mezcla la harina. Agrega el agua poco a poco y amasa. "
                   "Bate los huevos.",
                   "El Toque de Fuego: hornea las arepitas 12-15 minutos. Cuaja los huevos.",
                   "Montaje: sirve."],
    }
    assert go._ensure_ingredients_used_in_recipe(meal) >= 1, \
        "el 'agua' de la masa NO es evidencia de uso del ATÚN en agua (token filler)"
    assert any("atún en agua" in str(s) for s in meal["recipe"])


def test_closer_step_not_suppressed_by_masa_water(go):
    meal = {"name": "Arepitas",
            "recipe": ["Mise en place: mezcla la harina con el agua y amasa.",
                       "Montaje: sirve."]}
    assert go._append_closer_protein_step(meal, "atún en agua", False) is True, \
        "el agua de la masa no suprime el paso del atún"


def test_water_line_itself_still_skipped(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_REVERSE_COHERENCE_ENABLED", True)
    meal = {
        "name": "Arepitas",
        "ingredients": ["½ taza de harina de trigo", "¼ taza de agua"],
        "recipe": ["Mise en place: mezcla la harina con la mitad del líquido y amasa.",
                   "Montaje: sirve."],
    }
    _n = go._ensure_ingredients_used_in_recipe(meal)
    assert not any("(complemento)" in str(s) and "agua" in str(s).lower()
                   for s in meal["recipe"]), \
        "la línea de AGUA pura (solo tokens filler) jamás gana complemento (era el skip previo)"


def test_pan_recognized_as_used_len3_stem(go, monkeypatch):
    """[P1-STEM-SHORT-FOOD-NOUN · 2026-07-08] (vivo: Tostadas Francesas con Piña) 'pan' (3 chars)
    se filtraba del stem (len>=4) → ningún token de 'pan integral familiar' matcheaba un paso que
    solo dice 'pan' → complemento espurio. Bajar a len>=3 (protegido por \\b más abajo) lo cierra."""
    monkeypatch.setattr(go, "RECIPE_REVERSE_COHERENCE_ENABLED", True)
    meal = {
        "name": "Tostadas Francesas",
        "ingredients": ["7 lonjas de pan integral familiar"],
        "recipe": ["Mise en place: prepara la mezcla.",
                   "El Toque de Fuego: sumerge cada lonja de pan y dora ambos lados.",
                   "Montaje: sirve."],
    }
    n = go._ensure_ingredients_used_in_recipe(meal)
    assert n == 0, f"'pan' debe reconocerse como usado vía el stem corto: {meal['recipe']}"
    assert not any("(complemento)" in str(s) for s in meal["recipe"])


# ───────────── fruit-dedup compound ─────────────

def test_harina_de_guineo_never_swapped(go):
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Batido de Guineo",
         "ingredients": ["1 guineo"], "ingredients_raw": ["1 guineo"],
         "recipe": ["Montaje: licúa."]},
        {"meal": "Merienda", "name": "Arepitas de Guineo con Ricotta",
         "ingredients": ["½ taza de harina de guineo (64 g)", "¼ taza de queso ricotta"],
         "ingredients_raw": ["½ taza de harina de guineo (64 g)", "¼ taza de queso ricotta"],
         "recipe": ["El Toque de Fuego: mezcla la harina de guineo y cocina.", "Montaje: sirve."]},
    ]}]}
    go.dedup_featured_fruits_in_plan(plan)
    _mer = plan["days"][0]["meals"][1]
    _blob = _mer["name"] + " " + " ".join(str(x) for x in _mer["ingredients"])
    assert "guineo" in _blob.lower(), \
        f"fruta ESTRUCTURAL (harina de guineo) jamás se swapea — 'harina de Negrito' vivo: {_blob}"


# ───────────── repair light cap ─────────────

def test_repair_inherits_light_cap():
    i = _GO.index("[P2-REPAIR-LIGHT-CAP · 2026-07-06]")
    win = _GO[i:i + 700]
    assert "CLOSER_SNACK_MAX_ADD_G" in win and "_LIGHT_MEAL_HINT" in win, \
        "las adiciones del repair post-caps heredan el cap light (140g de queso en merienda vivo)"


# ───────────── plurales ─────────────

def test_guayaba_filete_concordance():
    from humanize_ingredients import _prettify_quantity_display
    assert _prettify_quantity_display("½ guayabas fresco").startswith("½ guayaba fresca")
    assert _prettify_quantity_display("1 filetes de pescado") == "1 filete de pescado"
    assert _prettify_quantity_display("1½ filete de pescado blanco (180 g)").startswith(
        "1½ filetes de pescado blanco")
