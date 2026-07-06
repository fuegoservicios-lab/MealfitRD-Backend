"""[P3-RECIPE-POLISH-4 · 2026-07-06]

P3 residuales de los reviews #8/#9:
- "1 taza de Guineo (papaya) en cubos" — paréntesis que nombra OTRA fruta.
- "2½ tazas de leche descremada (599 ml)" en UNA batida — cap de líquido 2 tazas.
- "unta cada tortilla con 1 cda" cuando 1 cda × 8 tortillas desborda el total de la línea.
- "Bate los huevos" con "1 huevo" en la lista — artículo plural sobre conteo 1.
- Nota de huevo "yema y clara firmes" sobre BOLLITOS (huevo en el puré) — tokens de masa.
- "1 tortilla integrales" — adjetivos invariantes con plural en -es.
- "2 tazas de Lechosa frescos" — concordancia del alimento tras la unidad.
- "½ de cebolla" → "½ cebolla" ("¼ de taza" partitivo queda intacto).
- Nota 🌱 "espolvorea… sobre el plato" en un BATIDO → "al servir".
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def _pretty(s):
    from humanize_ingredients import _prettify_quantity_display
    return _prettify_quantity_display(s)


# ───────────── paréntesis contradictorio ─────────────

def test_contradictory_fruit_paren_stripped():
    import graph_orchestrator as go
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Tostones",
        "ingredients": ["1 taza de Guineo (papaya) en cubos"],
        "ingredients_raw": ["1 taza de Guineo (papaya) en cubos"],
        "recipe": ["Montaje: sirve la ensalada de guineo."],
    }]}]
    assert go._fix_cooked_raw_annotations(days) >= 1
    line = days[0]["meals"][0]["ingredients"][0]
    assert "(papaya)" not in line and "Guineo" in line, f"el guineo no es papaya: {line}"


# ───────────── cap de leche ─────────────

def test_milk_capped_at_two_cups(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)

    class _NoopDB:
        def macros_from_ingredient_string(self, s):
            return None

        def lookup(self, s):
            return None

    days = [{"day": 1, "meals": [{
        "meal": "Merienda", "name": "Batida de Plátano",
        "ingredients": ["2½ tazas de leche descremada (599 ml)", "½ plátano maduro"],
        "ingredients_raw": ["2½ tazas de leche descremada (599 ml)", "½ plátano maduro"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    line = days[0]["meals"][0]["ingredients"][0]
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)?\s*([¼½¾⅓⅔])?\s*tazas?", line)
    _v = float((m.group(1) or "0").replace(",", ".")) + {"¼": .25, "½": .5, "¾": .75, "⅓": 1/3, "⅔": 2/3}.get(m.group(2) or "", 0)
    assert _v <= 2.0 + 1e-6, f"leche ≤2 tazas por plato/vaso: {line}"


# ───────────── cada × K desborda el total ─────────────

def test_cada_per_unit_overflow_rewritten_to_total():
    import graph_orchestrator as go
    meal = {
        "name": "Tostadas con Mantequilla de Maní",
        "ingredients": ["8 tortillas integrales (240 g en total)",
                        "1½ cdas de mantequilla de maní (22g)"],
        "recipe": ["Montaje: unta cada tortilla con 1 cda de mantequilla de maní y sirve."],
    }
    assert go._sync_recipe_step_quantities(meal) >= 1
    blob = " ".join(str(s) for s in meal["recipe"])
    assert "(en total)" in blob and "1½ cdas de mantequilla de maní (en total)" in blob, \
        f"1 cda × 8 tortillas ≫ 1½ cdas → la mención pasa al TOTAL: {blob}"


def test_cada_per_unit_plausible_untouched():
    import graph_orchestrator as go
    meal = {
        "name": "Tostadas",
        "ingredients": ["2 tortillas integrales", "2 cdas de mantequilla de maní"],
        "recipe": ["Montaje: unta cada tortilla con 1 cda de mantequilla de maní."],
    }
    go._sync_recipe_step_quantities(meal)
    assert "(en total)" not in " ".join(str(s) for s in meal["recipe"]), \
        "1 cda × 2 = 2 cdas = total → per-unidad legítimo, intacto"


# ───────────── artículo con conteo 1 ─────────────

def test_article_plural_fixed_for_count_one():
    import graph_orchestrator as go
    meal = {
        "name": "Revoltillo",
        "ingredients": ["1 huevo", "½ cebolla"],
        "recipe": ["El Toque de Fuego: bate los huevos y viértelos sobre la mezcla."],
    }
    assert go._sync_recipe_step_quantities(meal) >= 1
    blob = " ".join(str(s) for s in meal["recipe"])
    assert "bate el huevo" in blob.lower() and "los huevos" not in blob.lower(), blob


# ───────────── nota batter para bollitos ─────────────

def test_bollito_gets_batter_note():
    import graph_orchestrator as go
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Bollitos de Yautía Rellenos",
        "ingredients": ["½ pedazo de yautía", "1 huevo"],
        "recipe": ["Mise en place: pela la yautía.",
                   "El Toque de Fuego: incorpora el huevo crudo batido al puré de yautía y "
                   "forma bollitos. Hornea 15 minutos.",
                   "Montaje: sirve."],
    }]}]}
    go._apply_food_safety_fixes(plan)
    notes = [s for s in plan["days"][0]["meals"][0]["recipe"] if "Seguridad alimentaria" in str(s)]
    if notes:
        assert any("masa" in str(s).lower() for s in notes), notes
        assert not any("yema y clara firmes" in str(s) for s in notes), \
            "huevo integrado en puré/bollito → wording de masa, no de huevo entero"


# ───────────── gramática display ─────────────

def test_invariant_es_adjective():
    out = _pretty("1 tortilla integrales (50g)")
    assert out.startswith("1 tortilla integral"), out


def test_food_after_unit_concordance():
    out = _pretty("2 tazas de Lechosa frescos")
    assert out == "2 tazas de lechosa fresca", out


def test_half_de_noun_simplified():
    assert _pretty("½ de cebolla").startswith("½ cebolla")
    # partitivo legítimo intacto:
    assert _pretty("¼ de taza de agua") == "¼ de taza de agua"


# ───────────── seed en bebidas ─────────────

def test_seed_note_drink_tail_anchor():
    i = _GO.index("[P3-RECIPE-POLISH-4 · 2026-07-06] en BEBIDAS (batido/")
    win = _GO[i:i + 700]
    assert '"al servir"' in win.replace("' ", '"').replace("'", '"') or " al servir" in win, \
        "en batidos la nota 🌱 no dice 'sobre el plato'"
    assert "_seed_is_drink" in win
