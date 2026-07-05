"""[P2 batch · 2026-07-05] Review visual #6 (plan e49d44c3) — 6 P2:

- P2-CLARAS-CAP-KNOB-PARITY: "8 claras de huevo (267g)" con MEALFIT_MAX_EGG_WHITES_PER_MEAL=6 —
  el count-cap realista decía 8 hardcoded; ahora sigue al knob (SSOT).
- P2-QTYSYNC-CADA-TOTAL: "Unta CADA tortilla con 2.5 cdas de mantequilla de maní" cuando 2½ cdas
  es el TOTAL de la línea → leído per-unidad triplica la grasa → "(en total)".
- P2-QTYSYNC-MULTIUSE: línea "½ cda de aceite" con DOS menciones de "½ cdas de aceite" en pasos
  (suman 2× la línea) → la 2ª se reescribe a "el aceite restante".
- P2-COOKED-RAW-ANNOTATION: "(62g cocido, 60g raw)" — ratio imposible para granos (60g crudos
  rinden ~1 taza) → crudo recomputado a cocido/2.8 + "raw"→"crudo".
- P2-NOTE-LINE-NAME-ALIGN: nota "💪 Incorpora queso cottage…" con línea "110 g de queso" → la
  nota adopta el nombre de la línea (verdad de compras).
- P2-BOUNDARY-DISPLAY-POLISH: "Cdta de miel (opcional)" PERSISTIDO aunque el prettify lo resuelve
  → humanize display-only también en el persist boundary (paridad con el finalizador de updates).
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


# ───────────────────── claras ≤ knob ─────────────────────

def test_claras_count_cap_follows_knob(go):
    assert go._REALISM_COUNT_CAPS["clara"] == float(go.MAX_EGG_WHITES_PER_MEAL), \
        "el techo servible de claras sigue a MEALFIT_MAX_EGG_WHITES_PER_MEAL (era 8.0 hardcoded)"


def test_eight_claras_capped(go, monkeypatch):
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)

    class _NoopDB:
        def macros_from_ingredient_string(self, s):
            return None

        def lookup(self, s):
            return None

    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Yuca con Claras Revueltas",
        "ingredients": ["8 claras de huevo (267g)", "150 g de yuca"],
        "ingredients_raw": ["8 claras de huevo (267g)", "150 g de yuca"],
        "recipe": ["Mise en place: x.", "El Toque de Fuego: revuelve 3-4 min.", "Montaje: y."],
    }]}]
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    line = days[0]["meals"][0]["ingredients"][0]
    import re
    n = float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", line).group(1).replace(",", "."))
    assert n <= float(go.MAX_EGG_WHITES_PER_MEAL) + 0.01, f"8 claras → ≤{go.MAX_EGG_WHITES_PER_MEAL}: {line}"


# ───────────────────── qty-sync: cada + total ─────────────────────

def test_cada_with_line_total_gets_en_total(go):
    meal = {
        "name": "Tortillas con Mantequilla de Maní y Fresas",
        "ingredients": ["3 tortillas de trigo (90g)", "2½ cdas de mantequilla de maní (40g)",
                        "1 taza de fresas frescas (149g)"],
        "recipe": ["Mise en place: lava las fresas.",
                   "Montaje: unta cada tortilla tostada con 2½ cdas de mantequilla de maní. "
                   "Coloca las fresas encima."],
    }
    assert go._sync_recipe_step_quantities(meal) >= 1
    blob = " ".join(str(s) for s in meal["recipe"])
    assert "(en total)" in blob, f"el total per-unidad se anota: {blob}"
    # idempotente
    _before = list(meal["recipe"])
    go._sync_recipe_step_quantities(meal)
    assert meal["recipe"] == _before


def test_cada_with_per_unit_qty_untouched(go):
    meal = {
        "name": "Tostadas",
        "ingredients": ["3 tortillas de trigo (90g)", "3 cdas de mantequilla de maní (48g)"],
        "recipe": ["Montaje: unta cada tortilla con 1 cda de mantequilla de maní."],
    }
    go._sync_recipe_step_quantities(meal)
    assert "(en total)" not in " ".join(str(s) for s in meal["recipe"]), \
        "1 cda per-unidad ≠ total de línea (3 cdas) → legítimo, intacto"


# ───────────────────── qty-sync: multiuso ─────────────────────

def test_second_total_mention_becomes_restante(go):
    meal = {
        "name": "Frijoles Pintos Guisados con Arroz",
        "ingredients": ["¼ taza de frijoles pintos secos (49g)", "½ cda de aceite de oliva",
                        "½ cebolla"],
        "recipe": ["Mise en place: pica la cebolla.",
                   "El Toque de Fuego: calienta ½ cda de aceite de oliva y sofríe la cebolla. "
                   "Para las arepitas: cocina en una sartén con ½ cda de aceite de oliva 3 minutos por lado.",
                   "Montaje: sirve."],
    }
    assert go._sync_recipe_step_quantities(meal) >= 1
    blob = " ".join(str(s) for s in meal["recipe"])
    assert blob.count("½ cda de aceite") == 1, f"solo la 1ª mención conserva la cantidad: {blob}"
    assert "el aceite de oliva restante" in blob, f"la 2ª pasa a 'restante': {blob}"


def test_multiuse_mentions_not_force_synced_to_total(go):
    meal = {
        "name": "Guiso en dos tandas",
        "ingredients": ["1 cda de aceite de oliva"],
        "recipe": ["El Toque de Fuego: calienta ½ cda de aceite de oliva. "
                   "Luego añade ½ cda de aceite de oliva a la segunda tanda."],
    }
    go._sync_recipe_step_quantities(meal)
    blob = " ".join(str(s) for s in meal["recipe"])
    assert "1 cda de aceite" not in blob, \
        "las menciones ½+½ (que ya suman la línea) NO se sobreescriben cada una al total (eso duplicaba)"


# ───────────────────── anotación cocido/crudo ─────────────────────

def test_impossible_grain_ratio_recomputed(go):
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Cerdo con Bok Choy y Arroz Integral",
        "ingredients": ["⅓ taza de arroz integral cocido (62g cocido, 60g raw)"],
        "ingredients_raw": ["⅓ taza de arroz integral cocido (62g cocido, 60g raw)"],
        "recipe": ["Mise en place: cocina el arroz integral (60g raw rinden ~1 taza cocida)."],
    }]}]
    assert go._fix_cooked_raw_annotations(days) >= 2
    line = days[0]["meals"][0]["ingredients"][0]
    assert "raw" not in line and "crudo" in line, f"'raw' → 'crudo': {line}"
    import re
    m = re.search(r"\((\d+)g cocido, (\d+)g crudo\)", line)
    assert m, line
    assert int(m.group(2)) <= int(int(m.group(1)) * 0.55), \
        f"crudo recomputado a ~cocido/2.8 (los granos absorben agua): {line}"
    assert "raw" not in " ".join(days[0]["meals"][0]["recipe"]), "el paso también se traduce"


def test_plausible_ratio_untouched(go):
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Arroz",
        "ingredients": ["1 taza de arroz cocido (160g cocido, 55g crudo)"],
        "ingredients_raw": ["1 taza de arroz cocido (160g cocido, 55g crudo)"],
        "recipe": ["Mise: x."],
    }]}]
    assert go._fix_cooked_raw_annotations(days) == 0, "ratio plausible + ya en español → no-op"


# ───────────────────── nota 💪 ↔ línea ─────────────────────

def test_note_adopts_line_food_name(go):
    meal = {
        "name": "Lechosa con Maní y Queso",
        "ingredients": ["1 lechosa mediana (203g)", "110 g de queso"],
        "recipe": ["Mise en place: pela la lechosa.",
                   "Incorpora queso cottage a la preparación y mézclalo antes de servir.",
                   "Montaje: sirve."],
    }
    assert go._align_closer_note_food_names(meal) == 1
    assert any("Incorpora queso a la preparación" in str(s) for s in meal["recipe"]), \
        "la nota usa el nombre de la LÍNEA (lo que el usuario compra)"


def test_note_align_ambiguous_two_cheese_lines_untouched(go):
    meal = {
        "name": "Doble Queso",
        "ingredients": ["50 g de queso", "30 g de queso de freír"],
        "recipe": ["Incorpora queso cottage a la preparación y mézclalo antes de servir."],
    }
    assert go._align_closer_note_food_names(meal) == 0, "2 líneas con el mismo token → ambiguo, no tocar"


# ───────────────────── humanize en boundary ─────────────────────

def test_boundary_runs_display_humanize(go):
    days = [{"day": 1, "meals": [{
        "meal": "Merienda", "name": "Tostadas con Miel",
        "ingredients": ["3 tortillas de trigo (90g)", "Cdta de miel (opcional)"],
        "ingredients_raw": ["3 tortillas de trigo (90g)", "1 cdta de miel"],
        "recipe": ["Mise en place: tuesta las tortillas 1 minuto por lado.", "Montaje: unta y sirve."],
    }]}]
    go.finalize_plan_data_coherence(days)
    ings = days[0]["meals"][0]["ingredients"]
    assert any(str(s).startswith("1 cdta de miel") for s in ings), \
        f"el persist boundary re-humaniza display ('Cdta de miel' → '1 cdta de miel'): {ings}"


def test_boundary_polish_anchor_present():
    assert "P2-BOUNDARY-DISPLAY-POLISH" in _GO
    i = _GO.index("[P2-BOUNDARY-DISPLAY-POLISH · 2026-07-05]")
    assert "_prettify_quantity_display" in _GO[i:i + 1200], \
        ("el boundary llama SOLO al prettify cosmético — el humanize completo re-introduciría "
         "'lonjas' caseras que slice-grams acaba de convertir (test_p1_coherence_finalize lo ancla)")
