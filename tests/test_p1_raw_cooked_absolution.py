"""[P1-RAW-COOKED-ABSOLUTION + P1-CLOSER-INTEGRATE + P2s · 2026-07-06]

Review visual #8 (plan cb5f8480):
- Banner "pollo/marisco o carne CRUDOS… usa pollo previamente congelado" sobre POLLO HERVIDO 20
  minutos. Triple raíz: (a) la rama ambigua (ceviche/tartar + proteína) no chequeaba estado de
  cocción; (b) el scan leía las notas ⚠ inyectadas — que contienen "ceviche, sushi, tartar" — y
  el flag se AUTO-PERPETUABA; (c) nada removía una nota stale (el step-rewriter de una
  sustitución la renombró pescado→pollo).
- "15 g de queso blanco" + "200 g de queso" del closer sin paso que lo use → el closer ahora
  ESCALA la línea congruente existente; "4 huevos enteros"+"1 huevo" → tails ignorables.
- "Cocina filete de pescado blanco cocido a la plancha": el note-align copiaba el sufijo
  " cocido" y el eater de unidades era goloso sin unidad ("½ filete de pescado"→"pescado").
- "¼ cdta de mantequilla de maní (1g)": el trim dejaba líneas-polvo → piso per-línea 5g.
- "2 cebollas" + "100g Cebolla": merge cross-forma con peso unitario (→ "2½ cebollas").
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


# ───────────── absolución por cocción ─────────────

def _ceviche_pollo_cocido():
    return {"days": [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Ensalada estilo Ceviche de Pollo con Arándanos",
        "ingredients": ["1 muslo de pollo (con piel, cocido y desmenuzado, ~151g)",
                        "½ taza de arándanos frescos", "Jugo de 1 limón"],
        "recipe": ["Mise en place: Cocina el muslo de pollo en agua con sal hasta que esté "
                   "tierno (unos 20 minutos). Desmenúzalo.",
                   "El Toque de Fuego: El pollo se cocina aparte en agua hirviendo con ajo.",
                   "Montaje: mezcla el pollo desmenuzado con los arándanos y el jugo de limón."],
    }]}]}


def test_cooked_poultry_absolved(go):
    plan = _ceviche_pollo_cocido()
    assert go._scan_raw_seafood_meat_violations(plan) == [], \
        "pollo HERVIDO 20 min jamás gana el banner 'pollo CRUDO' aunque el nombre diga ceviche"


def test_genuinely_raw_still_flagged(go):
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Tartar de Atún",
        "ingredients": ["150g de atún fresco", "Jugo de 1 limón"],
        "recipe": ["Mise en place: corta el atún en cubos.",
                   "Montaje: mezcla con el limón y sirve de inmediato."],
    }]}]}
    assert go._scan_raw_seafood_meat_violations(plan) != [], \
        "tartar de atún SIN calor sigue flaggeado (la absolución exige evidencia de cocción)"


def test_scan_ignores_injected_safety_notes(go):
    plan = _ceviche_pollo_cocido()
    # nota vieja renombrada (pescado→pollo por el step-rewriter) pegada al final:
    plan["days"][0]["meals"][0]["recipe"].append(
        "⚠️ Seguridad alimentaria: este plato lleva pollo/marisco o carne CRUDOS (ceviche, sushi, "
        "tartar). El cítrico del ceviche NO cuece el pollo.")
    assert go._scan_raw_seafood_meat_violations(plan) == [], \
        "la nota ⚠ inyectada (contiene 'ceviche, sushi, tartar') no puede auto-perpetuar el flag"


def test_stale_note_removed_by_food_safety_pass(go):
    plan = _ceviche_pollo_cocido()
    plan["days"][0]["meals"][0]["recipe"].append(
        "⚠️ Seguridad alimentaria: este plato lleva pollo/marisco o carne CRUDOS (ceviche, sushi, "
        "tartar). El cítrico del ceviche NO cuece el pollo. Usa pollo previamente congelado.")
    plan["days"][0]["meals"][0]["_food_safety_seafood"] = True
    go._apply_food_safety_fixes(plan)
    rec = plan["days"][0]["meals"][0]["recipe"]
    assert not any("o carne CRUDOS (ceviche" in str(s) for s in rec), \
        "la nota stale se REMUEVE cuando el plato ya no califica"
    assert plan["days"][0]["meals"][0].get("_food_safety_seafood") is None


# ───────────── closer integra, no apila ─────────────

class _ScaleDB:
    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
        return float(m.group(1).replace(",", ".")) if m else None


def test_congruent_line_scaled_not_stacked(go):
    meal = {"name": "Bollitos de Yuca Rellenos de Queso",
            "ingredients": ["15 g de queso blanco", "150 g de yuca"],
            "ingredients_raw": ["15 g de queso blanco", "150 g de yuca"]}
    assert go._scale_congruent_protein_line(meal, "queso", 185, _ScaleDB()) is True
    ings = meal["ingredients"]
    assert len(ings) == 2, f"cero líneas nuevas: {ings}"
    n = float(re.match(r"^\s*(\d+)", ings[0]).group(1))
    assert abs(n - 200) <= 2, f"15g + 185g → ~200g en la MISMA línea: {ings[0]}"
    assert meal["ingredients_raw"][0] == ings[0], "lockstep raw"


def test_no_congruent_line_returns_false(go):
    meal = {"name": "Ensalada", "ingredients": ["100 g de lechuga"],
            "ingredients_raw": ["100 g de lechuga"]}
    assert go._scale_congruent_protein_line(meal, "queso", 100, _ScaleDB()) is False, \
        "sin línea congruente → el caller añade la línea nueva (comportamiento previo)"


def test_count_lines_ignorable_tail_merged(go, monkeypatch):
    monkeypatch.setattr(go, "INGREDIENT_LINE_CONSOLIDATE_ENABLED", True)
    days = [{"day": 1, "meals": [{
        "meal": "Desayuno", "name": "Revoltillo",
        "ingredients": ["4 huevos enteros", "1 huevo"],
        "ingredients_raw": ["4 huevos enteros", "1 huevo"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    assert go._consolidate_duplicate_gram_lines(days) >= 1
    ings = days[0]["meals"][0]["ingredients"]
    assert len(ings) == 1 and ings[0].startswith("5 huevos"), \
        f"'enteros' es tail ignorable → '4 huevos enteros'+'1 huevo' = '5 huevos enteros': {ings}"


# ───────────── cross-forma cebollas ─────────────

def test_crossform_count_plus_grams_merged(go, monkeypatch):
    monkeypatch.setattr(go, "INGREDIENT_LINE_CONSOLIDATE_ENABLED", True)
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Wrap de Hígado",
        "ingredients": ["2 cebollas", "135 g de hígado de res", "100g Cebolla"],
        "ingredients_raw": ["2 cebollas", "135 g de hígado de res", "100g Cebolla"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    assert go._consolidate_duplicate_gram_lines(days) >= 1
    ings = days[0]["meals"][0]["ingredients"]
    assert not any(re.match(r"^\s*100g", str(i)) for i in ings), f"la línea en gramos se fusiona: {ings}"
    _ceb = next(i for i in ings if "cebolla" in str(i).lower())
    assert _ceb.startswith("2½ cebollas") or _ceb.startswith("3 cebollas"), \
        f"2 + 100g/150g ≈ 2.67 → redondeo a ½ → '2½ cebollas': {_ceb}"


# ───────────── note-align: sufijo cocido + unit-eater ─────────────

def test_align_never_copies_cocido_suffix(go):
    meal = {"name": "Wrap de Batata",
            "ingredients": ["80g de filete de pescado blanco cocido", "½ batata mediana (76g)"],
            "recipe": ["Mise en place: hornea la batata.",
                       "💪 Cocina filete de pescado a la plancha o hervido y sírvelo como proteína del plato.",
                       "Montaje: enrolla."]}
    go._align_closer_note_food_names(meal)
    _note = next(s for s in meal["recipe"] if "💪" in str(s))
    assert "cocido a la plancha" not in _note, \
        f"el align jamás copia ' cocido' de la línea al paso: {_note}"


def test_lead_strip_preserves_food_without_unit(go):
    assert go._re.sub(go._LEAD_QTY_UNIT_STRIP_RE, "", "½ filete de pescado").strip() == \
        "filete de pescado", "sin unidad, el eater no se come 'filete'"
    assert go._re.sub(go._LEAD_QTY_UNIT_STRIP_RE, "", "250g de calamar limpio").strip() == \
        "calamar limpio"
    assert go._re.sub(go._LEAD_QTY_UNIT_STRIP_RE, "", "½ taza de yogurt griego").strip() == \
        "yogurt griego"


# ───────────── piso del trim (dust lines) ─────────────

def test_trim_never_leaves_dust_line(go):
    class _DustDB:
        def macros_from_ingredient_string(self, s):
            low = str(s).lower()
            m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
            g = float(m.group(1).replace(",", ".")) if m else 0.0
            if "mantequilla" in low:
                return {"fats": 0.5 * g, "protein": 0.0, "carbs": 0.0}
            return {}

        def grams_from_ingredient_string(self, s):
            m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
            return float(m.group(1).replace(",", ".")) if m else None

    import graph_orchestrator as g
    meals = [{"meal": "Merienda", "name": "Yogurt", "fats": 10, "protein": 10, "carbs": 20,
              "cals": 210,
              "ingredients": ["19 g de mantequilla de maní"],
              "ingredients_raw": ["19 g de mantequilla de maní"]}]
    monkey_group = g._ingredient_macro_group
    g._ingredient_macro_group = lambda s, db: "fats" if "mantequilla" in str(s).lower() else None
    try:
        g._trim_day_fats_to_target(meals, 1.0, _DustDB(), tol=0.0)
    finally:
        g._ingredient_macro_group = monkey_group
    line = meals[0]["ingredients"][0]
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)", line)
    assert m and float(m.group(1).replace(",", ".")) >= 5.0, \
        f"el trim jamás deja una línea-polvo (<5g): {line}"
