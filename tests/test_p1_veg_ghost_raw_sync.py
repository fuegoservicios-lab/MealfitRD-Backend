"""[P1-VEG-GHOST-RAW-SYNC + P2-MONTAJE-COOK-SPLIT · 2026-07-06]

Review visual #9 (plan d4a001eb, el degradado del incidente) — cierre del P1 CRÓNICO:
- ROOT del desalineador display↔raw (7-10 de 12 meals en los planes recientes): el veg-ghost —
  el más viejo de los ghosts — appendeaba "100g Cebolla" SOLO a `ingredients`, jamás al raw →
  len mismatch → los locksteps guardados degradan y el drop del shrink-floor saca líneas solo
  del display → desalineación a MITAD de lista (la que corrompió el integrate del closer).
  Ahora appendea a AMBAS listas + formato "100g de cebolla" (con "de", minúscula).
- Chip "Receta con pasos incompletos" en el Casabe: el tostado vivía en el MONTAJE → el split
  v2 extrae cocción también del Montaje; y el TdF PLACEHOLDER contradictorio ("No aplica
  (preparación en frío con pan tostado)" con "Tuesta las lonjas…" en el Mise) se REEMPLAZA por
  la cocción real.
- "1½ cebollas pequeñas" + "100g Cebolla": el cross-form no veía conteos MIXTOS.
- "8½ aceitunas verde" → aceituna/casabe/molondrón al mapa de plurales (concordancia).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ───────────── veg-ghost: raw lockstep + formato ─────────────

def test_veg_ghost_appends_to_both_lists(monkeypatch):
    import graph_orchestrator as go
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: [
        {"category": "Vegetales", "kcal_per_100g": 25, "name": "Cebolla"}])
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Guiso",
        "ingredients": ["150 g de pollo"],
        "ingredients_raw": ["150 g de pollo"],
        "recipe": ["Mise en place: pica la cebolla.",
                   "El Toque de Fuego: sofríe la cebolla y cocina el pollo 8 min.",
                   "Montaje: sirve."],
    }]}]
    assert go._add_missing_recipe_step_vegetables(days) == 1
    m = days[0]["meals"][0]
    assert len(m["ingredients"]) == len(m["ingredients_raw"]) == 2, \
        "el ROOT del desalineador crónico: el veg-ghost DEBE appendear a AMBAS listas"
    assert m["ingredients"][1] == m["ingredients_raw"][1] == "100g de cebolla", \
        f"formato con 'de' y minúscula (era '100g Cebolla'): {m['ingredients'][1]}"


# ───────────── split v2: Montaje + placeholder ─────────────

def _casabe():
    return {
        "meal": "Merienda", "name": "Casabe con Queso Mozzarella Fresco y Lechosa",
        "ingredients": ["4 trozos de casabe (60 g)", "40 g de queso", "¼ taza de lechosa"],
        "recipe": [
            "Mise en place: Lava lechosa. Corta el queso mozzarella en lonjas finas.",
            "Montaje: Tuesta ligeramente los trozos de casabe en una sartén seca a fuego medio, "
            "1 minuto por lado. Coloca las lonjas de queso mozzarella sobre el casabe caliente. "
            "Sirve con lechosa al lado.",
        ],
    }


def test_montaje_cooking_extracted_to_tdf():
    import graph_orchestrator as go
    meal = _casabe()
    assert go._split_cooking_from_mise(meal) is True
    steps = meal["recipe"]
    _tdf = next((s for s in steps if str(s).lower().startswith("el toque de fuego")), None)
    assert _tdf and "Tuesta ligeramente los trozos de casabe" in _tdf, steps
    _mo = next(s for s in steps if str(s).lower().startswith("montaje"))
    assert "Tuesta" not in _mo and "Coloca las lonjas" in _mo, \
        "el Montaje retiene el emplatado; la cocción se movió al TdF"
    assert go._recipe_step_contract_issues(meal) == [], \
        "el chip 'Receta con pasos incompletos' muere"
    assert go._split_cooking_from_mise(meal) is False, "idempotente"


def test_placeholder_tdf_replaced_with_real_cooking():
    import graph_orchestrator as go
    meal = {
        "meal": "Merienda", "name": "Tostadas con Guineo",
        "ingredients": ["6 lonjas de pan integral familiar", "½ guineo maduro"],
        "recipe": [
            "Mise en place: Tuesta las lonjas de pan integral en una tostadora o sartén a fuego "
            "medio hasta que estén doradas. Pela el guineo y córtalo en rodajas finas.",
            "El Toque de Fuego: No aplica (preparación en frío con pan tostado).",
            "Montaje: Unta cada tostada y sirve.",
        ],
    }
    assert go._split_cooking_from_mise(meal) is True
    _tdf = next(s for s in meal["recipe"] if str(s).lower().startswith("el toque de fuego"))
    assert "No aplica" not in _tdf and "Tuesta las lonjas" in _tdf, \
        f"el placeholder contradictorio se reemplaza por la cocción real: {_tdf}"
    assert sum(1 for s in meal["recipe"]
               if str(s).lower().startswith("el toque de fuego")) == 1, "sin TdF duplicado"


def test_real_tdf_untouched():
    import graph_orchestrator as go
    meal = {
        "name": "Pollo Guisado",
        "recipe": ["Mise en place: pica la cebolla. Sazona el pollo.",
                   "El Toque de Fuego: guisa el pollo 20-25 min a fuego medio tapado.",
                   "Montaje: sirve con arroz. Decora con cilantro."],
    }
    assert go._split_cooking_from_mise(meal) is False, "TdF real presente → no-op"


# ───────────── cross-form mixto ─────────────

def test_crossform_mixed_count_merged(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "INGREDIENT_LINE_CONSOLIDATE_ENABLED", True)
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Cerdo con Yautía",
        "ingredients": ["1½ cebollas pequeñas", "60 g de cerdo", "100g Cebolla"],
        "ingredients_raw": ["1½ cebollas pequeñas", "60 g de cerdo", "100g Cebolla"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    assert go._consolidate_duplicate_gram_lines(days) >= 1
    ings = days[0]["meals"][0]["ingredients"]
    assert not any(re.match(r"^\s*100g", str(i)) for i in ings), f"gram-line fusionada: {ings}"
    _ceb = next(i for i in ings if "cebolla" in str(i).lower())
    assert _ceb.startswith("2 cebollas") or _ceb.startswith("2½ cebollas"), \
        f"1½ + 100g/150g ≈ 2.17 → redondeo a ½ → '2 cebollas': {_ceb}"


# ───────────── plurales nuevos ─────────────

def test_aceitunas_adjective_concordance():
    from humanize_ingredients import _prettify_quantity_display
    out = _prettify_quantity_display("8½ aceitunas verde")
    assert out.startswith("8½ aceitunas verdes"), out


def test_veg_ghost_anchor_present():
    i = _GO.index("[P1-VEG-GHOST-RAW-SYNC · 2026-07-06] ROOT del desalineador")
    win = _GO[i:i + 1600]
    assert 'meal["ingredients_raw"].append(_veg_line)' in win
    assert '100g de {str(nm).lower()}' in win
