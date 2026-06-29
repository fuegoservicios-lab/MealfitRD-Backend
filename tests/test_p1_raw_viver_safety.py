"""[P1-RAW-VIVER-SAFETY · 2026-06-28] Guard food-safety: víveres cianogénicos (yuca/cassava/yautía/ñame) + leguminosas
secas (habichuela/frijol/lenteja/gandul/haba/garbanzo/edamame) servidos CRUDOS. Bug en vivo: "Ceviche de yuca con edamame"
servía yuca CRUDA (linamarina→cianuro) sin cocción. Leguminosas crudas = fitohemaglutinina (PHA). El cook-indicator
absuelve el caso cocido (común es-DO). Solo añade NOTA (macro-preservante, patrón seafood). Caso batido = nota fuerte.

Tests PUROS: scanners + helpers (string logic, sin Neon). La inyección se prueba con RAW_EGG_BLENDED_SUBSTITUTE_ENABLED=
False (evita construir IngredientNutritionDB → sin Neon).
"""
from __future__ import annotations

import graph_orchestrator as g
from constants import strip_accents


def _meal(name, ingredients, recipe):
    return {"name": name, "ingredients": ingredients, "recipe": recipe}


def _scan(meal):
    return g._scan_raw_viver_violations({"days": [{"meals": [meal]}]})


def test_yuca_cruda_detectada():
    v = _scan(_meal("Ceviche de yuca con edamame", ["0.5 yuca", "225g edamame", "mango"],
                    ["Corta la yuca en cubos.", "Mezcla y refrigera 15 min."]))
    assert len(v) == 1


def test_yuca_cocida_no_flag():
    assert _scan(_meal("Yuca Hervida con Pollo", ["150g yuca cocida", "120g pollo"],
                       ["Hierve la yuca 25 min.", "Sirve con pollo guisado."])) == []


def test_habichuelas_crudas_detectadas_plural():
    # PHA — plural "habichuelas" debe matchear (regex con (?:s|es)?)
    v = _scan(_meal("Ensalada de Habichuelas", ["100g habichuelas rojas", "tomate"],
                    ["Mezcla las habichuelas con tomate y sirve."]))
    assert len(v) == 1


def test_habichuelas_guisadas_no_flag():
    assert _scan(_meal("Arroz con Habichuelas Guisadas", ["120g habichuelas", "80g arroz"],
                       ["Guisa las habichuelas con sofrito."])) == []


def test_casabe_no_flag():
    # casabe = yuca ya deshidratada/tostada (cocida); no debe flaguear
    assert _scan(_meal("Atún con Casabe", ["80g atún en agua", "20g casabe"],
                       ["Escurre el atún y acompaña con casabe."])) == []


def test_blended_flag_y_es_blended():
    v = _scan(_meal("Batido de Yuca", ["100g yuca", "1 guineo"], ["Licúa la yuca con el guineo."]))
    assert len(v) == 1 and v[0][3] is True  # blended


def test_no_falso_positivo_vegetal_seguro():
    assert _scan(_meal("Ensalada de Zanahoria", ["100g zanahoria", "limón"], ["Ralla y aliña."])) == []
    # 'haba'⊄'habano', 'name'(ñame)⊄'nombre'
    assert _scan(_meal("Café con Leche", ["1 nombre x"], ["Sirve."])) == []


def test_cook_check_excludes_safety_note():
    # la propia nota de seguridad NO debe auto-absolver (verdict food-safety c-3)
    meal = _meal("Ceviche de yuca", ["0.5 yuca"], ["Corta la yuca.", g._FOOD_SAFETY_NOTE_RAW_VIVER])
    # el helper de cocción debe IGNORAR la línea de la nota → sigue considerándose crudo
    assert g._meal_viver_is_cooked(meal, strip_accents) is False


def test_inject_note_and_idempotent(monkeypatch):
    monkeypatch.setattr(g, "RAW_EGG_BLENDED_SUBSTITUTE_ENABLED", False)  # evita construir DB (sin Neon)
    plan = {"days": [{"meals": [_meal("Ceviche de yuca", ["0.5 yuca", "mango"], ["Corta la yuca y mezcla."])]}]}
    g._apply_food_safety_fixes(plan)
    rec = plan["days"][0]["meals"][0]["recipe"]
    assert any("BIEN COCIDOS" in str(s) for s in rec)
    n1 = len(rec)
    g._apply_food_safety_fixes(plan)  # 2da pasada
    assert len(plan["days"][0]["meals"][0]["recipe"]) == n1  # idempotente


def test_knob_off(monkeypatch):
    monkeypatch.setattr(g, "RAW_EGG_BLENDED_SUBSTITUTE_ENABLED", False)
    monkeypatch.setattr(g, "RAW_VIVER_SAFETY_ENABLED", False)
    plan = {"days": [{"meals": [_meal("Ceviche de yuca", ["0.5 yuca"], ["Corta y mezcla."])]}]}
    g._apply_food_safety_fixes(plan)
    assert not any("BIEN COCIDOS" in str(s) for s in plan["days"][0]["meals"][0]["recipe"])


def test_knob_and_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-RAW-VIVER-SAFETY" in src
    assert "RAW_VIVER_SAFETY_ENABLED" in src
    assert g.RAW_VIVER_SAFETY_ENABLED is True
    assert '"yuca"' in src and '"habichuela"' in src
