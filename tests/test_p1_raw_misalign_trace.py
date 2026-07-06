"""[P1-RAW-MISALIGN-TRACE + P2-STEM-BOUNDED + P2s · 2026-07-06]

Review visual #10 (plan 6a078619, la corrida de validación):
- El tracer de PRIMERA divergencia display↔raw: la telemetría del boundary confirma que hay un
  2º desalineador (raw congelado pre-quantize, tostadas SIN pan en la lista) — cada probe marca
  el meal con el stage donde la divergencia nació (`_misalign_traced` persiste → el forense
  nombra al actor por intervalo).
- "agua"⊂"aguacate": los stems por PREFIJO abierto hacían que "atún en agua" contara como usado
  porque el Montaje decía "cubos de aguacate" → línea de proteína sin paso (reverse-coherence Y
  closer-hygiene engañados). Bounded: `\\b<stem>(?:s|es)?\\b` en los 4 buscadores.
- "215 g de hígado de res" — techo de vísceras 150g (porción cultural + prudencia de retinol).
- "2½ puerro mediano (249 g)" — count-cap puerro 1 + plural.
- "Pela filete… y desvena si es necesario / hasta que estén rosados" — verbos de ESPECIE que la
  sustitución camarón→pescado dejó atrás.
- Tostadas sin pan → "lonjas de pan" al carb-ghost (materializa el síntoma del desalineador).
- Knob MEALFIT_REVIEWER_THINKING_EFFORT (low→max) para graduar el thinking del reviewer.
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


# ───────────── tracer ─────────────

def test_tracer_marks_first_stage_only(go, monkeypatch):
    monkeypatch.setattr(go, "RAW_MISALIGN_TRACER_ENABLED", True)
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "X",
        "ingredients": ["a", "b"], "ingredients_raw": ["a", "b", "c"],
    }]}]
    go._trace_misalign(days, "post_macro_engine")
    assert days[0]["meals"][0]["_misalign_traced"] == "post_macro_engine"
    go._trace_misalign(days, "post_humanize")
    assert days[0]["meals"][0]["_misalign_traced"] == "post_macro_engine", \
        "solo la PRIMERA divergencia queda marcada (nombra al actor por intervalo)"


def test_tracer_probes_wired_in_assemble():
    for stage in ("pre_engine", "post_macro_engine", "post_engine_seam", "post_humanize"):
        assert f'_trace_misalign(days, "{stage}"' in _GO or f'"{stage}")' in _GO, stage


# ───────────── stems bounded (agua⊄aguacate) ─────────────

def test_atun_en_agua_gets_usage_step(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_REVERSE_COHERENCE_ENABLED", True)
    meal = {
        "name": "Wrap de Tortilla con Revoltillo",
        "ingredients": ["2 tortillas integrales", "2 huevos", "100g de atún en agua",
                        "⅓ taza de aguacate en cubos"],
        "recipe": ["Mise en place: corta el tomate. Pela y corta aguacate en cubos.",
                   "El Toque de Fuego: bate los huevos y cuájalos. Calienta las tortillas.",
                   "Montaje: enrolla y sirve con los cubos de aguacate al lado."],
    }
    assert go._ensure_ingredients_used_in_recipe(meal) >= 1, \
        "'agua'⊄'aguacate': el atún en agua NO cuenta como usado por el aguacate del Montaje"
    assert any("atún en agua" in str(s) for s in meal["recipe"])


def test_closer_step_not_suppressed_by_aguacate(go):
    meal = {"name": "Wrap",
            "recipe": ["Mise en place: pela y corta aguacate en cubos.",
                       "Montaje: sirve con aguacate."]}
    assert go._append_closer_protein_step(meal, "atún en agua", False) is True, \
        "el aguacate de los pasos no debe suprimir el paso del atún"


# ───────────── caps hígado / puerro ─────────────

class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def test_organ_meat_capped(go, monkeypatch):
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Hígado Encebollado",
        "ingredients": ["215 g de hígado de res en tiras", "30 g de arroz integral (crudo)"],
        "ingredients_raw": ["215 g de hígado de res en tiras", "30 g de arroz integral (crudo)"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    line = days[0]["meals"][0]["ingredients"][0]
    n = float(re.match(r"^\s*(\d+)", line).group(1))
    assert n <= float(go.ORGAN_MEAT_CAP_G) + 0.01, f"vísceras al techo (retinol): {line}"


def test_puerro_count_capped(go, monkeypatch):
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Bollitos",
        "ingredients": ["2.5 puerro mediano (249 g), en rodajas finas", "55 g de yuca"],
        "ingredients_raw": ["2.5 puerro mediano (249 g), en rodajas finas", "55 g de yuca"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    line = days[0]["meals"][0]["ingredients"][0]
    n = float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", line).group(1).replace(",", "."))
    assert n <= 1.0 + 0.01, f"2½ puerros para unos bollitos → ≤1: {line}"


# ───────────── verbos de especie ─────────────

def test_species_verbs_cleaned_when_no_shrimp(go):
    days = [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Pescado en Salsa Criolla",
        "ingredients": ["½ filete de pescado", "½ tomate"],
        "ingredients_raw": ["½ filete de pescado", "½ tomate"],
        "recipe": ["Mise en place: Pela filete de pescado blanco y desvena si es necesario. "
                   "Pica la cebolla.",
                   "El Toque de Fuego: incorpora filete de pescado blanco y cocina 4 minutos "
                   "hasta que estén rosados."],
    }]}]
    assert go._fix_cooked_raw_annotations(days) >= 2
    blob = " ".join(str(s) for s in days[0]["meals"][0]["recipe"]).lower()
    assert "desvena" not in blob, "el pescado no se desvena (verbo de camarón post-sub)"
    assert "rosados" not in blob and "bien cocido" in blob, blob


def test_species_verbs_kept_for_shrimp(go):
    days = [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Camarones Criollos",
        "ingredients": ["200g de camarones"],
        "ingredients_raw": ["200g de camarones"],
        "recipe": ["Mise en place: pela los camarones y desvena si es necesario.",
                   "El Toque de Fuego: cocina hasta que estén rosados."],
    }]}]
    go._fix_cooked_raw_annotations(days)
    blob = " ".join(str(s) for s in days[0]["meals"][0]["recipe"]).lower()
    assert "desvena" in blob and "rosados" in blob, "en camarones los verbos SON correctos"


# ───────────── pan al ghost ─────────────

def test_missing_bread_materialized(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_STEP_CARB_GUARD_ENABLED", True)
    days = [{"day": 1, "meals": [{
        "meal": "Desayuno", "name": "Tostadas de Queso Mozzarella con Naranja",
        "ingredients": ["40 g de queso", "½ naranja"],
        "ingredients_raw": ["40 g de queso", "½ naranja"],
        "recipe": ["El Toque de Fuego: tuesta las lonjas de pan en una sartén (2-3 minutos).",
                   "Montaje: coloca las tostadas con queso en un plato."],
    }]}]
    assert go._add_missing_recipe_step_carbs(days, db=None, allergies=None) >= 1
    ings = days[0]["meals"][0]["ingredients"]
    assert any("pan integral" in str(i) for i in ings), \
        f"el pan de los pasos se materializa (no se puede tostar lo que no se compra): {ings}"


# ───────────── thinking effort knob ─────────────

def test_thinking_effort_knob():
    m = re.search(r'REVIEWER_THINKING_EFFORT = str\(_env_str\("MEALFIT_REVIEWER_THINKING_EFFORT", ""\)\)', _GO)
    assert m, "knob opcional (vacío = comportamiento actual)"
    i = _GO.index('_think_body = {"type": "enabled"}')
    win = _GO[i:i + 400]
    assert '_think_body["effort"] = REVIEWER_THINKING_EFFORT' in win, \
        "el effort graduable (low→max) entra al extra_body solo si el knob está seteado"
