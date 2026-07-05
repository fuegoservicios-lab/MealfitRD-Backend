"""[P1-SEAFOOD-MARINADE-BLANCH + P1-NOCOOK-COMPLEMENT-TDF · 2026-07-05]

Review visual #6 (plan e49d44c3):
- "Ceviche de Calamar": los pasos jamás aplican calor y afirman "el ácido del limón cocinará el
  calamar" — FALSO (el ácido no mata anisakis/Vibrio). Doble gap: (a) 'calamar' no estaba en
  `_RAW_ANIMAL_PROTEIN_TERMS` → la rama ambigua de "ceviche" no flageaba; (b) la nota ⚠ advertía
  pero dejaba la receta rota. Fix: blanqueo 1-2 min pre-marinado + corrección del claim + nota
  coherente con el blanqueo.
- "Guineo con mantequilla de maní" (plato FRÍO): el paso "El Toque de Fuego (complemento):
  incorpora también guineos… (~10-12 min a fuego medio)" — fuego falso inyectado por el backstop
  sobre el complemento del reverse-coherence. Fix en ambos lados: el generador emite wording frío
  sin prefijo de fuego, y el backstop reescribe complementos ya persistidos.
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


# ───────────────────── calamar en la co-ocurrencia ─────────────────────

def test_calamar_in_raw_animal_protein_terms(go):
    assert "calamar" in go._RAW_ANIMAL_PROTEIN_TERMS
    assert "sepia" in go._RAW_ANIMAL_PROTEIN_TERMS


def test_ceviche_de_calamar_now_flagged(go):
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Ceviche de Calamar con Pasta Integral",
        "ingredients": ["250g de calamar limpio", "½ taza de jugo de limón (122ml)"],
        "recipe": ["Mise en place: corta el calamar en anillas.",
                   "El Toque de Fuego: mezcla el calamar con el jugo de limón. Deja marinar 10 minutos."],
    }]}]}
    assert go._scan_raw_seafood_meat_violations(plan) != [], \
        "'ceviche' + calamar debe co-ocurrir como proteína animal cruda"


# ───────────────────── blanqueo + claim ─────────────────────

def _ceviche_meal():
    return {
        "meal": "Almuerzo", "name": "Ceviche de Calamar con Pasta Integral",
        "ingredients": ["250g de calamar limpio", "½ taza de jugo de limón (122ml)",
                        "½ taza de pasta integral seca (50g)"],
        "ingredients_raw": ["250g de calamar limpio", "½ taza de jugo de limón (122ml)",
                            "½ taza de pasta integral seca (50g)"],
        "recipe": [
            "Mise en place: corta el calamar en anillas finas.",
            "El Toque de Fuego: cocina la pasta integral en agua hirviendo con sal (8-10 minutos). "
            "Mientras tanto, mezcla el calamar, la cebolla, el jugo de limón. Deja marinar 10 minutos "
            "(el ácido del limón cocinará el calamar).",
            "Montaje: sirve la pasta con el ceviche encima.",
        ],
    }


def test_blanch_injected_and_claim_rewritten(go):
    meal = _ceviche_meal()
    assert go._inject_blanch_for_citrus_marinade(meal) is True
    blob = " ".join(str(s) for s in meal["recipe"]).lower()
    assert "blanquea" in blob, "el calamar crudo marinado recibe blanqueo previo"
    assert "cocinará el calamar" not in blob and "cocinara el calamar" not in blob, \
        "la afirmación falsa del cítrico desaparece"
    assert "no sustituye la cocción" in blob.replace("—", "-") or "no sustituye la coccion" in blob
    assert meal.get("_seafood_blanch_injected") == "calamar"
    # idempotente: 2ª pasada no duplica el blanqueo
    _before = list(meal["recipe"])
    go._inject_blanch_for_citrus_marinade(meal)
    assert " ".join(str(s) for s in meal["recipe"]).lower().count("blanquea") == \
        " ".join(str(s) for s in _before).lower().count("blanquea")


def test_seafood_already_heated_only_claim_fixed(go):
    meal = {
        "name": "Camarones al Limón",
        "ingredients": ["200g de camarones"],
        "recipe": ["Mise en place: pela los camarones.",
                   "El Toque de Fuego: saltea los camarones 4 minutos y deja marinar en el jugo "
                   "(el ácido del limón cocinará los camarones)."],
    }
    go._inject_blanch_for_citrus_marinade(meal)
    blob = " ".join(str(s) for s in meal["recipe"]).lower()
    assert "blanquea" not in blob, "marisco YA salteado → no se blanquea doble"
    assert "cocinará los camarones" not in blob, "pero el claim falso sí se corrige"


def test_food_safety_pass_uses_blanched_note(go):
    plan = {"days": [{"day": 1, "meals": [_ceviche_meal()]}]}
    assert go._apply_food_safety_fixes(plan) >= 1
    rec = plan["days"][0]["meals"][0]["recipe"]
    blob = " ".join(str(s) for s in rec)
    assert "por eso esta receta lo blanquea" in blob, \
        "con blanqueo inyectado, la nota es la variante coherente (no la de 'plato crudo')"
    assert not any("pescado/marisco o carne CRUDOS" in str(s) for s in rec)


def test_knob_off_no_blanch(go, monkeypatch):
    monkeypatch.setattr(go, "SEAFOOD_MARINADE_BLANCH_ENABLED", False)
    meal = _ceviche_meal()
    assert go._inject_blanch_for_citrus_marinade(meal) is False


# ───────────────────── complemento TdF en plato frío ─────────────────────

def _cold_pb_meal(extra_ing=None):
    m = {
        "meal": "Merienda", "name": "Guineo con Mantequilla de Maní",
        "ingredients": ["2 guineos mediano (239g)", "2½ cda de mantequilla de maní (40g)"],
        "ingredients_raw": ["2 guineos mediano (239g)", "2½ cda de mantequilla de maní (40g)"],
        "recipe": ["Mise en place: pela el guineo.",
                   "Montaje: unta la mantequilla de maní sobre el guineo. Sirve inmediatamente."],
    }
    if extra_ing:
        m["ingredients"].append(extra_ing)
        m["ingredients_raw"].append(extra_ing)
    return m


def test_reverse_coherence_cold_wording_on_no_cook(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_REVERSE_COHERENCE_ENABLED", True)
    meal = _cold_pb_meal(extra_ing="125 g de queso")
    assert go._ensure_ingredients_used_in_recipe(meal) >= 1
    _new = [s for s in meal["recipe"] if "queso" in str(s).lower()]
    assert _new, "el ingrediente sin uso gana un paso"
    assert "toque de fuego" not in str(_new[0]).lower(), \
        "plato FRÍO → paso de integración sin prefijo de fuego (nada que el backstop pueda 'temporizar')"


def test_persisted_complement_with_fake_time_rewritten(go):
    meal = _cold_pb_meal()
    meal["recipe"].insert(1, "El Toque de Fuego (complemento): incorpora también guineos mediano "
                             "durante la preparación, integrándolo al plato de forma coherente "
                             "(~10-12 min a fuego medio).")
    assert go._inject_recipe_time_temp_defaults(meal) is True
    blob = " ".join(str(s) for s in meal["recipe"]).lower()
    assert "fuego medio" not in blob, "el tiempo falso desaparece del plato frío"
    assert "toque de fuego" not in blob
    assert any(str(s).startswith("Incorpora también guineos") for s in meal["recipe"])
    assert meal.get("_nocook_complement_rewritten") is True


def test_complement_on_cooked_meal_untouched(go):
    meal = {
        "name": "Pollo Guisado con Arroz",
        "ingredients": ["150 g de pechuga de pollo", "60 g de arroz", "1 tomate"],
        "recipe": ["Mise en place: pica.",
                   "El Toque de Fuego: guisa el pollo 20-25 min a fuego medio tapado.",
                   "El Toque de Fuego (complemento): incorpora también tomate durante la preparación, "
                   "integrándolo al plato de forma coherente (~10-12 min a fuego medio).",
                   "Montaje: sirve."],
    }
    go._inject_recipe_time_temp_defaults(meal)
    assert any("(complemento)" in str(s) for s in meal["recipe"]), \
        "plato con fuego real → el complemento con tiempo es legítimo, intacto"
