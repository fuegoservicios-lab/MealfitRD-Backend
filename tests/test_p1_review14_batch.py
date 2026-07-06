"""[P1-STEP-HARINA-GHOST + P2-STEP-PHRASE-DEDUP + P2-RABANO-CAP · 2026-07-06] Review #14 del owner
sobre el plan renovado (17c3fa8f):

1. HARINA FANTASMA: el "Muslo de Pollo Horneado" pasa el muslo "por la harina sazonada" pero
   harina NO está en ingredients[] → costra incocinable + harina no se compra.
2. "queso blanco y queso blanco batidos" en la Batata Rellena — frase duplicada (artefacto LLM).
3. RÁBANO 9 paquetes = RD$765: la receta pedía 135g/porción de rábano (guarnición) × ~13
   repeticiones en 30 días → cap de guarnición.
"""
import pytest

import graph_orchestrator as go


# ───────────── 1. harina fantasma (carb-ghost) ─────────────

def _muslo():
    return [{"day": 1, "meals": [{
        "name": "Muslo de Pollo Horneado con Costra",
        "ingredients": ["1 muslo de pollo (149 g)", "15 g de queso blanco"],
        "ingredients_raw": ["1 muslo de pollo (149 g)", "15 g de queso blanco"],
        "recipe": ["Mise en place: bate el queso blanco. Mezcla la harina con sal, pimienta y orégano.",
                   "El Toque de Fuego: pasa el muslo por la harina sazonada y hornea a 200°C."],
    }]}]


def test_harina_phantom_materialized():
    days = _muslo()
    n = go._add_missing_recipe_step_carbs(days, allergies=None)
    ings = " ".join(days[0]["meals"][0]["ingredients"]).lower()
    assert n >= 1 and "harina de trigo" in ings, (
        f"la harina del empanizado se materializa (se compra + coherente): {ings}"
    )


def test_harina_maiz_not_materialized_as_wheat():
    # arepa: "harina de maíz" en pasos → NO materializar harina de trigo (es otro alimento).
    days = [{"day": 1, "meals": [{
        "name": "Arepitas de Maíz",
        "ingredients": ["1 huevo"],
        "ingredients_raw": ["1 huevo"],
        "recipe": ["Mezcla la harina de maíz precocida con agua tibia y forma discos."],
    }]}]
    go._add_missing_recipe_step_carbs(days, allergies=None)
    ings = " ".join(days[0]["meals"][0]["ingredients"]).lower()
    assert "harina de trigo" not in ings, f"harina de maíz ≠ harina de trigo: {ings}"


def test_harina_already_present_not_duplicated():
    days = [{"day": 1, "meals": [{
        "name": "Empanada",
        "ingredients": ["30 g de harina de trigo", "1 huevo"],
        "ingredients_raw": ["30 g de harina de trigo", "1 huevo"],
        "recipe": ["Amasa la harina con el huevo."],
    }]}]
    assert go._add_missing_recipe_step_carbs(days, allergies=None) == 0, "harina ya presente → no duplica"


def test_harina_blocked_by_gluten_allergy():
    days = _muslo()
    go._add_missing_recipe_step_carbs(days, allergies=["gluten"])
    ings = " ".join(days[0]["meals"][0]["ingredients"]).lower()
    assert "harina de trigo" not in ings, "fail-secure: no materializar trigo con alergia a gluten"


# ───────────── 2. dedup de frase repetida ─────────────

def test_phrase_dedup_collapses_repeat():
    assert go._dedup_repeated_phrase(
        "mezcla con queso blanco y queso blanco batidos") == "mezcla con queso blanco batidos"


def test_phrase_dedup_leaves_distinct_pair():
    # "X y Y" (distintos) intacto.
    assert go._dedup_repeated_phrase("sirve el arroz y las habichuelas") == "sirve el arroz y las habichuelas"
    assert go._dedup_repeated_phrase("cebolla y ajo picados") == "cebolla y ajo picados"


def test_phrase_dedup_in_plan_steps():
    days = [{"day": 1, "meals": [{
        "name": "Batata Rellena", "ingredients": ["55 g de queso blanco"],
        "ingredients_raw": ["55 g de queso blanco"],
        "recipe": ["El Toque de Fuego: mezcla con queso blanco y queso blanco batidos, sazona."],
    }]}]
    n = go._dedup_repeated_phrases_in_plan(days)
    step = days[0]["meals"][0]["recipe"][0]
    assert n == 1 and step.lower().count("queso blanco") == 1, f"dup colapsado: {step}"


# ───────────── 3. rábano cap (funcional, espejo del patrón guineo) ─────────────

def test_rabano_cap_fires(caplog):
    import logging
    from constants import strip_accents as _sa
    import shopping_calculator as sc
    with caplog.at_level(logging.WARNING):
        # 135g × 13 servings ≈ 1755g de rábano (guarnición sobre-asignada)
        sc.aggregate_and_deduct_shopping_list(
            plan_ingredients=["135 g rábano"] * 13, multiplier=1.0, structured=True)
    rab_caps = [r for r in caplog.records
                if "P5-VEG-CAP" in r.message and "rabano" in _sa(r.message.lower())]
    assert rab_caps, (
        f"P5-VEG-CAP debe capear el rábano (~1755g → guarnición). "
        f"Warnings: {[r.message for r in caplog.records if 'VEG-CAP' in r.message]}"
    )
    # baja MUCHO desde 1755g (el valor exacto depende de la density del master).
    assert any("1755g" in r.message for r in rab_caps), (
        f"el cap parte de ~1755g de demanda: {[r.message for r in rab_caps]}"
    )


def test_rabano_config_anchored():
    import shopping_calculator as sc
    import inspect
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "'rabano'" in src and "P2-RABANO-CAP" in src
