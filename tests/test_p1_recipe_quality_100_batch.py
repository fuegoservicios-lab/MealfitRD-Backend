"""[P1-RECIPE-QUALITY-100 · 2026-07-10] Batch de calidad de recetas — los defectos visibles en el
plan vivo 6d742f23 (12 screenshots del owner) que separaban las recetas del 100%:

A1. "Incorpora también aceite de oliva durante la preparación" REDUNDANTE ×3: el reverse-coherence
    excluye 'aceite' como token FILLER (correcto en cola: "atún en aceite") pero para "aceite de
    oliva" el head ES la identidad — los pasos dicen "el aceite" y solo se buscaba "oliva".
A2. Gramática plural del closer: "Cocina camarones ... o hervido y sírvelo" → hervidos/sírvelos.
B.  "pechuga de pollo de pechuga de pollo" (tortitas de maíz): los rewriters de sustitución
    matcheaban el token fuente ('pollo') DENTRO de una ocurrencia YA existente del reemplazo
    ('pechuga de pollo') → duplicación. Guard: matches dentro del target existente no se tocan.
C.  Líneas duplicadas "15 g de queso" + "40 g de queso" (bollitos ñame) — repro del consolidador.
D.  Pescado/camarones bolted a merienda/plato dulce vía _protein_repeat_autofix (el closer ya tenía
    sweet/light-guard; el autofix no): en contexto dulce/ligero el target se restringe a queso
    (dulce) o queso/legumbre (ligero salado); sin candidato → no tocar (decide el gate).

tooltip-anchor: P1-RECIPE-QUALITY-100
"""
from pathlib import Path

import graph_orchestrator as go

_GO_SRC = Path(go.__file__).read_text(encoding="utf-8")


# ── A1: aceite head-filler en reverse-coherence ──────────────────────────────────

def test_oil_used_as_generic_el_aceite_not_flagged():
    meal = {"name": "Revoltillo con Tostadas",
            "ingredients": ["1 cdta de aceite de oliva", "3 huevos", "2 rebanadas de pan integral"],
            "recipe": ["Mise en place: Casca los huevos y bátelos.",
                       "El Toque de Fuego: Calentar el aceite en una sartén a fuego medio y cuajar "
                       "los huevos 2-3 minutos. Tostar el pan integral.",
                       "Montaje: Sirve el revoltillo sobre el pan."]}
    n_before = len(meal["recipe"])
    go._ensure_ingredients_used_in_recipe(meal)
    joined = " ".join(meal["recipe"]).lower()
    assert "incorpora tambien aceite" not in joined.replace("é", "e")
    assert len(meal["recipe"]) == n_before          # cero pasos espurios


def test_oil_truly_unused_still_flagged():
    meal = {"name": "Ensalada",
            "ingredients": ["1 cdta de aceite de oliva", "1 tomate"],
            "recipe": ["Mise en place: Corta el tomate.",
                       "Montaje: Sirve el tomate en un plato."]}
    go._ensure_ingredients_used_in_recipe(meal)
    joined = " ".join(str(s) for s in meal["recipe"]).lower()
    assert "aceite de oliva" in joined              # sin mención ni genérica → sí se corrige


def test_atun_en_aceite_tail_filler_unchanged():
    # el fix del head NO debe reactivar el falso positivo de cola ("atún en aceite" usado como atún)
    meal = {"name": "Ensalada de Atún",
            "ingredients": ["1 lata de atún en aceite", "1 tomate"],
            "recipe": ["Mise en place: Escurre el atún y corta el tomate.",
                       "Montaje: Mezcla el atún con el tomate y sirve."]}
    n_before = len(meal["recipe"])
    go._ensure_ingredients_used_in_recipe(meal)
    assert len(meal["recipe"]) == n_before


# ── A2: pluralización del paso del closer ────────────────────────────────────────

def test_closer_step_plural_agreement():
    txt = go._closer_protein_step_text("camarones", no_cook=False)
    assert "hervidos" in txt and "sírvelos" in txt
    assert "sírvelo como" not in txt


def test_closer_step_singular_unchanged():
    txt = go._closer_protein_step_text("pechuga de pollo", no_cook=False)
    assert "hervido y sírvelo como proteína del plato" in txt


def test_cook_tail_dedup_regex_matches_both_numbers():
    assert go._COOK_TAIL_RE.match("Cocina camarones a la plancha o hervidos y sírvelos como proteína del plato.")
    assert go._COOK_TAIL_RE.match("Cocina pollo a la plancha o hervido y sírvelo como proteína del plato.")


# ── B: rewriter idempotente (no duplicar dentro del target existente) ────────────

def test_step_rewrite_does_not_duplicate_inside_existing_target():
    meal = {"name": "Tortitas de Maíz",
            "recipe": ["El Toque de Fuego: mezclar el maíz, la harina y la pechuga de pollo "
                       "desmenuzada, y hornear 12-15 minutos."]}
    go._rewrite_recipe_steps_after_subs(meal, [(["pollo"], "pechuga de pollo")])
    joined = " ".join(meal["recipe"]).lower()
    assert "pechuga de pechuga" not in joined
    assert "pollo de pechuga de pollo" not in joined
    assert joined.count("pechuga de pollo") == 1


def test_step_rewrite_still_replaces_bare_token():
    meal = {"name": "Wrap", "recipe": ["El Toque de Fuego: cocina el pollo a la plancha 4 minutos."]}
    changed = go._rewrite_recipe_steps_after_subs(meal, [(["pollo"], "pechuga de pollo")])
    assert changed is True
    assert "pechuga de pollo" in " ".join(meal["recipe"]).lower()


# ── C: consolidador de líneas duplicadas (repro bollitos: 15g + 40g de queso) ────

def test_consolidate_duplicate_gram_lines_same_name():
    days = [{"day": 1, "meals": [{"name": "Bollitos de Ñame",
                                  "ingredients": ["15 g de queso", "1 cdta de harina", "40 g de queso"]}]}]
    merged = go._consolidate_duplicate_gram_lines(days)
    ings = days[0]["meals"][0]["ingredients"]
    assert merged >= 1
    assert sum(1 for i in ings if "queso" in i.lower()) == 1
    assert any("55" in i for i in ings)             # 15+40 fusionado


def test_consolidate_plural_singular_names_merge():
    days = [{"day": 1, "meals": [{"name": "Batata",
                                  "ingredients": ["15g de frijoles pintos cocido", "30g de frijoles pintos cocidos"]}]}]
    go._consolidate_duplicate_gram_lines(days)
    ings = days[0]["meals"][0]["ingredients"]
    assert sum(1 for i in ings if "frijoles" in i.lower()) == 1


def test_consolidate_respects_raw_when_misaligned_but_still_merges_display():
    # ingredients_raw desalineado: NO tocar raw, pero el display sí debe consolidarse
    days = [{"day": 1, "meals": [{"name": "X",
                                  "ingredients": ["15 g de queso", "40 g de queso"],
                                  "ingredients_raw": ["algo distinto"]}]}]
    go._consolidate_duplicate_gram_lines(days)
    ings = days[0]["meals"][0]["ingredients"]
    assert sum(1 for i in ings if "queso" in i.lower()) == 1
    assert days[0]["meals"][0]["ingredients_raw"] == ["algo distinto"]  # raw intacto


# ── D: autofix slot/sweet-aware (parser del wiring) ─────────────────────────────

def test_autofix_has_sweet_light_target_guard():
    i = _GO_SRC.index("def _protein_repeat_autofix")
    window = _GO_SRC[i:i + 26000]
    assert "_is_sweet_meal" in window, "el autofix debe chequear contexto dulce antes de elegir target"
    assert "_meal_slot_is_light" in window, "el autofix debe chequear slot ligero antes de elegir target"
