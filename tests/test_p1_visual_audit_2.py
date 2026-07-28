"""[P1-VISUAL-AUDIT-2 · 2026-07-28] 2ª revisión visual del owner (plan ab2b0a16, 16:24).

Ocho defectos medidos en pantalla → ocho reparaciones deterministas. Cada test ancla el
caso VIVO exacto (verificado por invocación directa antes de wirear — un NameError dentro
del try ancho de finalize da verde desactivando el pase entero).

1. [P1-CHEESE-DUST-BUMP] "0.82 g de queso Gouda rallado": la exención 'rallado' del
   shrink-floor dejaba pasar polvo de queso → bump al piso (15 g).
2. [P1-EGG-COUNT-STEP-SYNC] "1 huevo" listado, pasos baten "el otro huevo" → 2 huevos.
3. [P1-MEAL-DOUBLE-MAIN-RESOLVE] Ropa Vieja de PAVO + "½ filete de PESCADO … como
   proteína del plato": fuera la 2ª principal, primaria re-escalada proteína-conservada,
   cirugía por ORACIÓN del paso bolt (no por paso).
4. [P1-GROUND-MEAT-STEP-NOUN] ingredientes "pollo molido", pasos "pechuga de pollo" ×3.
5. [P1-BIGFRUIT-COUNT-FRACTION] "1 lechosa mediana (202 g)" → "¼ de lechosa mediana".
6. [P1-GRAMMAR-GAPS-2] "1 toronjas mediano" / "1-2 nísperos fresco" / "2½ guineítos verde".
7. [P1-STEP-MANGLED-FRACTION] "añade 2–⅓ taza" (el 2/3 del LLM mutilado) → "⅔ taza".
8. [P1-TOASTED-ASIDE-TOO] yogurt "incorpóralo y mézclalo" sobre TOSTADAS → al lado.
9. [P1-FRESH-OVER-CANNED-DEFAULT] receta pide champiñones FRESCOS, la lista compraba
   "Lata Trozos y Tallos": con forma fresca disponible, la lata sale del set default
   (frasco≠fresco: el Tonnino de atún NO debe filtrar las latas baratas — cazado en
   la verificación, 98→469 RD$).

tooltip-anchor: P1-VISUAL-AUDIT-2
"""
from __future__ import annotations

import graph_orchestrator as go
import humanize_ingredients as hi


def test_cheese_dust_bump():
    days = [{"meals": [{
        "name": "Revoltillo", "protein": 16, "carbs": 8, "fats": 12, "cals": 210,
        "ingredients": ["2 huevos enteros", "0.82 g de queso Gouda rallado"],
    }]}]
    n = go._floor_subservible_portions(days)
    assert n >= 1
    assert days[0]["meals"][0]["ingredients"][1].startswith("15 g de queso")


def test_egg_count_step_sync():
    days = [{"meals": [
        {"name": "Tortitas", "ingredients": ["1 huevo"],
         "recipe": ["bate 1 huevo con la leche", "bate el otro huevo y cocina el revuelto"]},
        {"name": "Sin otro", "ingredients": ["1 huevo entero"], "recipe": ["Bate el huevo."]},
    ]}]
    assert go._egg_count_step_sync(days) == 1
    assert days[0]["meals"][0]["ingredients"][0] == "2 huevos"
    assert days[0]["meals"][1]["ingredients"][0] == "1 huevo entero"


def test_double_main_resolve_sentence_surgery():
    days = [{"meals": [{
        "name": "Ropa Vieja de Pavo Molido sobre Arepitas",
        "ingredients": ["140 g de pavo molido", "2 tomates", "½ filete de pescado"],
        "recipe": ["El Toque de Fuego: sofríe y cocina el pavo. Cocina Filete de pescado "
                   "blanco a la plancha o hervido y sírvelo como proteína del plato."],
    }]}]
    n = go._meal_double_main_resolve(days, db=None)
    assert n == 1
    m = days[0]["meals"][0]
    assert not any("pescado" in str(s).lower() for s in m["ingredients"])
    # La oración legítima del TdF sobrevive; la del bolt muere.
    assert any("cocina el pavo" in p.lower() for p in m["recipe"])
    assert not any("como proteína del plato" in p.lower() for p in m["recipe"])


def test_double_main_no_toca_extensores_ni_unicas():
    days = [{"meals": [
        {"name": "Ensalada con Atún y Huevo",
         "ingredients": ["1 lata de atún en agua", "1 huevo"], "recipe": ["Mezcla."]},
        {"name": "Pollo Guisado", "ingredients": ["150 g de pollo"], "recipe": ["Guisa."]},
    ]}]
    assert go._meal_double_main_resolve(days, db=None) == 0


def test_ground_meat_step_noun_sync():
    days = [{"meals": [{
        "name": "Pollo molido a la Plancha",
        "ingredients": ["125 g de pollo molido"],
        "recipe": ["Añade pechuga de pollo y cocina hasta que pechuga de pollo esté dorado."],
    }]}]
    assert go._ground_meat_step_noun_sync(days) == 1
    assert "pechuga" not in days[0]["meals"][0]["recipe"][0].lower()
    assert "el pollo molido" in days[0]["meals"][0]["recipe"][0]


def test_bigfruit_count_fraction():
    days = [{"meals": [{"ingredients": [
        "1 lechosa mediana (202 g)", "1 lechosa (630 g)", "½ mango mediano (101 g)"]}]}]
    assert go._bigfruit_count_fraction_honesty(days) == 1
    ings = days[0]["meals"][0]["ingredients"]
    assert ings[0].startswith("¼ de lechosa mediana")
    assert ings[1] == "1 lechosa (630 g)"
    assert ings[2] == "½ mango mediano (101 g)"


def test_grammar_gaps_segunda_tanda():
    assert hi._fix_display_grammar("1 toronjas mediano (aprox. 305g)") == \
        "1 toronja mediana (aprox. 305g)"
    assert hi._fix_display_grammar("1-2 nísperos fresco") == "1-2 nísperos frescos"
    assert hi._fix_display_grammar("2½ guineítos verde (aprox. 250g)") == \
        "2½ guineítos verdes (aprox. 250g)"


def test_step_mangled_fraction():
    days = [{"meals": [{"recipe": ["Si está muy seco, añade 2–⅓ taza de agua."]}]}]
    n = go._polish_recipe_step_decimals(days)
    assert n >= 1
    assert "⅔ taza de agua" in days[0]["meals"][0]["recipe"][0]
    assert "2–⅓" not in days[0]["meals"][0]["recipe"][0]


def test_toasted_dish_counts_as_baked_for_aside():
    from constants import strip_accents
    tostadas = {"name": "Tostadas de Pan Integral con Queso Blanco",
                "recipe": ["Tuesta las rebanadas de pan integral en una tostadora."]}
    guiso = {"name": "Pavo guisado", "recipe": ["Guisa el pavo en la olla."]}
    assert go._meal_is_baked(tostadas, strip_accents) is True
    assert go._meal_is_baked(guiso, strip_accents) is False


def test_fresh_over_canned_partition():
    """Estructural sobre la lógica de partición (sin DB): la lata cae SOLO si existe
    forma sin envase de estantería; el frasco no cuenta como fresca."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent / "shopping_calculator.py"
    text = src.read_text(encoding="utf-8")
    assert "P1-FRESH-OVER-CANNED-DEFAULT" in text
    assert '_has_truly_fresh = any(not p.get("_shelf") for p in pkgs)' in text
    assert '"frasco"' in text.split("P1-FRESH-OVER-CANNED-DEFAULT", 1)[1][:2400], (
        "el frasco debe contar como envase de estantería (no-fresco) — sin esto el "
        "Tonnino filtra las latas baratas de atún"
    )
