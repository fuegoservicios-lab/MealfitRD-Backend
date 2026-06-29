"""[P1-INGREDIENT-DOUBLE-FRACTION · 2026-06-29] Backstop de display: "0.5 jugo de 0.5 limón" (doble fracción malformada,
visto en vivo en 2 recetas) → "jugo de 0.5 limón". La cantidad líder es un prepend espurio sobre un nombre que ya trae su
fracción. Corre en humanize_plan_ingredients (última mutación, display-only → no toca macros/shopping). Test PURO.
"""
from __future__ import annotations

from humanize_ingredients import _collapse_double_fraction, humanize_plan_ingredients


def test_collapses_double_fraction():
    assert _collapse_double_fraction("0.5 jugo de 0.5 limón") == "jugo de 0.5 limón"


def test_handles_unicode_fraction():
    assert _collapse_double_fraction("½ jugo de ½ limón") == "jugo de ½ limón"
    assert _collapse_double_fraction("0.5 jugo de ½ limón") == "jugo de ½ limón"


def test_idempotent():
    once = _collapse_double_fraction("0.5 jugo de 0.5 limón")
    assert _collapse_double_fraction(once) == once  # 2da pasada no toca


def test_leaves_normal_ingredients_intact():
    for s in ("150g de arroz", "2 huevos", "1 cdta de aceite de oliva", "jugo de 0.5 limón",
              "1.5 taza de lechuga romana picada", "100g de pollo a la plancha"):
        assert _collapse_double_fraction(s) == s


def test_failsafe_non_string():
    assert _collapse_double_fraction(None) is None  # no lanza


def test_integration_in_humanize_plan():
    plan = {"days": [{"meals": [{"ingredients": ["0.5 jugo de 0.5 limón", "150g de arroz"], "recipe": []}]}]}
    humanize_plan_ingredients(plan)
    ings = plan["days"][0]["meals"][0]["ingredients"]
    # ya no hay doble fracción
    assert not any(i.count("0.5") >= 2 for i in ings)
    assert any("limón" in i or "limon" in i for i in ings)


def test_anchor():
    import pathlib
    import humanize_ingredients as hi
    src = pathlib.Path(hi.__file__).read_text(encoding="utf-8")
    assert "P1-INGREDIENT-DOUBLE-FRACTION" in src
    assert "INGREDIENT_DOUBLE_FRACTION_FIX" in src
    # el backstop se invoca dentro de humanize_plan_ingredients antes de humanize_ingredient
    i_call = src.index("_collapse_double_fraction(ing)")
    i_hum = src.index("humanize_ingredient(ing_clean)")
    assert i_call < i_hum
