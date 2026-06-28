"""[P1-SUBST-RECIPE-REWRITE · 2026-06-28] El motor de sustitución por condición/alérgeno removía el ingrediente de
`ingredients` pero dejaba la mención VIEJA en los PASOS de la receta ("añade la miel" cuando la miel ya no está). Fix:
`_rewrite_recipe_steps_after_subs` reescribe los pasos. Diseño verificado por review adversaria: acento-tolerante,
elimina el artículo (evita género roto), salta texturas culinarias, longest-token-first, `\\b` exacto (no `\\w*`).
"""
from __future__ import annotations

import graph_orchestrator as g


def test_phantom_miel_rewritten_in_steps():
    meal = {"recipe": ["Vierte la miel y mezcla bien.", "Añade la miel si deseas dulzor."]}
    ch = g._rewrite_recipe_steps_after_subs(meal, [(["miel"], "Stevia al gusto")])
    assert ch is True
    joined = " ".join(meal["recipe"]).lower()
    assert "miel" not in joined
    assert "stevia" in joined


def test_accent_tolerant_azucar():
    # el paso lleva tilde ("azúcar") pero el token buscado es sin tilde ("azucar") → debe matchear igual
    meal = {"recipe": ["Agrega el azúcar y revuelve hasta disolver."]}
    ch = g._rewrite_recipe_steps_after_subs(meal, [(["azucar", "azúcar", "sugar"], "Stevia al gusto")])
    assert ch is True
    assert "azúcar" not in meal["recipe"][0].lower() and "azucar" not in meal["recipe"][0].lower()
    assert "stevia" in meal["recipe"][0].lower()


def test_texture_word_not_touched():
    # 'crema' es textura culinaria (en skip-set) → NO debe reescribir "forma una crema suave"
    meal = {"recipe": ["Bate hasta formar una crema suave y homogénea."]}
    ch = g._rewrite_recipe_steps_after_subs(meal, [(["crema", "crema de leche"], "yogurt griego")])
    assert ch is False
    assert "crema suave" in meal["recipe"][0]


def test_multiword_longest_first_no_duplication():
    meal = {"recipe": ["Ralla el queso cheddar por encima."]}
    g._rewrite_recipe_steps_after_subs(meal, [(["queso cheddar", "cheddar"], "queso cottage")])
    s = meal["recipe"][0].lower()
    assert "queso cottage" in s
    assert "cheddar" not in s
    assert s.count("queso") == 1, f"no debe duplicar 'queso': {s}"


def test_cross_gender_drops_article():
    # mantequilla(fem) → aceite(masc): al eliminar el artículo evita el agramatical "la aceite"
    meal = {"recipe": ["Derrite la mantequilla en la sartén."]}
    g._rewrite_recipe_steps_after_subs(meal, [(["mantequilla"], "aceite de oliva")])
    s = meal["recipe"][0].lower()
    assert "la aceite" not in s
    assert "aceite de oliva" in s


def test_idempotent():
    meal = {"recipe": ["Vierte la miel y mezcla."]}
    g._rewrite_recipe_steps_after_subs(meal, [(["miel"], "Stevia al gusto")])
    ch2 = g._rewrite_recipe_steps_after_subs(meal, [(["miel"], "Stevia al gusto")])
    assert ch2 is False  # post-fix el token viejo ya no está


def test_failsafe_no_recipe():
    meal = {}  # sin recipe
    assert g._rewrite_recipe_steps_after_subs(meal, [(["miel"], "stevia")]) is False


def test_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-SUBST-RECIPE-REWRITE" in src
    assert "def _rewrite_recipe_steps_after_subs" in src
    assert "recipe_token_subs.append" in src  # se propaga desde _apply_substitutions_core
