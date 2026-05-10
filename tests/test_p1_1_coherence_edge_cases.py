"""[P1-1-COHERENCE-EDGE · 2026-05-10] Edge cases del guard recetas↔lista
no cubiertos por v3 (P3-A multiplier + P3-4 pavo).

Tres modos de falso positivo conocidos cerrados aquí:

  1. Plurales no-pavo: receta dice "manzanas" pero la lista canonicaliza a
     "manzana" → guard reportaba `cap_swallowed_modifier`.
  2. Modificadores triviales: receta dice "pollo orgánico" pero master_map
     no lo tiene como alias → guard veía dos foods distintos.
  3. Líquidos / condimentos: receta escala lineal por household_multiplier
     pero el usuario compra cantidad casi constante → magnitudes divergían
     siempre (50%+) en hogares 4×.

Cobertura del test:
    1. _singularize_food_es: heurística vowel-before-s + mapping irregular.
    2. _strip_trailing_modifier_es: solo strippea trailing reconocido + guards.
    3. _canonicalize_for_coherence: integra ambos como fallback (master_map gana).
    4. Helpers de líquidos: keyword set, tolerancia, _is_liquid_food.
    5. Filtro post-magnitudes: divergencias de líquidos dentro de tolerancia
       ampliada NO se reportan; fuera de tolerancia SÍ.
    6. Symmetric: ambos lados del guard reciben la misma canonicalización
       (test E2E mínimo con plan_data).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. Singularización es-DO
# ---------------------------------------------------------------------------
class TestSingularize:
    @pytest.mark.parametrize("plural,singular", [
        ("manzanas", "manzana"),
        ("tomates", "tomate"),
        ("cebollas", "cebolla"),
        ("pimientos", "pimiento"),
        ("papas", "papa"),
        ("aguacates", "aguacate"),
        ("zanahorias", "zanahoria"),
        ("plátanos", "plátano"),
    ])
    def test_regular_plural_stripped(self, plural, singular):
        from shopping_calculator import _singularize_food_es
        assert _singularize_food_es(plural) == singular

    @pytest.mark.parametrize("plural,singular", [
        ("limones", "limón"),
        ("jamones", "jamón"),
        ("frijoles", "frijol"),
        ("camarones", "camarón"),
        ("salmones", "salmón"),
        ("panes", "pan"),
        ("flores", "flor"),
        ("huevos", "huevo"),
        ("yogures", "yogur"),
    ])
    def test_irregular_plural_via_mapping(self, plural, singular):
        from shopping_calculator import _singularize_food_es
        assert _singularize_food_es(plural) == singular

    @pytest.mark.parametrize("name", [
        "arroz",     # no termina en -s
        "pez",       # idem
        "ajo",       # ya singular
        "pollo",     # idem
        "manzana",   # ya singular
        "pan",       # consonante, no -s
    ])
    def test_already_singular_unchanged(self, name):
        from shopping_calculator import _singularize_food_es
        assert _singularize_food_es(name) == name

    @pytest.mark.parametrize("bad", [None, "", "  ", 123, [], {}])
    def test_invalid_input_returns_input(self, bad):
        from shopping_calculator import _singularize_food_es
        assert _singularize_food_es(bad) == bad

    def test_too_short_not_stripped(self):
        """`as` (<4 chars) no se strippea — degeneración inaceptable."""
        from shopping_calculator import _singularize_food_es
        # 3 chars con final 's' precedido de vocal: no strippear.
        assert _singularize_food_es("ais") == "ais"


# ---------------------------------------------------------------------------
# 2. Strip de modificadores
# ---------------------------------------------------------------------------
class TestStripModifier:
    @pytest.mark.parametrize("name,stripped", [
        ("pollo orgánico", "pollo"),
        ("arroz integral", "arroz"),
        ("leche descremada", "leche"),
        ("tomate fresco", "tomate"),
        ("pimiento rojo", "pimiento"),
        ("frijoles negros", "frijoles"),  # plural se mantiene; singularize lo cubre después
        ("huevo entero", "huevo"),
        ("queso blanco", "queso"),
    ])
    def test_trailing_modifier_stripped(self, name, stripped):
        from shopping_calculator import _strip_trailing_modifier_es
        assert _strip_trailing_modifier_es(name) == stripped

    @pytest.mark.parametrize("name", [
        "pollo",                  # sin modificador
        "arroz blanco integral",  # solo strippea último; reduce a "arroz blanco"
        "leche",
        "tomate",
        "pan",
    ])
    def test_no_modifier_or_already_clean(self, name):
        from shopping_calculator import _strip_trailing_modifier_es
        out = _strip_trailing_modifier_es(name)
        # `arroz blanco integral` → "arroz blanco" (un solo strip).
        if name == "arroz blanco integral":
            assert out == "arroz blanco"
        else:
            assert out == name

    def test_single_word_not_stripped(self):
        """`orgánico` solo (1 token) no se strippea — quedaría vacío."""
        from shopping_calculator import _strip_trailing_modifier_es
        assert _strip_trailing_modifier_es("orgánico") == "orgánico"

    def test_short_remainder_preserved(self):
        """Si el strip dejara <3 chars, no se aplica."""
        from shopping_calculator import _strip_trailing_modifier_es
        # "té verde" → strippear "verde" dejaría "té" (2 chars) → no strip.
        assert _strip_trailing_modifier_es("té verde") == "té verde"

    @pytest.mark.parametrize("bad", [None, "", "  "])
    def test_invalid_input(self, bad):
        from shopping_calculator import _strip_trailing_modifier_es
        assert _strip_trailing_modifier_es(bad) == bad


# ---------------------------------------------------------------------------
# 3. Integración en _canonicalize_for_coherence
# ---------------------------------------------------------------------------
class TestCanonicalizeIntegration:
    def test_plural_collapses_when_master_silent(self):
        """Si master_map no tiene "manzanas" como alias, fallback singulariza."""
        from shopping_calculator import _canonicalize_for_coherence

        # Master map vacío → caemos al fallback genérico.
        with patch("shopping_calculator.get_master_ingredients", return_value=[]):
            out = _canonicalize_for_coherence(["manzanas", "manzana"])
        # Ambos deben colapsar al mismo canónico (set tiene 1 elemento).
        assert len(out) == 1
        assert "manzana" in out

    def test_modifier_collapses_when_master_silent(self):
        """`pollo orgánico` y `pollo` colapsan en fallback."""
        from shopping_calculator import _canonicalize_for_coherence

        with patch("shopping_calculator.get_master_ingredients", return_value=[]):
            out = _canonicalize_for_coherence(["pollo orgánico", "pollo"])
        assert len(out) == 1
        assert "pollo" in out

    def test_master_map_wins_over_fallback(self):
        """Si master_map tiene un alias explícito, NO aplicamos fallback."""
        from shopping_calculator import _canonicalize_for_coherence

        master = [{"name": "Manzana Roja", "aliases": ["manzanas"]}]
        with patch("shopping_calculator.get_master_ingredients", return_value=master):
            out = _canonicalize_for_coherence(["manzanas"])
        # master entrega "Manzana Roja" — fallback NO singulariza.
        assert "Manzana Roja" in out

    def test_pavo_rules_unaffected_by_p1_1(self):
        """Pavo sigue ganando vs strip/singularize (regresión P3-4)."""
        from shopping_calculator import _canonicalize_for_coherence

        with patch("shopping_calculator.get_master_ingredients", return_value=[]):
            out = _canonicalize_for_coherence(["pechuga de pavo fresca"])
        # canonicalize_pavo manda → "Pechuga de pavo".
        assert "Pechuga de pavo" in out


# ---------------------------------------------------------------------------
# 4. Helpers de líquidos
# ---------------------------------------------------------------------------
class TestLiquidHelpers:
    def test_default_keywords(self, monkeypatch):
        from shopping_calculator import _get_coherence_liquid_keywords
        monkeypatch.delenv("MEALFIT_COHERENCE_LIQUID_KEYWORDS", raising=False)
        kws = _get_coherence_liquid_keywords()
        assert "aceite" in kws
        assert "vinagre" in kws

    def test_custom_keywords_csv(self, monkeypatch):
        from shopping_calculator import _get_coherence_liquid_keywords
        monkeypatch.setenv("MEALFIT_COHERENCE_LIQUID_KEYWORDS", "miel,jarabe, salsa picante  ")
        kws = _get_coherence_liquid_keywords()
        assert kws == {"miel", "jarabe", "salsa picante"}

    def test_default_tolerance(self, monkeypatch):
        from shopping_calculator import _get_coherence_liquid_tolerance_pct
        monkeypatch.delenv("MEALFIT_COHERENCE_LIQUID_TOLERANCE_PCT", raising=False)
        assert _get_coherence_liquid_tolerance_pct() == 0.50

    def test_custom_tolerance(self, monkeypatch):
        from shopping_calculator import _get_coherence_liquid_tolerance_pct
        monkeypatch.setenv("MEALFIT_COHERENCE_LIQUID_TOLERANCE_PCT", "0.75")
        assert _get_coherence_liquid_tolerance_pct() == 0.75

    def test_invalid_tolerance_falls_back(self, monkeypatch):
        from shopping_calculator import _get_coherence_liquid_tolerance_pct
        monkeypatch.setenv("MEALFIT_COHERENCE_LIQUID_TOLERANCE_PCT", "9.0")
        # Validator rechaza >5.0 → cae al default.
        assert _get_coherence_liquid_tolerance_pct() == 0.50

    def test_is_liquid_food_substring_match(self):
        from shopping_calculator import _is_liquid_food
        kws = {"aceite", "vinagre"}
        assert _is_liquid_food("Aceite de oliva", kws) is True
        assert _is_liquid_food("Vinagre balsámico", kws) is True
        assert _is_liquid_food("Pollo", kws) is False
        assert _is_liquid_food("", kws) is False
        assert _is_liquid_food("Pollo", set()) is False


# ---------------------------------------------------------------------------
# 5. Filtro post-magnitudes para líquidos
# ---------------------------------------------------------------------------
class TestLiquidPostFilter:
    """[P1-1] Verifica que `run_shopping_coherence_guard` ignora divergencias
    de magnitud para líquidos cuyo `delta_pct` cae dentro de la tolerancia
    ampliada, pero las reporta cuando excede."""

    def _build_plan(self, recipe_qty: float, list_qty: float):
        """Plan mínimo con UNA receta y UN item de lista del mismo food."""
        return {
            "calc_household_multiplier": 1.0,
            "days": [{
                "meals": [{
                    "meal": "Almuerzo",
                    "ingredients_raw": [{
                        "quantity": recipe_qty, "unit": "ml", "name": "Aceite de oliva",
                    }],
                }],
            }],
            "aggregated_shopping_list": [
                {"name": "Aceite de oliva", "quantity": list_qty, "unit": "ml"},
            ],
        }

    def test_liquid_within_extended_tolerance_no_divergence(self, monkeypatch):
        """Aceite recipe=10ml, list=15ml → delta=50%; default tol_liquid=50% → OK, no divergencia magnitud."""
        from shopping_calculator import run_shopping_coherence_guard

        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "warn")
        # Tolerancia base 5%; líquidos 50%.
        monkeypatch.delenv("MEALFIT_COHERENCE_LIQUID_TOLERANCE_PCT", raising=False)

        with patch("shopping_calculator.get_master_ingredients", return_value=[]):
            plan = self._build_plan(recipe_qty=10.0, list_qty=15.0)
            divs = run_shopping_coherence_guard(plan, mode_override="warn", multiplier=1.0)

        # Sin divergencias de magnitud (presence/absence ambos OK también).
        assert all(d.get("magnitude") is not True for d in divs)

    def test_liquid_outside_extended_tolerance_reports(self, monkeypatch):
        """Aceite recipe=10ml, list=100ml → delta=900%; aún fuera de tol_liquid=50% → SI reporta."""
        from shopping_calculator import run_shopping_coherence_guard

        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "warn")

        with patch("shopping_calculator.get_master_ingredients", return_value=[]):
            plan = self._build_plan(recipe_qty=10.0, list_qty=100.0)
            divs = run_shopping_coherence_guard(plan, mode_override="warn", multiplier=1.0)

        # Debe haber al menos una divergencia magnitud.
        mag_divs = [d for d in divs if d.get("magnitude") is True]
        assert len(mag_divs) >= 1

    def test_non_liquid_uses_base_tolerance(self, monkeypatch):
        """Pollo (no-líquido) con delta=50% SIGUE reportándose porque base=10%."""
        from shopping_calculator import run_shopping_coherence_guard

        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "warn")
        # Tolerance base 10% (default).
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)

        plan = {
            "calc_household_multiplier": 1.0,
            "days": [{"meals": [{
                "meal": "Almuerzo",
                "ingredients_raw": [{"quantity": 100, "unit": "g", "name": "Pollo"}],
            }]}],
            "aggregated_shopping_list": [{"name": "Pollo", "quantity": 150, "unit": "g"}],
        }
        with patch("shopping_calculator.get_master_ingredients", return_value=[]):
            divs = run_shopping_coherence_guard(plan, mode_override="warn", multiplier=1.0)

        mag_divs = [d for d in divs if d.get("magnitude") is True]
        assert len(mag_divs) >= 1, "Pollo con delta=50% debe reportarse (no es líquido)"


# ---------------------------------------------------------------------------
# 6. E2E mínimo: plurales y modificadores no producen falsos positivos
# ---------------------------------------------------------------------------
class TestE2ENoFalsePositives:
    def test_plural_recipe_singular_list_no_divergence(self):
        """Receta dice 'manzanas', lista dice 'manzana' → guard limpio."""
        from shopping_calculator import run_shopping_coherence_guard

        plan = {
            "calc_household_multiplier": 1.0,
            "days": [{"meals": [{
                "meal": "Desayuno",
                "ingredients_raw": [{"quantity": 2, "unit": "uds", "name": "Manzanas"}],
            }]}],
            "aggregated_shopping_list": [
                {"name": "Manzana", "quantity": 2, "unit": "uds"},
            ],
        }
        with patch("shopping_calculator.get_master_ingredients", return_value=[]):
            divs = run_shopping_coherence_guard(plan, mode_override="warn", multiplier=1.0)

        # Plurales colapsan → sin divergencia presence; magnitudes coinciden.
        crit = [d for d in divs if d.get("hypothesis") == "cap_swallowed_modifier"]
        assert len(crit) == 0, f"Plural debería colapsar; divs={divs}"

    def test_modifier_recipe_plain_list_no_divergence(self):
        """Receta 'pollo orgánico', lista 'pollo' → guard limpio."""
        from shopping_calculator import run_shopping_coherence_guard

        plan = {
            "calc_household_multiplier": 1.0,
            "days": [{"meals": [{
                "meal": "Almuerzo",
                "ingredients_raw": [{"quantity": 200, "unit": "g", "name": "Pollo orgánico"}],
            }]}],
            "aggregated_shopping_list": [
                {"name": "Pollo", "quantity": 200, "unit": "g"},
            ],
        }
        with patch("shopping_calculator.get_master_ingredients", return_value=[]):
            divs = run_shopping_coherence_guard(plan, mode_override="warn", multiplier=1.0)

        crit = [d for d in divs if d.get("hypothesis") == "cap_swallowed_modifier"]
        assert len(crit) == 0, f"Modificador trailing debería colapsar; divs={divs}"
