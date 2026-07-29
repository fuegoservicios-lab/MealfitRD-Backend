"""[P1-INGREDIENT-SPREAD-SPECIES-GUARD · 2026-07-29] Adversarial review of the P1-FALLBACK-CAUSE-SPLIT
fix round found that Capa 1's whole-phrase `_is_seasoning_name(bare_norm)` exemption in
`_count_ingredient_meal_frequency` silently zeroes the ingredient-spread signal for real,
catalog-verified foods whose canonical name happens to start with (or contain) a word from
`_QTY_GUARD_SEASONING_SKIP`:

  - 'Ají morrón'/'Ají cubanela' (both `category='Vegetales'` in `master_ingredients`, 0.99g/0.86g
    protein per 100g — real, distinct, macro-bearing vegetables, NOT the pinch-quantity chili the
    seasoning list targets) were wholesale-dropped 43/43 and 89/89 occurrences across 30 live plans
    (100% miss rate) because `_is_seasoning_name` matches the standalone word 'aji' anywhere in the
    bare phrase.
  - 'Maní' (peanuts, 24.4g protein/100g — a major macro contributor) was dropped 8/69 occurrences
    whenever the LLM wrote 'sin sal' ('without salt'), because 'sal' matched as a standalone word
    even though the phrase declares the ABSENCE of salt, not its use.

## Why a macro-magnitude threshold can't fix this

SELECT against `master_ingredients` shows genuine seasonings are *more* macro-dense per-100g than
Ají morrón, not less:

    Ají morrón      0.99g protein / 6.03g carbs  / 100g
    Pimienta negra  10.4g protein / 64.0g carbs   / 100g
    Comino          17.81g protein / 44.24g carbs / 100g
    Laurel          7.61g protein / 74.97g carbs  / 100g

Any per-100g macro floor that lets Ají morrón through also lets Pimienta negra/Comino/Laurel through
— reopening the exact 'gusto'/'negra' false positives `test_p1_ingredient_spread_gusto_fix.py`
already closed.

## The actual fix: the `category` column already on every row (no new list)

`master_ingredients.category` has 6 values: Despensa/Frutas/Lácteos/Proteínas/Vegetales/Víveres.
Genuine seasonings from `_QTY_GUARD_SEASONING_SKIP` (Sal, Pimienta negra, Comino, Laurel, ...) all
live in `category='Despensa'`; Ají morrón/Ají cubanela are `category='Vegetales'`. If the bare
resolves to a catalog row whose category is NOT 'Despensa', Capa 1's whole-phrase exemption does
NOT apply (`IngredientNutritionDB.category_of`, nutrition_db.py). A second, independent guard
(`_is_negated_seasoning_mention`) covers 'sin <sazón>' for foods that ARE themselves catalogued as
'Despensa' (Maní) — a category check alone can't save them since Maní and Sal share the same
category bucket.

## What this does NOT touch

`_is_seasoning_name` itself (the SSOT shared with `_ensure_ingredient_quantities` and
`_ensure_ingredients_used_in_recipe`) is untouched — those two callsites correctly continue to treat
'aji cubanela'/'sal' as seasoning-classified strings for THEIR purposes (default-gram injection /
recipe-step requirement), anchored by
`test_p1_seasoning_word_boundary.py::test_los_sazonadores_reales_siguen_exentos`. Both new guards
live ONLY inside `_count_ingredient_meal_frequency`.
"""
from __future__ import annotations

import pathlib

import nutrition_db
import graph_orchestrator as g


class _FakeInfo:
    def __init__(self, name):
        self.name = name


class _FakeCatalogDB:
    """Deterministic, offline stand-in mirroring the shape of `master_ingredients` for the
    fixtures this test cares about: real foods resolve to a non-'Despensa' category (or, for Maní,
    to 'Despensa' but WITH real macro so the negation guard is what saves it); genuine seasonings
    resolve to 'Despensa'."""

    def __init__(self, *a, **kw):
        pass

    def lookup(self, raw_name):
        low = str(raw_name).lower()
        if "pollo" in low:
            return _FakeInfo("Pollo")
        if "arroz" in low:
            return _FakeInfo("Arroz blanco")
        if "aji morron" in low or "ají morrón" in low:
            return _FakeInfo("Ají morrón")
        if "mani" in low or "maní" in low:
            return _FakeInfo("Maní")
        if "pimienta negra" in low:
            return _FakeInfo("Pimienta negra")
        if low.strip() in ("sal", "sal al gusto"):
            return _FakeInfo("Sal")
        return None

    def category_of(self, raw_name):
        low = str(raw_name).lower()
        if "aji morron" in low or "ají morrón" in low:
            return "Vegetales"
        if "pollo" in low:
            return "Proteínas"
        if "arroz" in low:
            return "Víveres"
        if "mani" in low or "maní" in low:
            return "Despensa"
        if "pimienta negra" in low:
            return "Despensa"
        if "sal" in low:
            return "Despensa"
        return None


def _meal(name, ingredients):
    return {"meal": name, "ingredients": ingredients}


def _days_fixture_aji_morron():
    # 4 comidas: 'Ají morrón' en 3/4 (75%, señal REAL de acaparamiento de un vegetal, no sazón).
    return [
        {"day": 1, "meals": [
            _meal("Desayuno", ["1 taza de Ají morrón picado", "150g de pollo"]),
            _meal("Almuerzo", ["1 taza de Ají morrón picado", "100g de arroz"]),
        ]},
        {"day": 2, "meals": [
            _meal("Cena", ["1 taza de Ají morrón picado", "100g de arroz"]),
            _meal("Merienda", ["100g de arroz"]),
        ]},
    ]


def _days_fixture_mani_sin_sal():
    # 4 comidas: 'Maní sin sal' en 3/4 (75%) — la ausencia declarada de sal no debe exentar
    # el alimento REAL (maní) de la cuenta de acaparamiento.
    return [
        {"day": 1, "meals": [
            _meal("Desayuno", ["30g de Maní sin sal", "150g de pollo"]),
            _meal("Almuerzo", ["30g de Maní sin sal", "100g de arroz"]),
        ]},
        {"day": 2, "meals": [
            _meal("Cena", ["30g de Maní sin sal", "100g de arroz"]),
            _meal("Merienda", ["100g de arroz"]),
        ]},
    ]


def _days_fixture_real_seasoning():
    # Regresión: 'Sal al gusto'/'Pimienta negra al gusto' (sazón GENUINO, sin negación) en 100% de
    # las comidas debe seguir exento — el fix no debe reabrir gusto/negra.
    return [
        {"day": 1, "meals": [
            _meal("Desayuno", ["Sal al gusto", "Pimienta negra al gusto", "150g de pollo"]),
            _meal("Almuerzo", ["Sal al gusto", "150g de pollo"]),
        ]},
        {"day": 2, "meals": [
            _meal("Cena", ["Sal al gusto", "Pimienta negra al gusto", "150g de pollo"]),
            _meal("Merienda", ["Sal al gusto", "150g de pollo"]),
        ]},
    ]


def test_aji_morron_survives_as_a_token(monkeypatch):
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeCatalogDB)
    result = g._count_ingredient_meal_frequency(_days_fixture_aji_morron())
    assert "morron" in result, (
        f"'Ají morrón' (category='Vegetales', vegetal real) no debe quedar invisible al detector "
        f"solo porque 'aji' vive en _QTY_GUARD_SEASONING_SKIP. Result: {result}"
    )
    assert result["morron"] >= 0.5


def test_category_guard_is_load_bearing_for_aji(monkeypatch):
    """SABOTAGE CHECK: sin la señal de categoría (simulada aquí devolviendo None siempre, como
    ocurriría si `category_of` fallara o el catálogo no tuviera la columna poblada), 'Ají morrón'
    vuelve a ser invisible — prueba que el guard de categoría es lo que lo salva, no una
    coincidencia de otra parte del pipeline."""
    class _NoCategoryDB(_FakeCatalogDB):
        def category_of(self, raw_name):
            return None

    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _NoCategoryDB)
    result = g._count_ingredient_meal_frequency(_days_fixture_aji_morron())
    assert "morron" not in result, (
        f"Sin la señal de categoría, 'Ají morrón' debería volver a exentarse (comportamiento "
        f"pre-fix) — si esto falla, el fixture ya no prueba que el guard es load-bearing. "
        f"Result: {result}"
    )


def test_mani_sin_sal_survives_as_a_token(monkeypatch):
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeCatalogDB)
    result = g._count_ingredient_meal_frequency(_days_fixture_mani_sin_sal())
    assert "mani" in result, (
        f"'Maní sin sal' no debe quedar invisible solo porque 'sal' aparece NEGADA en la frase. "
        f"Result: {result}"
    )
    assert result["mani"] >= 0.5


def test_negation_guard_is_load_bearing_for_mani(monkeypatch):
    """SABOTAGE CHECK: si el guard de negación se apaga (simulado monkeypatcheando el helper a
    `lambda *_: False`), 'Maní sin sal' vuelve a ser invisible — category_of='Despensa' (igual
    que Sal) no alcanza por sí sola; el guard de negación es lo que lo salva."""
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeCatalogDB)
    monkeypatch.setattr(g, "_is_negated_seasoning_mention", lambda *_a, **_kw: False)
    result = g._count_ingredient_meal_frequency(_days_fixture_mani_sin_sal())
    assert "mani" not in result, (
        f"Sin el guard de negación, 'Maní sin sal' debería volver a exentarse (comportamiento "
        f"pre-fix) — si esto falla, el fixture ya no prueba que el guard es load-bearing. "
        f"Result: {result}"
    )


def test_genuine_seasoning_without_negation_still_exempt(monkeypatch):
    """Regresión: 'Sal al gusto'/'Pimienta negra al gusto' (sazón real, SIN 'sin') en el 100% de
    las comidas debe seguir exento — el fix no debe reabrir gusto/negra
    (test_p1_ingredient_spread_gusto_fix.py)."""
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeCatalogDB)
    result = g._count_ingredient_meal_frequency(_days_fixture_real_seasoning())
    assert "gusto" not in result
    assert "negra" not in result
    assert "sal" not in result


def test_is_seasoning_name_ssot_untouched():
    """El fix NO debe modificar `_is_seasoning_name` — sigue siendo la SSOT pura (sin DB) de los
    otros 2 callsites, que DEBEN seguir clasificando 'aji cubanela'/'sal' como sazón para sus
    propios propósitos (inyección de gramos default / exigencia de paso de receta). Ver
    test_p1_seasoning_word_boundary.py::test_los_sazonadores_reales_siguen_exentos."""
    assert g._is_seasoning_name("aji cubanela") is True
    assert g._is_seasoning_name("ajies morrones") is True
    assert g._is_seasoning_name("sal") is True


def test_anchor_present_in_source():
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-INGREDIENT-SPREAD-SPECIES-GUARD" in src
    assert "_is_negated_seasoning_mention" in src
    assert "category_of" in src
    src_db = pathlib.Path(nutrition_db.__file__).read_text(encoding="utf-8")
    assert "def category_of" in src_db
