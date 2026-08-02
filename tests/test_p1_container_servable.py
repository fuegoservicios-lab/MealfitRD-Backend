"""[P1-CONTAINER-SERVABLE · 2026-08-02] EL ENVASE NO ES EL CONTENIDO.

Forense 2026-08-02: `user_inventory` del owner tenía "Huevo: 2 cartón (20 uds.)" y el pantry
guard (`constants.validate_ingredients_against_pantry`) leyó "límite 1.0 unidad" (con 1 cartón)
para un plato de 3 huevos, con ~20-40 huevos reales en la nevera. El ledger espejo de
regenerate-day (`routers.plans._inventory_grams_ledger`) tiene el mismo problema para envases
por PESO ("1 pote (1.96 kg) de Yogurt griego" → 0g disponibles vía `to_grams` crudo).

Raíz: ambos comparaban unidades-de-PLATO contra unidades-de-ENVASE sin expandir — "2 cartón"
se contaba como "2 huevos sueltos" (el ENVASE, no su CONTENIDO).

Este archivo ancla:
  A) el helper SSOT `canonical_units.expand_container_to_servable` en aislamiento.
  B) el caso E2E real del pantry guard (swap): 2 cartones (20 uds.) → 40 huevos → 3 huevos PASA.
  C) el caso E2E real del ledger de regenerate-day: 1 pote (1.96 kg) → ~1960g → 150g PASA.
  D) fail-open: items inexpandibles se comportan EXACTAMENTE como antes del fix.
"""
import pytest

from canonical_units import expand_container_to_servable


# ---------------------------------------------------------------------------
# A) Helper SSOT en aislamiento
# ---------------------------------------------------------------------------

def test_egg_carton_with_uds_annotation_expands_to_units():
    """Caso real del owner: 2 cartones de 20 uds. → 40 huevos, no 2."""
    got = expand_container_to_servable("Huevo", 2, "2 cartones (20 uds.) de Huevo")
    assert got == (40.0, "unidad")


def test_egg_carton_singular_with_uds_annotation():
    got = expand_container_to_servable("Huevo", 1, "1 cartón (20 uds.) de Huevo")
    assert got == (20.0, "unidad")


def test_egg_carton_unit_column_only_no_name_suffix():
    """`_inventory_grams_ledger` pasa SOLO la columna `unit` cruda ("cartón (20 uds.)"),
    sin "de Huevo" — debe funcionar igual (la búsqueda es por keyword, no match exacto)."""
    got = expand_container_to_servable("Huevo", 2, "cartón (20 uds.)")
    assert got == (40.0, "unidad")


def test_egg_carton_without_annotation_uses_rd_default_15():
    """Sin "(N uds.)" en el string, la tabla canónica RD asume cartón=15 (mediana 12-30)."""
    got = expand_container_to_servable("Huevo", 1, "cartón")
    assert got == (15.0, "unidad")


def test_egg_carton_plural_without_annotation():
    got = expand_container_to_servable("Huevo", 2, "cartones")
    assert got == (30.0, "unidad")


def test_pote_with_kg_annotation_expands_to_grams():
    """El caso real del pote de yogurt: 1.96 kg → 1960 g."""
    got = expand_container_to_servable("Yogurt griego", 1, "pote (1.96 kg)")
    assert got == pytest.approx((1960.0, "g"))


def test_pote_with_unicode_fraction_lb_annotation():
    got = expand_container_to_servable("Queso blanco", 1, "paquete (½ lb)")
    assert got is not None
    qty, unit = got
    assert unit == "g"
    assert qty == pytest.approx(226.796, abs=0.01)


def test_grams_container_without_annotation_needs_opt_in():
    """Sin anotación de tamaño y `allow_category_fallback=False` (default) → fail-open (None):
    es EXACTAMENTE lo que `constants.validate_ingredients_against_pantry` necesita preservar
    (no inventar un peso para un envase 'pelado' — ese caller ya maneja ese caso él mismo)."""
    assert expand_container_to_servable("Pasta", 2, "paquete", allow_category_fallback=False) is None


def test_grams_container_without_annotation_opt_in_uses_master_container_weight_g():
    """Con `allow_category_fallback=True` (el modo de `_inventory_grams_ledger`) SÍ expande,
    usando `container_weight_g` del master si está poblado."""
    from nutrition_db import IngredientNutritionDB
    rows = [{
        "name": "Yogurt griego", "aliases": ["yogurt griego", "yogur griego"],
        "kcal_per_100g": 59, "protein_g_per_100g": 10, "carbs_g_per_100g": 3.6,
        "fats_g_per_100g": 0.4, "category": "Lácteos", "container_weight_g": 1960.0,
    }]
    db = IngredientNutritionDB(rows=rows)
    got = expand_container_to_servable("Yogurt griego", 1, "pote", allow_category_fallback=True, db=db)
    assert got == pytest.approx((1960.0, "g"))


def test_grams_container_category_fallback_when_no_container_weight_g():
    """Sin `container_weight_g` curado, cae al MISMO fallback conservador por categoría que ya
    usa el aggregator de la lista de compras (SSOT único, no un 3er número inventado aquí)."""
    from nutrition_db import IngredientNutritionDB
    rows = [{
        "name": "Pasta", "aliases": ["pasta"],
        "kcal_per_100g": 131, "protein_g_per_100g": 5, "carbs_g_per_100g": 25,
        "fats_g_per_100g": 1.1, "category": "Despensa",
    }]
    db = IngredientNutritionDB(rows=rows)
    got = expand_container_to_servable("Pasta", 1, "paquete", allow_category_fallback=True, db=db)
    assert got == pytest.approx((450.0, "g"))  # _fallback_container_weight_g('despensa')


def test_non_container_unit_returns_none():
    """Fail-open: unidades que NO son un envase reconocible → `None` siempre."""
    assert expand_container_to_servable("Huevo", 3, "unidad") is None
    assert expand_container_to_servable("Huevo", 100, "g") is None


def test_zero_or_negative_or_missing_qty_returns_none():
    assert expand_container_to_servable("Huevo", 0, "cartón (20 uds.)") is None
    assert expand_container_to_servable("Huevo", -1, "cartón (20 uds.)") is None
    assert expand_container_to_servable("Huevo", "no-numero", "cartón (20 uds.)") is None


def test_empty_unit_returns_none():
    assert expand_container_to_servable("Huevo", 2, "") is None
    assert expand_container_to_servable("Huevo", 2, None) is None


# ---------------------------------------------------------------------------
# B) E2E real: pantry guard del swap (constants.validate_ingredients_against_pantry)
# ---------------------------------------------------------------------------

def test_e2e_swap_guard_two_egg_cartons_covers_three_egg_dish():
    """Caso real del owner: 'Huevo: 2 cartón (20 uds.)' en la nevera. Antes del fix, el guard
    contaba 2 cartones como ~2 huevos sueltos y un plato de 3 huevos rebotaba con
    "límite: 2" (o "límite: 1" con 1 solo cartón) pese a haber ~40 unidades reales."""
    from constants import validate_ingredients_against_pantry
    generated = ["3 huevos"]
    pantry = ["2 cartones (20 uds.) de Huevo"]
    result = validate_ingredients_against_pantry(generated, pantry, strict_quantities=True)
    assert result is True, f"3 huevos sobre 2 cartones (40 uds.) NO debe rechazarse; guard devolvió: {result!r}"


def test_e2e_swap_guard_single_egg_carton_covers_three_egg_dish():
    """Espejo con 1 solo cartón (el escenario EXACTO del forense: "límite 1.0 unidad")."""
    from constants import validate_ingredients_against_pantry
    generated = ["3 huevos"]
    pantry = ["1 cartón (20 uds.) de Huevo"]
    result = validate_ingredients_against_pantry(generated, pantry, strict_quantities=True)
    assert result is True, f"3 huevos sobre 1 cartón (20 uds.) NO debe rechazarse; guard devolvió: {result!r}"


def test_e2e_swap_guard_still_rejects_genuinely_excessive_eggs():
    """El fix NO vuelve el guard permisivo: pedir más huevos que los disponibles sigue
    rechazando (40 unidades × 1.30 tolerancia = 26 techo; 50 excede)."""
    from constants import validate_ingredients_against_pantry
    generated = ["50 huevos"]
    pantry = ["1 cartón (20 uds.) de Huevo"]
    result = validate_ingredients_against_pantry(generated, pantry, strict_quantities=True)
    assert isinstance(result, str), "pedir 50 huevos sobre un cartón de 20 DEBE rechazarse"


# ---------------------------------------------------------------------------
# C) E2E real: ledger de regenerate-day (routers.plans._inventory_grams_ledger)
# ---------------------------------------------------------------------------

def test_e2e_regenerate_day_ledger_pote_yogurt_covers_150g_portion():
    """Caso real: 'pote (1.96 kg)' de yogurt en el inventario. Antes del fix,
    `_inventory_grams_ledger` no expandía NINGÚN envase — `db.to_grams` caía a 'unidad' sin
    `density_g_per_unit` → 0g disponibles con ~1.96kg reales, y una porción de 150g rebotaba
    por 'pantry_insufficient_for_goal'/`over_limit`."""
    from routers.plans import _inventory_grams_ledger
    from nutrition_db import IngredientNutritionDB

    rows_master = [{
        "name": "Yogurt griego", "aliases": ["yogurt griego", "yogur griego", "yogurt"],
        "kcal_per_100g": 59, "protein_g_per_100g": 10, "carbs_g_per_100g": 3.6,
        "fats_g_per_100g": 0.4, "category": "Lácteos", "container_weight_g": 1960.0,
    }]
    db = IngredientNutritionDB(rows=rows_master)
    inventory_rows = [{"ingredient_name": "Yogurt griego", "quantity": 1, "unit": "pote (1.96 kg)"}]

    ledger = _inventory_grams_ledger(inventory_rows, db)

    assert "Yogurt griego" in ledger
    assert ledger["Yogurt griego"] == pytest.approx(1960.0)
    # Una porción de 150g PASA: sobran ~1810g para el resto del día.
    assert ledger["Yogurt griego"] >= 150.0


def test_e2e_regenerate_day_ledger_egg_carton_covers_full_content():
    """Mismo caso de huevos, mirado desde el OTRO espejo (regenerate-day)."""
    from routers.plans import _inventory_grams_ledger
    from nutrition_db import IngredientNutritionDB

    rows_master = [{
        "name": "Huevo", "aliases": ["huevo", "huevos"],
        "kcal_per_100g": 155, "protein_g_per_100g": 13, "carbs_g_per_100g": 1.1,
        "fats_g_per_100g": 11, "category": "Proteínas", "density_g_per_unit": 50.0,
    }]
    db = IngredientNutritionDB(rows=rows_master)
    inventory_rows = [{"ingredient_name": "Huevo", "quantity": 2, "unit": "cartón (20 uds.)"}]

    ledger = _inventory_grams_ledger(inventory_rows, db)

    assert ledger.get("Huevo") == pytest.approx(2000.0)  # 40 huevos × 50g/huevo


# ---------------------------------------------------------------------------
# D) Fail-open: comportamiento previo intacto para items inexpandibles
# ---------------------------------------------------------------------------

def test_e2e_regenerate_day_ledger_fail_open_for_inexpandable_item():
    """Item cuya unidad no es un envase reconocible (ni antes ni después del fix) — el ledger
    simplemente no lo cuenta, comportamiento previo intacto."""
    from routers.plans import _inventory_grams_ledger
    from nutrition_db import IngredientNutritionDB

    rows_master = [{
        "name": "Especia rara", "aliases": ["especia rara"],
        "kcal_per_100g": 300, "protein_g_per_100g": 10, "carbs_g_per_100g": 50,
        "fats_g_per_100g": 5, "category": "Despensa",
    }]
    db = IngredientNutritionDB(rows=rows_master)
    inventory_rows = [{"ingredient_name": "Especia rara", "quantity": 3, "unit": "pizca"}]

    ledger = _inventory_grams_ledger(inventory_rows, db)
    assert "Especia rara" not in ledger


def test_e2e_swap_guard_casabe_unit_unaffected_by_fix():
    """Regresión ancla: el override físico de casabe (peso-por-unidad, no un envase) sigue
    intacto — este fix no toca esa ruta (unidad 'g' no es container-type)."""
    from constants import validate_ingredients_against_pantry
    generated = ["1 unidad de Casabe"]
    pantry = ["281 g de Casabe"]
    result = validate_ingredients_against_pantry(generated, pantry, strict_quantities=True)
    assert result is True, f"casabe 1 unidad no debe exceder el paquete; guard devolvió: {result!r}"
