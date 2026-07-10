"""[P2-MICRO-DENSITY-QTYLESS · 2026-07-10] Cierra los dos gaps observados en prod (18×
P1-MICRO-DENSITY-OBSERVABLE en 6h) que descartaban micros de ingredientes RESUELTOS:

1. "Cdta de miel (opcional)" — string SIN número líder pero que EMPIEZA con unidad conocida.
   El parser retornaba qty=0.0 → to_grams None → micros descartados. En español "Cdta de miel"
   significa 1 cdta: default qty=1.0 cuando la unidad lidera sin número.
2. "Yogurt griego sin azúcar" — string nombre-solo (sin qty ni unidad). Solo para MICROS
   (no toca el path del solver/macros): si el row tiene density_g_per_unit, asumir 1 unidad.

Offline (rows inyectados). tooltip-anchor: P2-MICRO-DENSITY-QTYLESS
"""
import nutrition_db as ndb

_ROWS = [
    {"name": "Miel", "aliases": [], "kcal_per_100g": 304, "protein_g_per_100g": 0.3,
     "carbs_g_per_100g": 82, "fats_g_per_100g": 0, "density_g_per_cup": 340,
     "density_g_per_unit": None, "sodium_mg_per_100g": 4, "fiber_g_per_100g": 0.2},
    {"name": "Yogurt griego sin azúcar", "aliases": [], "kcal_per_100g": 59,
     "protein_g_per_100g": 10, "carbs_g_per_100g": 3.6, "fats_g_per_100g": 0.4,
     "density_g_per_cup": 245, "density_g_per_unit": 170, "sodium_mg_per_100g": 36,
     "fiber_g_per_100g": 0},
    {"name": "Sal", "aliases": [], "kcal_per_100g": 0, "protein_g_per_100g": 0,
     "carbs_g_per_100g": 0, "fats_g_per_100g": 0, "density_g_per_cup": None,
     "density_g_per_unit": None, "sodium_mg_per_100g": 38758, "fiber_g_per_100g": 0},
]


def _db():
    return ndb.IngredientNutritionDB(rows=_ROWS)


def test_unit_leading_string_defaults_to_qty_1():
    qty, unit, name = ndb._split_qty_unit_name("Cdta de miel (opcional)")
    assert qty == 1.0
    assert unit.lower() in ("cdta", "cucharadita")
    assert "miel" in name.lower()


def test_cdta_de_miel_now_yields_grams():
    g = _db().grams_from_ingredient_string("Cdta de miel (opcional)")
    assert g is not None and 5.0 <= g <= 9.0     # 1 cdta ≈ 4.93ml × (340/240) ≈ 7g


def test_bare_name_with_unit_density_yields_micros():
    m = _db().micros_from_ingredient_string("Yogurt griego sin azúcar")
    assert m is not None                          # 1 unidad × 170g (density_g_per_unit)
    assert 150 <= m["grams"] <= 190


def test_bare_name_without_unit_density_still_none():
    # sin density_g_per_unit no se adivina (comportamiento previo intacto)
    assert _db().micros_from_ingredient_string("Miel") is None


def test_sal_al_gusto_unchanged():
    qty, unit, _ = ndb._split_qty_unit_name("Sal al gusto")
    assert qty == 0.0                              # sin unidad líder → sin default
    assert _db().grams_from_ingredient_string("Sal al gusto") is None


def test_numeric_strings_unchanged():
    qty, unit, name = ndb._split_qty_unit_name("2 tazas de avena")
    assert qty == 2.0 and unit == "tazas" and "avena" in name.lower()
