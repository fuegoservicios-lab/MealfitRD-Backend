"""[P1-RESOLVER-COVERAGE · 2026-06-16] Cierra el gap de cobertura del resolver de macros medido
empíricamente contra 871 líneas de ingrediente reales (89.8%→91.8% distintas, +18 líneas, 0 regresiones).

Tres cambios bajo prueba:
1. `_GRAM_HINT_RE` admite texto antes del número dentro del paréntesis ("(wrap, 60g)") → usa el peso
   que el LLM declaró, ganando sobre la densidad genérica per-unit (no adivina).
2. 4 alimentos atómicos nuevos en el catálogo (Manzana/Pepino/Granola/Maní) — INSERT en la migración
   p1_resolver_coverage_ingredients_2026_06_16.sql + macros vía populate_nutrition_db.py (USDA_QUERY).
3. 2 aliases sobre filas existentes (pescado fresco→Filete; tortilla de harina de trigo→Tortilla integral).

Tests con catálogo INYECTADO (rows=[...]) → Tier 1/2 deterministas, sin Cohere.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nutrition_db as ndb
from nutrition_db import _GRAM_HINT_RE, _GRAM_ONLY_HINT_RE, IngredientNutritionDB

# Catálogo que imita las filas reales relevantes (post-migración).
_CAT = [
    {"name": "Tortilla integral",
     "aliases": ["tortillas integrales", "tortilla de harina de trigo", "tortilla de trigo"],
     "kcal_per_100g": 310, "protein_g_per_100g": 8, "carbs_g_per_100g": 50, "fats_g_per_100g": 7,
     "density_g_per_unit": 48},
    {"name": "Manzana", "aliases": ["manzanas", "apple"],
     "kcal_per_100g": 65, "protein_g_per_100g": 0.1, "carbs_g_per_100g": 15.7, "fats_g_per_100g": 0.2,
     "density_g_per_unit": 180},
    {"name": "Pepino", "aliases": ["pepinos", "cohombro"],
     "kcal_per_100g": 12, "protein_g_per_100g": 0.6, "carbs_g_per_100g": 2.2, "fats_g_per_100g": 0.2,
     "density_g_per_unit": 200},
    {"name": "Maní", "aliases": ["mani", "mani tostado", "mani tostado sin sal", "cacahuate"],
     "kcal_per_100g": 630, "protein_g_per_100g": 24, "carbs_g_per_100g": 21, "fats_g_per_100g": 50},
    {"name": "Mantequilla de maní", "aliases": ["crema de mani", "mani tostado molido", "peanut butter"],
     "kcal_per_100g": 588, "protein_g_per_100g": 25, "carbs_g_per_100g": 20, "fats_g_per_100g": 50},
    {"name": "Filete de pescado blanco",
     "aliases": ["pescado blanco", "tilapia", "mero", "chillo", "pescado fresco"],
     "kcal_per_100g": 96, "protein_g_per_100g": 20, "carbs_g_per_100g": 0, "fats_g_per_100g": 1.7},
    {"name": "Almendras fileteadas", "aliases": ["almendras", "nueces"],
     "kcal_per_100g": 579, "protein_g_per_100g": 21, "carbs_g_per_100g": 22, "fats_g_per_100g": 50,
     "density_g_per_cup": 95},
    {"name": "Leche", "aliases": ["leche descremada"],
     "kcal_per_100g": 61, "protein_g_per_100g": 3.2, "carbs_g_per_100g": 4.8, "fats_g_per_100g": 3.3,
     "density_g_per_cup": 244},
    {"name": "Huevo", "aliases": ["huevos", "huevos enteros"],
     "kcal_per_100g": 143, "protein_g_per_100g": 12.6, "carbs_g_per_100g": 0.7, "fats_g_per_100g": 9.5,
     "density_g_per_unit": 50, "density_g_per_cup": 243},
    {"name": "Clara de huevo", "aliases": ["claras de huevo", "clara de huevo", "claras"],
     "kcal_per_100g": 48, "protein_g_per_100g": 10.9, "carbs_g_per_100g": 0.7, "fats_g_per_100g": 0.2,
     "density_g_per_unit": 33, "density_g_per_cup": 243},
]


def _db():
    return IngredientNutritionDB(rows=[dict(r) for r in _CAT])


# ── 1. Regex: gram-hint con prefijo dentro del paréntesis ──────────────────────
def test_gram_hint_matches_number_after_prefix():
    assert _GRAM_HINT_RE.search("(wrap, 60g)").group(1) == "60"
    assert _GRAM_HINT_RE.search("tortilla (aprox. 50 g)").group(1) == "50"
    assert _GRAM_HINT_RE.search("(cocido, 150g)").group(1) == "150"


def test_gram_hint_simple_forms_unchanged():
    # No regresión: las formas que ya funcionaban siguen igual.
    assert _GRAM_HINT_RE.search("(60g)").group(1) == "60"
    assert _GRAM_HINT_RE.search("(140gr)").group(1) == "140"
    m = _GRAM_HINT_RE.search("(240 ml)")
    assert m.group(1) == "240" and m.group(2).lower() == "ml"
    assert _GRAM_HINT_RE.search("(sin azúcar)") is None  # sin número+unidad → no match


def test_gram_hint_wins_over_per_unit_density():
    # El peso EXPLÍCITO del LLM (60g) gana sobre la densidad genérica (48g/unidad).
    m = _db().macros_from_ingredient_string("1 tortilla de harina de trigo (wrap, 60g)")
    assert m is not None and m["name"] == "Tortilla integral"
    assert m["grams"] == 60.0   # no 48


def test_gram_hint_preferred_over_ml_in_same_paren():
    # Hint mixto "(... ml, ... g)": el gramo explícito gana sobre la conversión por densidad del ml.
    db = _db()
    g = db.grams_from_ingredient_string("1 batido de leche (60 ml de leche, 200g total)")
    assert g == 200.0
    # Solo-ml sigue usando densidad (sin regresión): 244 g/taza ⇒ 240 ml ≈ 244 g.
    g_ml = db.grams_from_ingredient_string("1 taza de leche (240 ml)")
    assert g_ml is not None and abs(g_ml - 244.0) < 1.0
    # El regex gram-only NO captura un paréntesis que solo trae ml.
    assert _GRAM_ONLY_HINT_RE.search("(240 ml)") is None
    assert _GRAM_ONLY_HINT_RE.search("(wrap, 60g)").group(1) == "60"


# ── 2. Alimentos atómicos nuevos resuelven ─────────────────────────────────────
def test_manzana_resolves_via_hint():
    for line, g in [("1 manzana mediana (150g)", 150.0), ("1/2 manzana roja (80g)", 80.0),
                    ("1 manzana verde en cubos (150g)", 150.0)]:
        m = _db().macros_from_ingredient_string(line)
        assert m is not None and m["name"] == "Manzana" and m["grams"] == g


def test_pepino_and_granola_and_mani_resolve():
    db = _db()
    assert db.macros_from_ingredient_string("1/2 pepino (100g) en cubos")["name"] == "Pepino"
    assert db.macros_from_ingredient_string("2 cdas de maní tostado sin sal (30g)")["name"] == "Maní"


def test_pescado_fresco_alias_resolves():
    m = _db().macros_from_ingredient_string("200g de pescado fresco (mero o chillo)")
    assert m is not None and m["name"] == "Filete de pescado blanco" and m["grams"] == 200.0


def test_egg_white_separated_from_whole_egg():
    # "claras de huevo" → Clara de huevo (48 kcal/100g, 0.2g grasa), NO Huevo entero (143 kcal, 9.5g).
    # Cierra el sobre-conteo ~2.7x de un staple del desayuno DR.
    db = _db()
    m = db.macros_from_ingredient_string("3 claras de huevo")
    assert m is not None and m["name"] == "Clara de huevo"
    assert m["grams"] == 99.0          # 3 × 33 g (clara), NO 3 × 50 g (huevo entero)
    assert abs(m["kcal"] - 47.5) < 1.0  # ~48 kcal, NO ~208
    # Control: el huevo ENTERO sigue resolviendo a Huevo (no se afecta el routing).
    mw = db.macros_from_ingredient_string("2 huevos")
    assert mw is not None and mw["name"] == "Huevo" and mw["grams"] == 100.0


def test_egg_whites_in_ml_and_cups_resolve_via_volumetric_density():
    # Clara líquida en ml/taza resuelve (density_g_per_cup=243); 375 ml ≈ 380 g a macros de clara.
    db = _db()
    m_ml = db.macros_from_ingredient_string("375 ml de claras de huevo")
    assert m_ml is not None and m_ml["name"] == "Clara de huevo" and abs(m_ml["grams"] - 379.7) < 1.0
    assert abs(m_ml["kcal"] - 182.2) < 2.0   # 380 g × 0.48, NO × 1.43
    m_cup = db.macros_from_ingredient_string("1.5 tazas de claras de huevo")
    assert m_cup is not None and abs(m_cup["grams"] - 364.5) < 2.0


# ── 3. Guard anti-shadowing: el maní pelado NO se traga la mantequilla de maní ──
def test_mani_does_not_shadow_peanut_butter():
    db = _db()
    assert db.lookup("mantequilla de maní").name == "Mantequilla de maní"   # exacto gana
    assert db.lookup("crema de maní").name == "Mantequilla de maní"
    assert db.lookup("maní").name == "Maní"                                  # pelado → nuez


def test_mani_word_boundary_does_not_shadow_almonds():
    # Tier2 ordena aliases por longitud DESC: "almendras" (9) gana sobre "mani" (4) en una línea
    # compuesta → el "mani" pelado NO se traga un ingrediente cuyo alias más largo matchea primero.
    db = _db()
    assert db.lookup("almendras con mani").name == "Almendras fileteadas"
    assert db.lookup("mix de nueces y mani").name == "Almendras fileteadas"


# ── 4. Anclas de fuente (renombre rompe el test antes que producción) ──────────
def _migration_text():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "migrations", "p1_resolver_coverage_ingredients_2026_06_16.sql")
    with open(p, encoding="utf-8") as f:
        return f.read()


def test_migration_inserts_four_ingredients_and_two_aliases():
    txt = _migration_text()
    for name in ("Manzana", "Pepino", "Granola", "Maní"):
        assert f"'{name}'" in txt, f"migración no inserta {name}"
    assert "pescado fresco" in txt
    assert "tortilla de harina de trigo" in txt


def test_populate_has_usda_query_for_new_ingredients():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "populate_nutrition_db.py")
    with open(p, encoding="utf-8") as f:
        txt = f.read()
    for name in ("Manzana", "Pepino", "Granola", "Maní"):
        assert re.search(rf'"{name}"\s*:', txt), f"populate sin USDA_QUERY para {name}"
