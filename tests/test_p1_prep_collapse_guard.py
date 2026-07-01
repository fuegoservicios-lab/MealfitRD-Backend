"""[P1-PREP-COLLAPSE-GUARD · 2026-07-01] (audit creatividad G3/G4, confirmado en vivo)

Colapsos de PREPARACIÓN→producto-equivocado en los resolvers (generalización de P1-NUT-BUTTER-DISTINCT):
  - "harina de avena" → Harina de TRIGO (alias 'harina' ganaba el Tier-2 longest-first) — gluten en la
    lista de un celíaco sin que el allergen-guard dispare (el string no dice "trigo").
  - "harina de plátano" → Plátano verde fresco (~3× drift de kcal) vía Tier-2 y canonicalize_musaceae.
  - "tortilla de maíz" → Maíz dulce en granos (producto distinto; el catálogo solo tiene tortillas de trigo).

Fix: helper SSOT `resolve_preparation_distinct` (puro) compartido por normalize_name,
nutrition_db._match_row y los canonicalizers de víveres/musáceas.
"""
from __future__ import annotations

from pathlib import Path

from shopping_calculator import resolve_preparation_distinct, canonicalize_viveres, canonicalize_musaceae
from nutrition_db import IngredientNutritionDB

_BACKEND = Path(__file__).resolve().parent.parent
_SHOP = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
_NUTDB = (_BACKEND / "nutrition_db.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Helper SSOT (puro, sin catálogo)
# ---------------------------------------------------------------------------
def test_flour_with_catalog_equivalent_canonicalizes():
    assert resolve_preparation_distinct("harina de avena") == (True, "Avena")
    assert resolve_preparation_distinct("2 tazas de harina de maíz") == (True, "Harina de maíz precocida")
    assert resolve_preparation_distinct("harina de trigo") == (True, "Harina de trigo")


def test_distinct_preparations_do_not_collapse():
    for prep in ("harina de plátano", "harina de yuca", "harina de arroz", "harina de coco",
                 "harina de almendras", "tortilla de maíz", "crema de coco"):
        handled, canon = resolve_preparation_distinct(prep)
        assert handled is True and canon is None, (
            f"'{prep}' es un producto DISTINTO sin fila propia — debe marcarse (True, None), "
            f"dio ({handled}, {canon})"
        )


def test_non_preparations_fall_through():
    for name in ("harina de negrito", "arroz blanco", "plátano verde", "avena", "coco rallado", None, ""):
        handled, _ = resolve_preparation_distinct(name)
        assert handled is False, f"'{name}' debe seguir por los tiers normales"


# ---------------------------------------------------------------------------
# 2. Canonicalizers no colapsan preparaciones
# ---------------------------------------------------------------------------
def test_canonicalizers_skip_flour_preparations():
    assert canonicalize_viveres("harina de yuca") is None
    assert canonicalize_musaceae("harina de plátano") is None
    # controles positivos: el colapso legítimo de preparaciones FRESCAS sigue intacto.
    assert canonicalize_viveres("Yuca hervida") == "Yuca"
    assert canonicalize_musaceae("Plátano maduro frito") == "Plátano"


# ---------------------------------------------------------------------------
# 3. nutrition_db._match_row (rows inyectados → offline, determinista)
# ---------------------------------------------------------------------------
_ROWS = [
    {"name": "Harina de trigo", "aliases": ["harina"], "kcal_per_100g": 364,
     "protein_g_per_100g": 10, "carbs_g_per_100g": 76, "fats_g_per_100g": 1},
    {"name": "Avena", "aliases": ["avena"], "kcal_per_100g": 389,
     "protein_g_per_100g": 17, "carbs_g_per_100g": 66, "fats_g_per_100g": 7},
    {"name": "Plátano verde", "aliases": ["platano"], "kcal_per_100g": 122,
     "protein_g_per_100g": 1, "carbs_g_per_100g": 32, "fats_g_per_100g": 0},
]


def test_match_row_flour_of_oats_resolves_to_avena_not_trigo():
    db = IngredientNutritionDB(rows=_ROWS)
    row = db._match_row("harina de avena")
    assert row is not None and row["name"] == "Avena", (
        f"'harina de avena' debe computar macros de AVENA molida, no de trigo — dio {row}"
    )


def test_match_row_flour_of_platano_returns_none():
    db = IngredientNutritionDB(rows=_ROWS)
    assert db._match_row("harina de plátano") is None, (
        "'harina de plátano' NO debe computar macros del plátano fresco (~3× drift)"
    )


def test_match_row_plain_flour_still_resolves():
    db = IngredientNutritionDB(rows=_ROWS)
    row = db._match_row("harina de trigo")
    assert row is not None and row["name"] == "Harina de trigo"


# ---------------------------------------------------------------------------
# 4. Wiring estructural
# ---------------------------------------------------------------------------
def test_wired_in_normalize_name_and_match_row():
    assert "resolve_preparation_distinct(orig_name)" in _SHOP, \
        "normalize_name no consulta el guard de preparaciones"
    assert "resolve_preparation_distinct" in _NUTDB and "P1-PREP-COLLAPSE-GUARD" in _NUTDB, \
        "_match_row no consulta el guard de preparaciones"
    # el guard corre ANTES de los tiers de alias en ambos resolvers.
    i_guard = _SHOP.find("resolve_preparation_distinct(orig_name)")
    i_tier1 = _SHOP.find("INTENTO 1: Match Exacto")
    assert -1 < i_guard < i_tier1, "el guard debe correr ANTES del Tier-1 de normalize_name"
    assert "P1-PREP-COLLAPSE-GUARD" in _SHOP
