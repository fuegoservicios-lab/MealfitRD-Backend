"""[P4-UNIFIED-RESOLVER · 2026-06-14] Resolver de ingredientes UNIFICADO (fuzzy difflib + Cohere v4
semántico) → mata el "0 silencioso" #1: ingredientes del plan que NO resolvían al catálogo aportaban
0 macros al solver. Antes había DOS resolvers: `shopping_calculator.normalize_name` (5 tiers + Cohere)
para la lista de compras, y `nutrition_db._match_row` (2 tiers baratos, SIN semántico) para los macros.
Ahora nutrition_db delega sus misses a normalize_name (un solo resolver), y normalize_name gana un tier
FUZZY (difflib) antes del embedding. Tests con catálogo mockeado (sin Cohere) → deterministas.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

CATALOG = [
    {"name": "Plátano maduro", "aliases": ["platano maduro", "amarillo"],
     "kcal_per_100g": 122, "protein_g_per_100g": 1.3, "carbs_g_per_100g": 32, "fats_g_per_100g": 0.4},
    {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"],
     "kcal_per_100g": 165, "protein_g_per_100g": 31, "carbs_g_per_100g": 0, "fats_g_per_100g": 3.6},
    {"name": "Naranja", "aliases": ["china"],
     "kcal_per_100g": 47, "protein_g_per_100g": 0.9, "carbs_g_per_100g": 12, "fats_g_per_100g": 0.1},
]


@pytest.fixture(autouse=True)
def _mock_catalog(monkeypatch):
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: [dict(r) for r in CATALOG])
    # sin Cohere → el tier semántico se salta; los tests dependen del fuzzy (determinista)
    monkeypatch.setattr(sc, "get_semantic_cache", lambda: None)
    monkeypatch.setenv("MEALFIT_NUTRITION_UNIFIED_RESOLVER", "true")


# ─────────────────────────────────────────────────────────────────────────────
# Tier FUZZY en normalize_name (atrapa typos antes del embedding)
# ─────────────────────────────────────────────────────────────────────────────
def test_normalize_name_fuzzy_matches_typo():
    import shopping_calculator as sc
    assert sc.normalize_name("platanno maduro") == "Plátano maduro"   # typo (doble n)
    assert sc.normalize_name("pechuga de poyo") == "Pechuga de pollo"  # typo (poyo)


def test_normalize_name_fuzzy_is_conservative():
    """No debe fuzzy-matchear cosas suficientemente distintas (umbral 0.87)."""
    import shopping_calculator as sc
    # "espinaca" no está en el catálogo y no se parece a nada → devuelve el string limpio, NO un master
    out = sc.normalize_name("espinaca cruda")
    assert out not in {"Plátano maduro", "Pechuga de pollo", "Naranja"}


def test_normalize_name_exact_still_wins():
    import shopping_calculator as sc
    assert sc.normalize_name("pollo") == "Pechuga de pollo"          # alias exacto, no fuzzy
    assert sc.normalize_name("Plátano maduro") == "Plátano maduro"   # nombre canónico exacto


# ─────────────────────────────────────────────────────────────────────────────
# nutrition_db delega a normalize_name → resuelve lo que sus tiers baratos no
# ─────────────────────────────────────────────────────────────────────────────
def test_nutrition_db_resolves_typo_via_delegation():
    import nutrition_db as ndb
    db = ndb.IngredientNutritionDB()   # rows=None → catálogo real (mockeado) + delegación activa
    info = db.lookup("platanno maduro")   # cheap tiers fallan; normalize_name fuzzy lo resuelve
    assert info is not None
    assert info.name == "Plátano maduro"
    assert info.kcal == 122


def test_nutrition_db_cheap_tier_still_works_without_delegation():
    import nutrition_db as ndb
    db = ndb.IngredientNutritionDB()
    info = db.lookup("pollo")   # alias exacto → tier barato, sin tocar normalize_name
    assert info is not None and info.name == "Pechuga de pollo"


def test_unified_resolver_knob_off_disables_delegation(monkeypatch):
    monkeypatch.setenv("MEALFIT_NUTRITION_UNIFIED_RESOLVER", "off")
    import importlib
    import nutrition_db as ndb
    importlib.reload(ndb)   # re-lee el knob del env
    try:
        db = ndb.IngredientNutritionDB()
        # con el knob OFF, el typo NO se resuelve (vuelve al comportamiento de 2 tiers)
        assert db.lookup("platanno maduro") is None
        # pero el alias exacto sigue resolviendo
        assert db.lookup("pollo") is not None
    finally:
        monkeypatch.setenv("MEALFIT_NUTRITION_UNIFIED_RESOLVER", "true")
        importlib.reload(ndb)


def test_injected_rows_skip_delegation():
    """Con rows inyectados (test offline) la delegación se salta — determinismo, sin catálogo externo."""
    import nutrition_db as ndb
    db = ndb.IngredientNutritionDB(rows=[dict(CATALOG[0])])   # _injected=True
    assert db.lookup("plátano maduro") is not None        # tier barato
    assert db.lookup("platanno maduro") is None           # delegación saltada → no fuzzy
