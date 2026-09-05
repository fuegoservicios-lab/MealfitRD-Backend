# -*- coding: utf-8 -*-
"""[P1-PANTRY-POOL-MATCH · 2026-09-05] Bloque 2 de la prueba B (plan c350dec0, mercado US + cocina DO, vegetariana),
entregado degradado tras 3 rechazos: el modo rotación reconocía la Nevera contra el catálogo DO y exigía el nombre
letra por letra en el pool US (2 bases, 0 proteínas), y el pool vegetariano de un mercado beta se quedaba en 2 proteínas."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import ai_helpers as ah  # noqa: E402
from constants import _get_fast_filtered_catalogs, strip_accents, DOMINICAN_CARBS, DOMINICAN_PROTEINS  # noqa: E402

NEVERA = ["Arroz integral", "Avena", "Habichuelas blancas", "Habichuelas rojas", "Lentejas", "Tortilla integral", "Queso blanco",
          "Queso cottage", "Yogurt", "Edamame", "Huevo", "Soya texturizada", "Papa", "Yuca", "Plátano", "Auyama"]


def _picks(items, full, syn, allowed):
    return {it: ah._pantry_pick_in_pool(strip_accents(it.lower()), full, syn, set(allowed)) for it in items}


def test_pantry_matches_the_market_pool_names():
    fp, fc, _, _ = _get_fast_filtered_catalogs((), (), "vegetarian", country="US", market_extras=True, culture_country="DO")
    carbs = {k: v for k, v in _picks(NEVERA, DOMINICAN_CARBS, ah.carb_synonyms, fc).items() if v}
    assert carbs["Papa"] == "Papa" and carbs["Habichuelas rojas"] == "Habichuelas rojas" and carbs["Lentejas"] == "Lentejas"
    assert carbs["Yuca"] == "Yuca" and carbs["Avena"] == "Avena"
    assert len(carbs) >= 5, carbs
    prots = {k: v for k, v in _picks(NEVERA, DOMINICAN_PROTEINS, ah.protein_synonyms, fp).items() if v}
    assert prots["Huevo"] in ("Huevo", "Huevos")
    assert len(prots) >= 5, prots   # antes: solo Huevo (y el pool tenía 2)


def test_do_market_is_unchanged_by_the_pool_first_lookup():
    fp, fc, _, _ = _get_fast_filtered_catalogs((), (), "balanced", country="DO")
    carbs = _picks(["Papa", "Yuca", "Plátano"], DOMINICAN_CARBS, ah.carb_synonyms, fc)
    legacy = {it: ah._catalog_pick_wb(strip_accents(it.lower()), DOMINICAN_CARBS, ah.carb_synonyms, set(fc)) for it in carbs}
    assert carbs == legacy


def test_vegetarian_and_vegan_pools_have_a_protein_floor_in_beta_markets():
    fp_veg = _get_fast_filtered_catalogs((), (), "vegetarian", country="US", market_extras=True, culture_country="DO")[0]
    assert len(fp_veg) >= 6, fp_veg
    low = " ".join(fp_veg).lower()
    assert "lentejas" in low and "queso" in low and "huevo" in low
    assert not any(t in low for t in ("pollo", "res", "pescado", "atun", "atún", "cerdo", "camaron"))
    fp_vegan = _get_fast_filtered_catalogs((), (), "vegan", country="US", market_extras=True, culture_country="DO")[0]
    assert len(fp_vegan) >= 3, fp_vegan
    lowv = " ".join(fp_vegan).lower()
    assert "lentejas" in lowv and "queso" not in lowv and "huevo" not in lowv and "yogur" not in lowv
    # omnívoro: sin cambios (el piso no aplica)
    fp_bal = _get_fast_filtered_catalogs((), (), "balanced", country="US", market_extras=True, culture_country="DO")[0]
    assert "Pechuga de pollo" in fp_bal and not any("lenteja" in x.lower() for x in fp_bal)


def test_anchor():
    assert "tooltip-anchor: P1-PANTRY-POOL-MATCH" in (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert "BETA_VEG_PROTEIN_FLOOR" in (_BACKEND / "constants.py").read_text(encoding="utf-8")
