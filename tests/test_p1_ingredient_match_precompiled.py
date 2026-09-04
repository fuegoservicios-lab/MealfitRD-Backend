"""[P1-INGREDIENT-MATCH-PRECOMPILED · 2026-09-04] `IngredientNutritionDB._match_row` recompilaba
~700 regex `\\b<alias>\\b` en CADA lookup (el caché de `re` tiene 512 entradas: thrash). Perfil con el
plan real del dueño: el cerrador de micros de un swap persistido = 1.418 lookups → 751.727
compilaciones → 66 s de CPU (103 s con cProfile); en prod ≈ 33 s bajo el FOR UPDATE del persist,
más que el propio LLM (20 s). Fix: patrones compilados UNA vez por índice (mismo orden: largos
primero) + memo de `_match_row` por instancia. Misma semántica, mismos tiers, mismo orden.
"""
from __future__ import annotations

import re
import time

import pytest


def _db(rows):
    from nutrition_db import IngredientNutritionDB
    return IngredientNutritionDB(rows=rows)


_ROWS = [
    {"name": "Plátano", "aliases": ["platano verde", "guineo verde"], "kcal_per_100g": 122, "protein_g_per_100g": 1.3, "carbs_g_per_100g": 32, "fats_g_per_100g": 0.4},
    {"name": "Plátano maduro", "aliases": ["maduro"], "kcal_per_100g": 122, "protein_g_per_100g": 1.3, "carbs_g_per_100g": 32, "fats_g_per_100g": 0.4},
    {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 107, "protein_g_per_100g": 22, "carbs_g_per_100g": 0, "fats_g_per_100g": 2},
    {"name": "Ajonjolí", "aliases": ["sesamo"], "kcal_per_100g": 573, "protein_g_per_100g": 17, "carbs_g_per_100g": 23, "fats_g_per_100g": 50},
]


def test_index_precompiles_patterns_in_the_same_order_as_aliases():
    db = _db(_ROWS)
    assert [a for a, _r in db._aliases if a] == [a for a, _p, _r in db._alias_patterns]
    assert all(isinstance(p, re.Pattern) for _a, p, _r in db._alias_patterns)
    # largos primero: 'platano maduro' antes que 'platano'
    names = [a for a, _p, _r in db._alias_patterns]
    assert names.index("platano maduro") < names.index("platano")


def test_match_semantics_unchanged_versus_naive_reference():
    db = _db(_ROWS)
    from nutrition_db import _strip_accents

    def naive(raw):
        n = re.sub(r"\(.*?\)", "", str(raw).lower()).strip()
        ns = _strip_accents(n)
        for a, r in db._aliases:
            if ns == a:
                return r
        for a, r in db._aliases:
            if a and re.search(r"\b" + re.escape(a) + r"\b", ns):
                return r
        return None

    for raw in ["2 platanos maduros", "Plátano maduro", "150 g de pechuga de pollo", "ajonjolí tostado",
                "guineo verde hervido", "pollo guisado (sin piel)", "sésamo", "algo que no existe"]:
        got = db._match_row(raw)
        assert (got or {}).get("name") == (naive(raw) or {}).get("name"), raw


def test_memo_serves_repeated_lookups_without_rematching(monkeypatch):
    db = _db(_ROWS)
    calls = {"n": 0}
    orig = db._match_row_uncached

    def counting(raw):
        calls["n"] += 1
        return orig(raw)

    monkeypatch.setattr(db, "_match_row_uncached", counting)
    for _ in range(5):
        assert db._match_row("150 g de pechuga de pollo")["name"] == "Pechuga de pollo"
        assert db._match_row("no existe") is None
    assert calls["n"] == 2  # una vez por string distinto (None también se memoiza)
    assert "150 g de pechuga de pollo" in db._match_memo


def test_thousand_lookups_are_fast():
    rows = [{"name": f"Alimento {i}", "aliases": [f"alias{i} largo", f"al{i}"], "kcal_per_100g": 100,
             "protein_g_per_100g": 1, "carbs_g_per_100g": 1, "fats_g_per_100g": 1} for i in range(400)]
    db = _db(rows)
    t0 = time.perf_counter()
    for i in range(1000):
        db._match_row(f"{i % 50} g de alias{i % 400} largo cocido")
    dt = time.perf_counter() - t0
    assert dt < 2.0, f"1000 lookups tardaron {dt:.2f}s"


def test_marker_present():
    from pathlib import Path
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "P1-INGREDIENT-MATCH-PRECOMPILED · 2026-09-04" in app
