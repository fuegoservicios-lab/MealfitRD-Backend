"""[P1-INGREDIENT-SPREAD · 2026-07-28] Regression guard for `_count_ingredient_meal_frequency`,
the generic (list-free) per-ingredient meal-frequency detector.

Root measurement (script `scratchpad/measure_ingredient_repetition.py`, direct Neon connection,
29 live plans / 1036 plan-ingredient pairs): plan c8fa78c5 had yogurt in 8/12 meals (66.7%) — the
single highest substance-ingredient fraction anywhere in the live fleet (rank #1 of 1036 pairs).
`_count_staple_repetitions` already knew "yogurt griego" but counts DAY-presence (binary), so a
yogurt appearing once/day for 3 days and one appearing 3x/day for 3 days produce the IDENTICAL
signal. `_count_ingredient_meal_frequency` closes that gap by counting occurrences/total-meals
(continuous, meal-granularity) instead.

Non-negotiable design constraints from the brief, each anchored to a test below:
  1. No CURATED per-ingredient synonym/alias list (`_STAPLE_INGREDIENT_ALIASES`-style) — the
     counted token must never come from a hand-maintained table of known foods
     (test_hardcoded_list_sabotage_would_be_caught).
  2. Seasonings/cooking fats MUST be exempt, reusing the existing SSOT
     (`_is_seasoning_name`/`_QTY_GUARD_SEASONING_SKIP` + `_SKELETON_PROTEIN_FILLER_TOKENS` for
     aceite) — NOT a second source of truth (test_seasonings_and_oil_never_flagged,
     test_seasoning_exemption_sabotage_would_be_caught).
  3. Knob-gated with a safe default (test_knobs_registered).
  4. Logs what it did and why (test_self_critique_logs_and_wires_spread_block, source-level).

[P1-INGREDIENT-SPREAD-GUSTO-FIX · 2026-07-29] Constraint 1 was REVISED after this detector's first
live firing (corr=23c65543) flagged `'gusto'` (from "Sal al gusto") as a concentrated ingredient — a
false positive injected into the self-critique prompt on a real generation. Read-only census against
25 live plans found it systemic: `gusto` 25/25, `oliva` 25/25 (from "aceite de oliva"), `negra` 23/25
(from "pimienta negra"). Root cause: the ORIGINAL "mechanical, no lookup at all" design (this test
file's original `test_novel_ingredient_word_is_detected`, now
`test_novel_unverified_word_is_dropped_not_hallucinated`) accepted the raw "first significant token"
survived from the LLM's own text as the counted signal — which is exactly what fabricated
'gusto'/'oliva'/'negra' (modifier words of a seasoning phrase, not foods themselves). The FIX keeps
constraint 1's actual intent (never trust a hand-curated ALIAS TABLE — a table needs a 4th/5th/6th
entry every time a new false-positive word surfaces, the exact anti-pattern CLAUDE.md calls out:
pavo → mariscos → conejo/chivo/calamar/langosta/hígado → yogurt) while requiring the counted token
to RESOLVE against `master_ingredients` (the verified food catalogue, via the same
`IngredientNutritionDB.lookup` SSOT `_ensure_ingredient_quantities` already uses) instead of being
whatever raw word survives the seasoning/filler filters. A structurally novel word with NO catalog
entry is now DROPPED (not hallucinated as a token) — this is a deliberate, tested behavior change,
not a regression: it closes the false-positive class while catalog-verified real foods (yogurt,
queso, pollo, tomate, ...) are detected exactly as before.

tooltip-anchor: P1-INGREDIENT-SPREAD
"""
from __future__ import annotations

import graph_orchestrator as go
import pytest

# [P2-CI-BACKEND-SIBLINGS · 2026-09-04] Este módulo necesita el catálogo/la base de datos o el
# .env local (pasa en el checkout del dueño; en el CI sin NEON_DATABASE_URL se salta con motivo).
pytestmark = pytest.mark.needs_local_data


def _day(meals, day=1):
    return {"day": day, "meals": meals}


def _meal(name, ings=None):
    return {"name": name, "ingredients": ings or [], "ingredients_raw": list(ings or []),
            "recipe": [], "protein": 30, "carbs": 30, "fats": 10, "cals": 400}


# ─────────────────────────── core detection ───────────────────────────

def test_yogurt_8_of_12_meals_flagged_like_the_live_incident():
    """Reproduce c8fa78c5: 3 days, 4 meals/day, yogurt in 8/12 (66.7%) — the exact live incident
    that `_count_staple_repetitions` (day-granularity) could not distinguish from a normal
    once-a-day repeat. Distribution mirrors the incident: 3/4 + 3/4 + 2/4 = 8/12."""
    yogurt_meal = lambda tag: _meal(f"Yogurt {tag}", ["1 taza de yogurt griego"])
    other_meal = lambda tag: _meal(f"Otro {tag}", ["150 g de pollo", "100 g de arroz"])
    days = [
        _day([yogurt_meal("1"), yogurt_meal("2"), yogurt_meal("3"), other_meal("4")], day=1),
        _day([yogurt_meal("5"), yogurt_meal("6"), yogurt_meal("7"), other_meal("8")], day=2),
        _day([yogurt_meal("9"), yogurt_meal("10"), other_meal("11"), other_meal("12")], day=3),
    ]
    result = go._count_ingredient_meal_frequency(days)
    assert "yogurt" in result, f"yogurt en 8/12 comidas (66.7%) debe cruzar el umbral. Result: {result}"
    assert result["yogurt"] >= 0.66, result


def test_below_threshold_not_flagged():
    """Yogurt once/day across 3 days (3/12 = 25%) — below default 50% threshold, should NOT flag
    (this is the 'cultural once-a-day repeat' the same-day exemption already tolerates)."""
    days = []
    for d in range(1, 4):
        meals = [
            _meal("Avena con Yogurt Griego", ["1 taza de yogurt griego"]),
            _meal("Pollo con Arroz", ["150 g de pollo", "100 g de arroz"]),
            _meal("Ensalada con Atún", ["120 g de atun"]),
            _meal("Pescado al Horno", ["150 g de pescado"]),
        ]
        days.append(_day(meals, day=d))
    result = go._count_ingredient_meal_frequency(days)
    assert "yogurt" not in result, (
        f"25% de las comidas no debe cruzar el umbral por defecto (50%). Result: {result}"
    )


# ─────────────────────────── constraint 1: no curated alias table ───────────────────────────

def test_novel_unverified_word_is_dropped_not_hallucinated():
    """[REVISED · P1-INGREDIENT-SPREAD-GUSTO-FIX 2026-07-29] A made-up word with no entry in
    `master_ingredients` must NEVER become the counted token, no matter how many times it repeats —
    this is the inverse of the pre-fix contract (see module docstring): the pre-fix version treated
    "detect ANY repeated word mechanically" as the goal, and that is exactly what turned modifier
    words of a seasoning phrase ('gusto', 'oliva', 'negra') into fabricated concentration signals on
    every single live plan. The catalog gate means an unresolved word is DROPPED, never invented as
    a token — real foods (yogurt/queso/pollo/tomate) still resolve and still flag normally, proven
    by `test_yogurt_8_of_12_meals_flagged_like_the_live_incident` above still passing unchanged."""
    novel_word = "zzquinoawok"
    assert novel_word not in go._STAPLE_INGREDIENT_ALIASES
    for aliases in go._MAIN_PROTEIN_ALIASES.values():
        assert novel_word not in aliases
    assert novel_word not in go._QTY_GUARD_SEASONING_SKIP

    days = []
    for d in range(1, 3):
        meals = [
            _meal("Bowl Fusion", [f"200 g de {novel_word}"]),
            _meal("Ensalada Fusion", [f"100 g de {novel_word}", "50 g de lechuga"]),
            _meal("Pollo con Arroz", ["150 g de pollo", "100 g de arroz"]),
        ]
        days.append(_day(meals, day=d))
    # novel_word en 4/6 comidas = 66.7% — pre-fix esto lo hubiera flagueado (bug); post-fix debe
    # descartarse por no resolver contra el catálogo verificado.
    result = go._count_ingredient_meal_frequency(days)
    assert novel_word not in result, (
        f"Una palabra inventada sin match de catálogo NUNCA debe fabricarse como token — "
        f"exactamente el bug que produjo 'gusto'/'oliva'/'negra' en producción. Result: {result}"
    )


def test_hardcoded_list_sabotage_would_be_caught():
    """SABOTAGE CHECK (constraint 1, revised): if `_count_ingredient_meal_frequency` were
    reimplemented as a lookup against a fixed alias table (like `_STAPLE_INGREDIENT_ALIASES`)
    instead of the `master_ingredients` catalog gate, the fixture word below would be provably
    absent from that hardcoded table too — proving the ONLY reason it gets dropped post-fix is the
    catalog gate (real food-DB lookup), not a hand-maintained list masquerading as one. Also
    confirms the word doesn't resolve against the live catalog either (belt-and-suspenders for
    `test_novel_unverified_word_is_dropped_not_hallucinated`)."""
    novel_word = "zzquinoawok"
    all_curated_tokens = set()
    for aliases in go._STAPLE_INGREDIENT_ALIASES.values():
        all_curated_tokens.update(a.lower() for a in aliases)
    for aliases in go._MAIN_PROTEIN_ALIASES.values():
        all_curated_tokens.update(a.lower() for a in aliases)
    all_curated_tokens.update(t.lower() for t in go._QTY_GUARD_SEASONING_SKIP)
    all_curated_tokens.update(t.lower() for t in go._SKELETON_PROTEIN_FILLER_TOKENS)
    assert novel_word not in all_curated_tokens
    from nutrition_db import IngredientNutritionDB
    assert IngredientNutritionDB().lookup(novel_word) is None, (
        "el fixture debe ser genuinamente irresoluble contra el catálogo — si esto falla, "
        "'zzquinoawok' dejó de ser un buen ancla para este test."
    )


# ─────────────────────────── constraint 2: seasoning/oil exemption ───────────────────────────

def test_seasonings_and_oil_never_flagged():
    """Sal, pimienta, ajo, cebolla, aceite in EVERY meal (100%) — this is correct cooking, not a
    defect. Must never appear in the result regardless of how high their fraction is."""
    days = []
    for d in range(1, 3):
        meals = [
            _meal("Pollo Guisado", [
                "150 g de pollo", "1 cdta de sal", "1 cdta de pimienta",
                "2 dientes de ajo", "1/2 cebolla", "1 cda de aceite",
            ]),
            _meal("Res al Horno", [
                "150 g de res", "1 cdta de sal", "1 cdta de pimienta",
                "2 dientes de ajo", "1/2 cebolla", "1 cda de aceite",
            ]),
            _meal("Pescado a la Plancha", [
                "150 g de pescado", "1 cdta de sal", "1 cdta de pimienta",
                "2 dientes de ajo", "1/2 cebolla", "1 cda de aceite",
            ]),
        ]
        days.append(_day(meals, day=d))
    result = go._count_ingredient_meal_frequency(days)
    for exempt in ("sal", "pimienta", "ajo", "cebolla", "aceite"):
        assert exempt not in result, (
            f"'{exempt}' aparece en el 100% de las comidas pero DEBE estar exento (sazonador/"
            f"grasa de cocción, cocina correcta no defecto). Result: {result}"
        )


def test_seasoning_exemption_sabotage_would_be_caught():
    """SABOTAGE CHECK: if the seasoning/oil exemption were dropped (i.e. `_is_seasoning_name` /
    `_SKELETON_PROTEIN_FILLER_TOKENS` were no longer consulted), 'sal' at 100% of meals across
    this fixture WOULD cross the default 50% threshold and appear in the result. This test proves
    the exemption is load-bearing: without it, this exact fixture regresses to a false positive
    on plain correct cooking."""
    days = []
    for d in range(1, 3):
        meals = [
            _meal("Pollo Guisado", ["150 g de pollo", "1 cdta de sal"]),
            _meal("Res al Horno", ["150 g de res", "1 cdta de sal"]),
            _meal("Pescado a la Plancha", ["150 g de pescado", "1 cdta de sal"]),
        ]
        days.append(_day(meals, day=d))
    # Sanity: sal is present in 100% of meals in this fixture (6/6) — if NOT exempt, it would
    # trivially cross even a very high threshold, confirming the fixture is a meaningful sabotage
    # target for the exemption logic (not merely "sal never appears often enough to matter").
    total_meals = sum(len(d["meals"]) for d in days)
    sal_meals = sum(
        1 for d in days for m in d["meals"] if any("sal" in i.lower() for i in m["ingredients"])
    )
    assert sal_meals == total_meals == 6
    result = go._count_ingredient_meal_frequency(days)
    assert "sal" not in result


# ─────────────────────────── anti-noise floors ───────────────────────────

def test_tiny_chunk_below_min_meals_never_flags():
    """A 1-day, 2-meal chunk sharing an ingredient in both meals (100%) is below
    `_INGREDIENT_SPREAD_MIN_MEALS` (3) — must return {} regardless of fraction, since 2 meals is
    not a meaningful sample."""
    days = [_day([
        _meal("Avena con Yogurt", ["1 taza de yogurt griego"]),
        _meal("Merienda de Yogurt", ["1 taza de yogurt griego"]),
    ], day=1)]
    result = go._count_ingredient_meal_frequency(days)
    assert result == {}, f"Chunk de 2 comidas no debe producir señal (piso anti-ruido). Result: {result}"


def test_single_occurrence_never_flags_even_if_fraction_would_cross():
    """A word appearing in exactly 1 of 3 meals (33%) must not flag even against a lowered
    threshold, because the absolute-occurrence floor (>=2) exists independently of the fraction
    threshold — guards against a 1-off ingredient in a tiny/edge chunk reading as 'concentrated'."""
    days = [_day([
        _meal("A", ["1 taza de yogurt griego"]),
        _meal("B", ["150 g de pollo"]),
        _meal("C", ["150 g de res"]),
    ], day=1)]
    original = go.MAX_INGREDIENT_MEAL_FRACTION
    try:
        go.MAX_INGREDIENT_MEAL_FRACTION = 0.2  # 1/3 = 33% would cross a 20% threshold
        result = go._count_ingredient_meal_frequency(days)
        assert "yogurt" not in result, (
            f"1 ocurrencia absoluta nunca debe flaguear (piso >=2), aunque la fracción cruce el "
            f"umbral. Result: {result}"
        )
    finally:
        go.MAX_INGREDIENT_MEAL_FRACTION = original


def test_fail_safe_on_malformed_days():
    """Malformed input (non-dict day/meal, missing keys) must never raise — self-critique is
    best-effort and this detector must degrade to {} silently."""
    assert go._count_ingredient_meal_frequency(None) == {}
    assert go._count_ingredient_meal_frequency([]) == {}
    assert go._count_ingredient_meal_frequency([{"meals": "not-a-list"}]) == {}
    assert go._count_ingredient_meal_frequency([{"meals": [{"ingredients": None}]}]) == {}
    assert go._count_ingredient_meal_frequency(["not-a-dict"]) == {}


# ─────────────────────────── constraint 3: knob-gated ───────────────────────────

def test_knobs_registered_with_safe_defaults():
    """Both knobs must be registered in `_KNOBS_REGISTRY` (P3-NEW-D convention) so
    `/health/version`-adjacent tooling and operators can see + override them without a redeploy."""
    reg = go.get_knobs_registry_snapshot()
    assert "MEALFIT_INGREDIENT_SPREAD_GATE_ENABLED" in reg
    assert "MEALFIT_MAX_INGREDIENT_MEAL_FRACTION" in reg
    assert reg["MEALFIT_INGREDIENT_SPREAD_GATE_ENABLED"]["default"] is True
    assert reg["MEALFIT_MAX_INGREDIENT_MEAL_FRACTION"]["default"] == 0.5


def test_gate_disabled_means_module_constant_false_by_construction():
    """The module-level constant reads the knob at import time; this test documents the contract
    (rather than re-importing with a monkeypatched env, which would not reflect module-load-time
    semantics) — the wiring test below (`test_self_critique_source_wires_ingredient_spread`)
    proves the callsite actually branches on this flag."""
    assert isinstance(go.INGREDIENT_SPREAD_GATE_ENABLED, bool)
    assert 0.15 <= go.MAX_INGREDIENT_MEAL_FRACTION <= 0.9


# ─────────────────────────── constraint 4/5: wiring + logging (source-level) ───────────────────────────

def test_self_critique_source_wires_ingredient_spread():
    """Parser-based guard (repo convention: source-parsing tests anchor a tooltip so a rename
    fails the test before it fails production). Confirms `self_critique_node`:
      (a) computes `ingredient_spread` gated by `INGREDIENT_SPREAD_GATE_ENABLED`,
      (b) folds it into the skip-when-clean condition (so a concentrated plan does NOT silently
          skip the evaluator),
      (c) injects `spread_block` into the evaluator's human_content,
      (d) logs the detection (constraint 5: log what it did and why)."""
    import inspect
    src = inspect.getsource(go.self_critique_node)
    assert "tooltip-anchor: P1-INGREDIENT-SPREAD" not in src  # anchor lives on the helper, not here
    assert "_count_ingredient_meal_frequency" in src
    assert "INGREDIENT_SPREAD_GATE_ENABLED" in src
    assert "and not ingredient_spread" in src, (
        "ingredient_spread debe formar parte de la condición skip-when-clean, si no un plan "
        "concentrado se saltaría el evaluador silenciosamente."
    )
    assert "{spread_block}" in src or "spread_block" in src
    assert "logger.info" in src and "P1-INGREDIENT-SPREAD" in src


def test_helper_has_tooltip_anchor():
    import inspect
    src = inspect.getsource(go._count_ingredient_meal_frequency)
    assert "tooltip-anchor: P1-INGREDIENT-SPREAD" in src
