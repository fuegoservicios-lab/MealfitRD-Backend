"""[P1-INGREDIENT-SPREAD-GUSTO-FIX · 2026-07-29] `_count_ingredient_meal_frequency`'s first live
firing (corr=23c65543) flagged `'gusto'` (de "Sal al gusto") as a concentrated ingredient — a false
positive injected into the self-critique prompt on a real generation. Read-only census against the
25 most recent live plans (script `measure_gusto_fix.py`, Neon, no writes/no LLM) confirmed it was
systemic, not a fluke: `gusto` 25/25 (100%), `oliva` 25/25 (100%, de "aceite de oliva"), `negra` 23/25
(92%, de "pimienta negra"). Root cause: `_is_seasoning_name` is designed to receive the WHOLE bare
ingredient string (its other 2 callsites in this file, P2-QTY-PRESENCE-GUARD /
P1-SEASONING-WORD-BOUNDARY, call it that way) — this detector called it PER-TOKEN after `.split()`,
so 'sal'/'pimienta' (the actual seasoning root) got exempted but 'gusto'/'negra' (modifiers of the
SAME phrase, not seasonings themselves) survived. 'aceite' has an identical problem via the separate
`_SKELETON_PROTEIN_FILLER_TOKENS` set: filtering just the word 'aceite' left 'oliva' as the
"first significant token" survivor.

Fix, without a growing per-incident stopword list (CLAUDE.md explicitly forbids that pattern — this
repo already burned itself 4× the same way: pavo/mariscos/conejo/yogurt):
  1. Whole-PHRASE exemption: `_is_seasoning_name(bare_norm)` on the full bare string (same contract
     as its other 2 callsites) + an analogous whole-phrase check for the 'aceite' root (cooking oil
     is functionally a condiment for this detector regardless of variety — 'agua'/'lata'/'latas'
     stay token-level since they're packaging descriptors, not the food category, and dropping them
     entirely would hide real over-repetition like canned tuna).
  2. Catalog gate: the counted token is now the first significant word of the CANONICAL name
     resolved via `IngredientNutritionDB.lookup` (same SSOT `_ensure_ingredient_quantities` already
     uses) — never raw LLM text. Unmatched ingredient lines are DROPPED, not hallucinated into a
     token (`_dropped_unmatched` in the source logs the count for observability).

Post-fix re-run of the SAME 25-plan census: `gusto`/`oliva`/`negra` at 0/25. Surviving signals
(queso 5/25, yogurt 4/25, tomate 4/25, miel 2/25) are all real foods, consistent with the pre-fix
baseline for genuine concentration (queso 6/25, yogurt 3/25 per the incident report) — the detector
still catches what it was built to catch.
"""
from __future__ import annotations

import pathlib

import nutrition_db
import graph_orchestrator as g


class _FakeInfo:
    def __init__(self, name):
        self.name = name


class _FakeCatalogDB:
    """Stand-in for `IngredientNutritionDB` — deterministic, offline, no Neon.
    Only resolves the handful of foods this test cares about; everything else is
    None (unresolved), mirroring an ingredient absent from `master_ingredients`."""

    def __init__(self, *a, **kw):
        pass

    def lookup(self, raw_name):
        low = str(raw_name).lower()
        if "pollo" in low:
            return _FakeInfo("Pollo")
        if "queso" in low:
            return _FakeInfo("Queso blanco")
        # [P1-INGREDIENT-SPREAD-GUSTO-FIX] Mirrors what the REAL master_ingredients catalog does in
        # production: "pimienta negra al gusto" DOES resolve (unlike "sal al gusto", which misses
        # the real catalog entirely) to a canonical row whose name still contains 'negra' as a
        # significant word. Live-data sabotage (capa 1 disabled, re-run against 25 real plans)
        # showed 'negra' resurfacing at 23/25 (92%) specifically BECAUSE of this catalog-resolution
        # path — the catalog gate (capa 2) alone does NOT save 'negra'; only the whole-phrase
        # seasoning exemption (capa 1) does. Without this fixture entry, this test file could not
        # catch a regression that removed capa 1 (its absence would be silently masked by capa 2
        # for every OTHER fixture ingredient, since none of them resolve in the fake catalog).
        if "pimienta negra" in low:
            return _FakeInfo("Pimienta negra")
        # Mirrors production: "aceite de oliva" DOES resolve in `master_ingredients` (it's a
        # purchasable catalog item) to a canonical name that still contains 'oliva' as a
        # significant word — this is exactly why 'oliva' fired 25/25 in the live census until the
        # dedicated whole-phrase 'aceite' exemption was added (the catalog gate alone does not
        # save it, symmetric to 'negra'/'pimienta negra' above).
        if "aceite de oliva" in low:
            return _FakeInfo("Aceite de oliva")
        return None


def _meal(name, ingredients):
    return {"meal": name, "ingredients": ingredients}


def _days_fixture():
    # 4 comidas: 'pollo' en 3/4 (75%, señal REAL), 'queso' en 1/4 (no cruza el
    # umbral), y en TODAS: sal/pimienta/aceite bajo distintas frases de sazón +
    # un ingrediente inventado que el catálogo fixture NO resuelve.
    return [
        {"day": 1, "meals": [
            _meal("Desayuno", [
                "Sal al gusto", "Pimienta negra al gusto", "1 cda Aceite de oliva",
                "150g Pechuga de pollo",
            ]),
            _meal("Almuerzo", [
                "Sal al gusto", "1 cda Aceite de oliva", "150g Pechuga de pollo",
                "30g Queso blanco rallado",
            ]),
        ]},
        {"day": 2, "meals": [
            _meal("Desayuno", ["Sal al gusto", "1 cda Aceite de oliva", "150g Pechuga de pollo"]),
            _meal("Cena", [
                "Sal al gusto", "Pimienta negra al gusto", "1 cda Aceite de oliva",
                "1 unidad Ingrediente Alienigena Ficticio",
            ]),
        ]},
    ]


def test_gusto_never_survives_as_a_token(monkeypatch):
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeCatalogDB)
    result = g._count_ingredient_meal_frequency(_days_fixture())
    assert "gusto" not in result, f"'gusto' resucitó como token: {result}"


def test_oliva_never_survives_as_a_token(monkeypatch):
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeCatalogDB)
    result = g._count_ingredient_meal_frequency(_days_fixture())
    assert "oliva" not in result, f"'oliva' resucitó como token: {result}"


def test_negra_never_survives_as_a_token(monkeypatch):
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeCatalogDB)
    result = g._count_ingredient_meal_frequency(_days_fixture())
    assert "negra" not in result, f"'negra' resucitó como token: {result}"


def test_normalizer_never_emits_a_non_food_token(monkeypatch):
    """Regresión general (no solo gusto/oliva/negra): CUALQUIER palabra que no
    resuelva contra el catálogo verificado debe estar AUSENTE del resultado —
    nunca inventada como token. Cubre el ingrediente ficticio del fixture Y
    sirve de red para futuros modificadores de frases de sazón que el census
    original no vio (p.ej. si mañana "cebolla en polvo" produce 'polvo')."""
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeCatalogDB)
    result = g._count_ingredient_meal_frequency(_days_fixture())
    _KNOWN_FOOD_TOKENS = {"pollo", "queso"}
    for tok in result:
        assert tok in _KNOWN_FOOD_TOKENS, (
            f"Token '{tok}' no es un alimento reconocido por el catálogo fixture — "
            f"el normalizador está fabricando señal en vez de descartar lo no resuelto."
        )


def test_real_concentration_signal_still_fires(monkeypatch):
    """El fix no debe volver el detector mudo: 'pollo' en 3/4 comidas (75%,
    por encima del umbral 50% default) SÍ debe seguir disparando — la señal
    real de concentración, no solo los falsos positivos, es lo que se mide."""
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _FakeCatalogDB)
    result = g._count_ingredient_meal_frequency(_days_fixture())
    assert "pollo" in result, f"'pollo' (75% de las comidas) debería seguir disparando: {result}"
    assert result["pollo"] >= 0.5


def test_catalog_lookup_failure_fails_open_to_empty(monkeypatch):
    """Si `IngredientNutritionDB()` explota al construirse (Neon caído durante
    self-critique), el detector debe degradar a `{}` — nunca romper el pipeline
    (best-effort, ya lo garantiza el try/except exterior existente)."""
    def _boom(*a, **kw):
        raise RuntimeError("Neon down")
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _boom)
    result = g._count_ingredient_meal_frequency(_days_fixture())
    assert result == {}


def test_anchor_present_in_source():
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-INGREDIENT-SPREAD-GUSTO-FIX" in src
    assert '"aceite" in bare_norm.split()' in src
    assert "_dropped_unmatched" in src
