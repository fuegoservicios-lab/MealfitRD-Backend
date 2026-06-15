"""[P2-SUBS-RESOLUBILITY · 2026-06-15] Contrato de resolubilidad de reemplazos + dedup quantity-aware.

Cierra dos gaps P2 del audit:
  - P2-17: el contrato de resolubilidad cubría SOLO dislipidemia → un reemplazo de HTA/alérgeno que NO
    resuelve al catálogo pasaba CI sin detección (bug P2-16: "Pescado fresco" → fantasma en la lista +
    delta de macros perdía el ingrediente). Ahora cubre TODAS las familias: DM2/HTA/dislipidemia +
    alérgenos (fish/shellfish/soy/gluten). Solo aplica a reemplazos `preserve_qty=True` (los que
    cargan peso/macros; los condimentos bare como "Stevia al gusto" no son ingredientes de catálogo).
  - P2-18: el dedup `if new not in out` colapsaba dos staples del mismo gramaje que mapean al mismo
    reemplazo (salami+longaniza→2×pollo) a UNA entrada → media comida + macros inflados. Ahora los
    staples con cantidad se conservan; solo los condimentos bare se deduplican.

El contrato de resolubilidad necesita el catálogo (Neon). Sin DB se salta — se valida en el VPS.
Los tests de dedup son deterministas (string-based, sin DB).
"""
from __future__ import annotations

import pytest

import condition_rules as cr

_CONDITIONS = ["Diabetes tipo 2", "Hipertensión", "Colesterol alto"]
_ALLERGIES = ["pescado", "mariscos", "soya", "gluten"]


def _preserve_qty_replacements():
    """Todos los reemplazos `preserve_qty=True` (los que deben resolver al catálogo) de TODAS las familias."""
    out = set()
    for c in _CONDITIONS:
        for s in cr.collect_substitutions({"medicalConditions": [c]}):
            if s.get("preserve_qty"):
                out.add(s["replacement"])
    for a in _ALLERGIES:
        for s in cr.collect_allergen_substitutions({"allergies": [a]}):
            if s.get("preserve_qty"):
                out.add(s["replacement"])
    return sorted(out)


@pytest.fixture(scope="module")
def db():
    import nutrition_db as ndb
    d = ndb.IngredientNutritionDB()
    d._ensure_loaded()
    return d


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


# ════════════════════════════════════════════════════════════════════════════════════════════════
# A. Contrato de resolubilidad — TODAS las familias (P2-17)
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_contract_is_non_trivial():
    """Sanity: el contrato cubre varios reemplazos (no pasa vacuamente)."""
    repls = _preserve_qty_replacements()
    assert len(repls) >= 4, repls


def test_all_preserve_qty_replacements_resolve(db):
    if not getattr(db, "_rows", None):
        pytest.skip("catálogo no disponible (sin DB en este entorno; se valida en el VPS)")
    failures = [r for r in _preserve_qty_replacements() if db.lookup(r) is None]
    assert not failures, f"reemplazos preserve_qty que NO resuelven al catálogo: {failures}"


# ════════════════════════════════════════════════════════════════════════════════════════════════
# B. Dedup quantity-aware (P2-18) — deterministas, sin DB
# ════════════════════════════════════════════════════════════════════════════════════════════════
def _plan(ings):
    return {"days": [{"meals": [{
        "name": "Almuerzo", "ingredients": list(ings), "ingredients_raw": list(ings),
        "recipe": ["Cocina"], "protein": 40, "carbs": 50, "fats": 20, "cals": 540,
    }]}]}


def test_two_staples_same_replacement_are_preserved(go):
    """salami + longaniza (mismo gramaje) → DOS pollos (no colapsar a uno) → peso comprable preservado."""
    plan = _plan(["100g de salami", "100g de longaniza", "1 taza de arroz"])
    go._apply_condition_substitutions(plan, {"medicalConditions": ["Hipertensión"]})
    ings = plan["days"][0]["meals"][0]["ingredients"]
    chicken = [i for i in ings if "pollo" in str(i).lower()]
    assert len(chicken) == 2, ("ambos staples deben conservarse", ings)
    assert all("100" in str(c) for c in chicken), ("cantidad preservada en ambos", chicken)


def test_bare_condiments_still_dedup(go):
    """Condimentos bare (preserve_qty=False) SÍ se deduplican: 2 azúcares → 1 'Stevia' (no ruido)."""
    plan = _plan(["1 cda de azucar", "1 cdta de azucar", "1 taza de avena"])
    go._apply_condition_substitutions(plan, {"medicalConditions": ["Diabetes tipo 2"]})
    ings = plan["days"][0]["meals"][0]["ingredients"]
    stevia = [i for i in ings if "stevia" in str(i).lower()]
    assert len(stevia) == 1, ("condimentos bare se deduplican", ings)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# C. P2-16: el reemplazo HTA de pescado salado es un nombre del catálogo (no 'Pescado fresco')
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_hta_salted_fish_replacement_is_catalog_name():
    repls = {s["replacement"] for s in cr.collect_substitutions({"medicalConditions": ["Hipertensión"]})}
    assert "Pescado fresco" not in repls, "'Pescado fresco' no resuelve al catálogo — usar nombre real"
    assert "Filete de pescado blanco" in repls, "el reemplazo de pescado salado debe ser un nombre del catálogo"
