"""[P1-DIET-HARD-GUARD · 2026-06-15] (gap-audit P1-3) Backstop DETERMINISTA de dietType.

Cierra el gap: la ÚNICA defensa contra un producto animal en un plan vegano era el `critical` del
revisor LLM — falible Y degradable por el soft-reject DM2 (P1-1). Este P-fix añade:
  - `_scan_diet_violations(plan, dietType)`: escanea ingredientes contra productos animales prohibidos
    por la dieta (vegano/vegetariano/pescetariano), con exclusión de análogos plant-based por adyacencia.
  - bloque en `review_plan_node` que escala a CRÍTICO + marca `_had_diet_critical` (no degradable).
  - `_diet_restricted_tokens` + `_fallback_restricted_tokens` diet-aware → el fallback es veg-safe (sin
    esto, escalar vegano→crítico→fallback servía pollo/huevo igual, counterproductivo).

Validación determinista de helpers puros (sin LLM/DB/créditos) + parser-anchors sobre el source de prod.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_GO_PATH = Path(__file__).resolve().parent.parent / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def _plan_with(ingredients):
    return {"days": [{"day": 1, "meals": [{"name": "Almuerzo", "ingredients": list(ingredients)}]}]}


# ---------------------------------------------------------------------------
# _scan_diet_violations — detección por dieta
# ---------------------------------------------------------------------------
def test_vegan_flags_chicken(go):
    viol = go._scan_diet_violations(_plan_with(["180g pechuga de pollo", "1 taza arroz"]), "vegan")
    assert viol and any("pollo" in v[1].lower() for v in viol)


def test_vegan_flags_egg_and_dairy(go):
    viol_egg = go._scan_diet_violations(_plan_with(["2 huevos revueltos"]), "vegan")
    viol_milk = go._scan_diet_violations(_plan_with(["1 taza de leche entera"]), "vegano")
    assert viol_egg, "vegano debe rechazar huevo"
    assert viol_milk, "vegano debe rechazar leche (lácteo)"


def test_vegan_allows_plant_analogs(go):
    # análogos plant-based adyacentes: NO violan
    for ing in ["100g carne de soya", "1 taza de leche de coco", "salami vegano",
                "queso vegetal", "crema de maní", "chorizo de soya"]:
        viol = go._scan_diet_violations(_plan_with([ing]), "vegan")
        assert not viol, f"{ing!r} es plant-based, NO debe marcarse: {viol}"


def test_real_meat_with_distant_plant_still_flagged(go):
    # plant lejano NO excusa la proteína animal real
    viol = go._scan_diet_violations(_plan_with(["pollo guisado con leche de coco"]), "vegan")
    assert viol, "pollo real (con coco lejano) DEBE marcarse"


def test_vegetarian_allows_egg_dairy_but_not_fish(go):
    assert not go._scan_diet_violations(_plan_with(["2 huevos", "queso fresco"]), "vegetarian"), \
        "vegetariano permite huevo/lácteo"
    assert go._scan_diet_violations(_plan_with(["filete de pescado"]), "vegetariano"), \
        "vegetariano NO permite pescado"


def test_pescatarian_allows_fish_not_land_meat(go):
    assert not go._scan_diet_violations(_plan_with(["atún a la plancha"]), "pescatarian"), \
        "pescetariano permite pescado"
    assert go._scan_diet_violations(_plan_with(["chuleta de cerdo"]), "pescetariano"), \
        "pescetariano NO permite carne de tierra"


def test_balanced_or_unknown_never_flags(go):
    for d in ("balanced", "Omnívora", "", None):
        assert go._scan_diet_violations(_plan_with(["pollo", "huevo", "queso"]), d) == [], \
            f"dieta {d!r} no debe restringir"


# ---------------------------------------------------------------------------
# Fallback diet-aware
# ---------------------------------------------------------------------------
def test_diet_restricted_tokens(go):
    vegan = go._diet_restricted_tokens({"dietType": "vegan"})
    veg = go._diet_restricted_tokens({"dietType": "vegetarian"})
    assert {"chicken", "fish", "beef", "pork", "egg", "dairy"} <= vegan
    assert "chicken" in veg and "egg" not in veg, "vegetariano permite huevo en el fallback"
    assert go._diet_restricted_tokens({"dietType": "balanced"}) == frozenset()


def test_fallback_restricted_tokens_union(go, monkeypatch):
    monkeypatch.setattr(go, "DIET_HARD_GUARD", True)
    toks = go._fallback_restricted_tokens({"dietType": "vegan"})
    assert "egg" in toks and "chicken" in toks
    # knob OFF → sin tokens de dieta
    monkeypatch.setattr(go, "DIET_HARD_GUARD", False)
    monkeypatch.setattr(go, "FALLBACK_ALLERGEN_FILTER", False)
    assert go._fallback_restricted_tokens({"dietType": "vegan"}) == frozenset()


def test_vegan_fallback_plan_has_no_animal_ingredients(go):
    """El fallback matemático para un vegano NO debe contener productos animales."""
    toks = go._fallback_restricted_tokens({"dietType": "vegan"})
    nutr = {"target_calories": 1800, "macros": {"protein_g": 120, "carbs_g": 180, "fats_g": 60}}
    plan = go._get_extreme_fallback_plan(nutr, "lose_fat", num_days=2, restricted_tokens=toks)
    animal = ("pollo", "huevo", "pescado", "carne ", "res", "cerdo", "pechuga", "atun",
              "atún", "queso", "leche ", "jamon", "salmon")
    for day in plan["days"]:
        for meal in day["meals"]:
            text = " ".join(meal["ingredients"]).lower()
            for a in animal:
                assert a not in text, f"fallback vegano contiene {a!r}: {meal['ingredients']}"


# ---------------------------------------------------------------------------
# Knob + registro de seguridad
# ---------------------------------------------------------------------------
def test_knob_default_on_and_registered(go):
    assert go.DIET_HARD_GUARD is True, "DIET_HARD_GUARD default True (safety guard)"
    names = [k[0] for k in go._SAFETY_CRITICAL_KNOBS]
    assert "MEALFIT_DIET_HARD_GUARD" in names, "el guard de dieta debe alertar si se apaga en prod"


# ---------------------------------------------------------------------------
# Post-review: formas legacy ES femeninas (P1) + animales procesados (P2)
# ---------------------------------------------------------------------------
def test_canonicalize_diet_type_covers_legacy_forms(go):
    assert go._canonicalize_diet_type("vegana") == "vegan"
    assert go._canonicalize_diet_type("vegano") == "vegan"
    assert go._canonicalize_diet_type("vegetariana") == "vegetarian"
    assert go._canonicalize_diet_type("pescetariano") == "pescatarian"
    assert go._canonicalize_diet_type("Omnívora") == "balanced"
    assert go._canonicalize_diet_type("") == "balanced"


def test_legacy_feminine_vegana_is_guarded(go):
    # P1 post-review: 'vegana' (legacy ES femenino, aceptado por el backend) DEBE comportarse como vegano
    assert go._scan_diet_violations(_plan_with(["pechuga de pollo"]), "vegana"), \
        "'vegana' debe escanear como vegano"
    toks = go._diet_restricted_tokens({"dietType": "vegana"})
    assert "egg" in toks and "chicken" in toks, "'vegana' debe restringir tokens en el fallback"


def test_legacy_feminine_vegetariana_is_guarded(go):
    assert go._scan_diet_violations(_plan_with(["filete de pescado"]), "vegetariana"), \
        "'vegetariana' debe rechazar pescado"
    assert "chicken" in go._diet_restricted_tokens({"dietType": "vegetariana"})


def test_vegana_fallback_has_no_animal(go):
    toks = go._fallback_restricted_tokens({"dietType": "vegana"})
    nutr = {"target_calories": 1800, "macros": {"protein_g": 120, "carbs_g": 180, "fats_g": 60}}
    plan = go._get_extreme_fallback_plan(nutr, "lose_fat", num_days=2, restricted_tokens=toks)
    animal = ("pollo", "huevo", "pescado", "res", "cerdo", "pechuga", "atun", "queso", "leche ")
    for day in plan["days"]:
        for meal in day["meals"]:
            text = " ".join(meal["ingredients"]).lower()
            for a in animal:
                assert a not in text, f"fallback 'vegana' contiene {a!r}: {meal['ingredients']}"


def test_vegan_flags_processed_animal_products(go):
    # P2 post-review: huevo (mayonesa/merengue/mousse/alioli) + lácteo (helado/mantecado/flan) — espejo del allergen guard
    for ing in ["2 cda de mayonesa", "1 bola de helado", "flan de leche", "1 porción de mantecado",
                "merengue dominicano", "mousse de chocolate", "alioli casero"]:
        assert go._scan_diet_violations(_plan_with([ing]), "vegan"), f"{ing!r} debe marcarse para vegano"
    # análogos plant-based siguen excusados por adyacencia (over-detect no rompe versiones veganas comunes)
    for ing in ["helado de coco", "flan de coco", "mayonesa vegana"]:
        assert not go._scan_diet_violations(_plan_with([ing]), "vegan"), f"{ing!r} es vegano, no debe marcarse"


# ---------------------------------------------------------------------------
# Parser-anchors sobre el source de producción
# ---------------------------------------------------------------------------
def test_review_block_wires_diet_scan():
    src = _GO_PATH.read_text(encoding="utf-8")
    assert '_scan_diet_violations(plan, form_data.get("dietType"))' in src, \
        "review_plan_node debe invocar el scan de dieta"
    assert "_had_diet_critical = True" in src, "el scan debe marcar _had_diet_critical"


def test_soft_reject_excludes_diet_critical():
    src = _GO_PATH.read_text(encoding="utf-8")
    # la condición del soft-reject DM2 debe excluir el critical de dieta
    assert "and not _had_diet_critical" in src, \
        "el soft-reject DM2 no debe degradar un critical de dieta (P1-3)"
