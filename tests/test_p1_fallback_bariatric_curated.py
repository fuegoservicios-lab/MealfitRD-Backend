"""[P1-FALLBACK-BARIATRIC-CURATED · 2026-06-28] Cuando un usuario bariátrico cae al fallback crítico (rechazo médico),
en vez del plan matemático GENÉRICO de 3 comidas (que no respeta el pouch y a veces viola reglas) recibe un plan
bariátrico CURADO de 6 comidas clínicamente vetadas POR CONSTRUCCIÓN. Diseñado + verificado por workflow adversario
(código + clínico ASMBS) que corrigió el pool: cottage→yogurt griego (el cap de queso lo amputaría a 30g), volúmenes
≤290g (cap 300), galleta/caldo (no resuelven en catálogo) → casabe/auyama.

El invariante CENTRAL (test_cap_is_noop): correr cap_bariatric_portions sobre el día curado recorta CERO porciones →
el pool ya cumple todos los caps por construcción. Si este test falla, el pool tiene una porción fuera de cap.

Tests PUROS (sin Neon): _get_extreme_fallback_plan / _build_fallback_day / cap_bariatric_portions parsean gramos del
string (grams_from_ingredient_string) sin DB.
"""
from __future__ import annotations

import graph_orchestrator as g

_BAR = {"medicalConditions": ["Cirugía Bariátrica (manga gástrica)"]}
_NON = {"medicalConditions": ["Diabetes tipo 2"]}
_NUTR = {"target_calories": 1900, "macros": {"protein_g": 110, "carbs_g": 160, "fats_g": 55}}

_SLOTS6 = ["Desayuno", "Merienda AM", "Almuerzo", "Merienda PM", "Cena", "Merienda Nocturna"]
_MEAL_KEYS = {"meal", "time", "name", "desc", "prep_time", "difficulty",
              "cals", "protein", "carbs", "fats", "macros", "ingredients", "recipe"}


def test_bariatric_gets_6_curated_meals():
    p = g._get_extreme_fallback_plan(_NUTR, "Mantenimiento", num_days=1, form_data=_BAR)
    day = p["days"][0]
    assert len(day["meals"]) == 6
    assert [m["meal"] for m in day["meals"]] == _SLOTS6
    assert p.get("_fallback_bariatric_curated") is True
    assert p.get("_is_fallback") is True


def test_non_bariatric_stays_generic_3():
    p = g._get_extreme_fallback_plan(_NUTR, "Mantenimiento", num_days=1, form_data=_NON)
    assert len(p["days"][0]["meals"]) == 3
    assert [m["meal"] for m in p["days"][0]["meals"]] == ["Desayuno", "Almuerzo", "Cena"]
    assert p.get("_fallback_bariatric_curated") is None


def test_form_data_none_is_generic():
    # callers legacy que no pasan form_data → comportamiento histórico (3 comidas)
    p = g._get_extreme_fallback_plan(_NUTR, "Mantenimiento", num_days=1)
    assert len(p["days"][0]["meals"]) == 3


def test_knob_off_falls_to_generic(monkeypatch):
    monkeypatch.setattr(g, "FALLBACK_BARIATRIC_CURATED_ENABLED", False)
    p = g._get_extreme_fallback_plan(_NUTR, "Mantenimiento", num_days=1, form_data=_BAR)
    assert len(p["days"][0]["meals"]) == 3  # rollback sin redeploy


def test_cap_is_noop():
    """INVARIANTE CENTRAL: el pool curado pasa los caps bariátricos por construcción → cap recorta CERO."""
    p = g._get_extreme_fallback_plan(_NUTR, "Mantenimiento", num_days=1, form_data=_BAR)
    days = [{"day": 0, "meals": [dict(m) for m in p["days"][0]["meals"]]}]
    n = g.cap_bariatric_portions(days, _BAR)  # sin db: parsea gramos del string
    assert n == 0, f"el pool curado debería ser cap-no-op, pero recortó {n} porción(es)"


def test_ratios_sum_to_one():
    assert abs(sum(g._BARIATRIC_FALLBACK_RATIOS[6].values()) - 1.0) < 1e-9


def test_pool_structure():
    pool = g._FALLBACK_MEAL_POOLS_BARIATRIC
    assert set(pool.keys()) == set(_SLOTS6)
    for slot, entries in pool.items():
        assert len(entries) >= 2, slot
        for name, tokens, desc, ings in entries:
            assert isinstance(name, str) and name
            assert isinstance(tokens, frozenset)
            assert isinstance(desc, str) and desc
            assert isinstance(ings, list) and ings
        # último item neutral (frozenset vacío) → _select_safe_fallback_meal siempre tiene retorno seguro
        assert entries[-1][1] == frozenset(), f"{slot}: el último item debe ser neutral"


def test_meal_format_compatible():
    p = g._get_extreme_fallback_plan(_NUTR, "Mantenimiento", num_days=1, form_data=_BAR)
    for m in p["days"][0]["meals"]:
        assert _MEAL_KEYS.issubset(set(m.keys())), set(m.keys())


def test_fallback_flags_and_empty_shopping():
    p = g._get_extreme_fallback_plan(_NUTR, "Mantenimiento", num_days=1, form_data=_BAR)
    assert p["_is_fallback"] is True
    assert p.get("_review_disclaimer")
    for k in ("aggregated_shopping_list", "aggregated_shopping_list_weekly",
              "aggregated_shopping_list_biweekly", "aggregated_shopping_list_monthly"):
        assert p[k] == []


def test_allergen_filter_bariatric():
    p = g._get_extreme_fallback_plan(_NUTR, "Mantenimiento", num_days=1,
                                     restricted_tokens=frozenset({"fish"}), form_data=_BAR)
    for m in p["days"][0]["meals"]:
        low = m["name"].lower()
        assert not any(w in low for w in ("atún", "sardina", "mero", "tilapia")), m["name"]


def test_knob_and_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-FALLBACK-BARIATRIC-CURATED" in src
    assert "FALLBACK_BARIATRIC_CURATED_ENABLED" in src
    assert "_FALLBACK_MEAL_POOLS_BARIATRIC" in src
    assert g.FALLBACK_BARIATRIC_MEAL_COUNT == 6
