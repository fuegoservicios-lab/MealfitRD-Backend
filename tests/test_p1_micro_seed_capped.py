"""[P1-MICRO-SEED-CAPPED · 2026-07-05] Seeder v2 + plural del name-honesty.

Forense del plan 08ed036d (renovación 2026-07-05 05:15, banner micro_worst_day):

1. **Seeder v2**: el Día 1 TENÍA portador de omega-3 (sardinas) pero quedó en 0.44× del piso —
   el escalado richest-first está acotado (MAX_SCALE/UL/kcal) y no cierra gaps 2-3×; el seed v1
   solo actuaba SIN portadores. v2: siembra también cuando `day_total < MICRO_SEED_BELOW_RATIO
   × piso` (default 0.6 = alineado con `low_ratio_threshold` del banner per-día: si el día
   sería flaggeado, se siembra). Anti-doble-siembra por (día, micro).

2. **[P2-NAME-HONESTY-PLURAL]**: "Croquetas de Sardina" (nombre singular) con "½ lata de
   sardinaS en agua" (ingrediente plural) → `\bsardina\b` no matcheaba el plural → falso
   `_name_honesty_degraded` + chip "el nombre puede no reflejar la proteína real" en la UI
   sobre un plato honesto. Fix: sufijo plural español opcional `(?:s|es)?` en ambos lados.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
    """omega3: linaza 0.6 / nueces 0.9; kcal fija por línea."""

    @staticmethod
    def _norm(s):
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                       if unicodedata.category(c) != "Mn")

    def micros_from_ingredient_string(self, s):
        low = self._norm(s)
        if "linaza" in low:
            return {"omega3_g": 0.6}
        if "nueces" in low:
            return {"omega3_g": 0.9}
        return {"omega3_g": 0.0}

    def macros_from_ingredient_string(self, s):
        return {"kcal": 55.0}


def _mk_report(key, piso):
    return {"panel": [{"key": key, "piso": piso, "valor": 0.0, "status": "bajo"}],
            "gaps": [{"key": key, "piso": piso, "status": "bajo"}],
            "coverage": 1.0, "per_day_floors": {"flagged": False}}


def _mk_plan(carrier=None):
    meals = [
        {"meal": "Desayuno", "name": "Avena", "ingredients": ["40 g de avena"],
         "ingredients_raw": ["40 g de avena"], "recipe": ["Cocina."]},
        {"meal": "Merienda", "name": "Fruta", "ingredients": ["1 manzana"],
         "ingredients_raw": ["1 manzana"], "recipe": ["Sirve."]},
    ]
    if carrier:
        meals[0]["ingredients"].append(carrier)
        meals[0]["ingredients_raw"].append(carrier)
    return {"days": [{"day": 1, "meals": meals}]}


def _run(go, monkeypatch, plan, piso):
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report",
                        lambda *a, **kw: _mk_report("omega3_g", piso))
    go._close_micro_gaps_for_plan(plan, {}, _FakeDB())
    return plan


# ---------------------------------------------------------------------------
# seeder v2
# ---------------------------------------------------------------------------

def test_knob_default_aligned_with_banner_threshold():
    assert '_env_float("MEALFIT_MICRO_SEED_BELOW_RATIO", 0.6)' in _GO


def test_seeds_when_carrier_deeply_short(go, monkeypatch):
    # portador linaza 0.6 con piso 2.0 → 0.6 < 0.6×2.0=1.2 → SIEMBRA pese al portador.
    plan = _run(go, monkeypatch, _mk_plan(carrier="10 g de semillas de linaza"), piso=2.0)
    _all = " | ".join(i for d in plan["days"] for m in d["meals"] for i in m["ingredients"])
    assert _all.count("linaza") >= 2 or "nueces" in _all, \
        "con portador PROFUNDAMENTE corto el v2 debe sembrar (el escalado no cierra 3.3×)"


def test_no_seed_when_carrier_moderately_short(go, monkeypatch):
    # portador nueces 0.9 con piso 1.0 → 0.9 > 0.6×1.0 → NO siembra (solo escala, v1 intacto).
    plan = _run(go, monkeypatch, _mk_plan(carrier="10 g de nueces"), piso=1.0)
    _all = " | ".join(i for d in plan["days"] for m in d["meals"] for i in m["ingredients"])
    assert "linaza" not in _all and _all.count("nueces") == 1


def test_no_double_seed(go, monkeypatch):
    plan = _mk_plan(carrier="10 g de semillas de linaza")
    plan["days"][0]["meals"][1]["_micro_seed_applied"] = "omega3_g"
    plan = _run(go, monkeypatch, plan, piso=2.0)
    _all = " | ".join(i for d in plan["days"] for m in d["meals"] for i in m["ingredients"])
    assert _all.count("linaza") == 1, "día ya sembrado para el micro → no re-sembrar"


# ---------------------------------------------------------------------------
# [P2-NAME-HONESTY-PLURAL] plural español en el matcher
# ---------------------------------------------------------------------------

def test_sardinas_plural_not_flagged(go):
    from constants import strip_accents
    meal = {"name": "Croquetas de Sardina y Bulgur al Horno",
            "ingredients": ["½ lata de sardinas en agua (62g)", "0.33 taza de bulgur seco"],
            "ingredients_raw": ["½ lata de sardinas en agua (62g)", "0.33 taza de bulgur seco"]}
    renamed = go._fix_phantom_protein_in_name(meal, strip_accents)
    assert renamed is False
    assert not meal.get("_name_honesty_degraded"), \
        "sardina (nombre) vs sardinaS (ingrediente) es el MISMO alimento — falso positivo del plural"


def test_real_phantom_still_flags(go):
    from constants import strip_accents
    meal = {"name": "Pollo a la Plancha", "ingredients": ["200 g de ñame", "1 yogurt griego"],
            "ingredients_raw": ["200 g de ñame", "1 yogurt griego"]}
    go._fix_phantom_protein_in_name(meal, strip_accents)
    assert meal.get("_name_honesty_degraded") is True, \
        "el propósito original sigue: pollo en el título con 0 pollo y sin carne real → flag"


def test_marker_anchored_in_source():
    assert "P1-MICRO-SEED-CAPPED" in _GO
    assert "P2-NAME-HONESTY-PLURAL" in _GO
