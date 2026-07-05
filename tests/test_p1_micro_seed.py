"""[P1-MICRO-SEED · 2026-07-04] El closer siembra una fuente cuando el día no tiene ninguna.

Caso vivo (renovaciones 2026-07-04/05): el closer de micros solo ESCALA fuentes existentes —
un día sin ningún portador de omega-3/vit E quedaba <60% del piso sin remedio (residual
"closer insuficiente (contribuyentes)") → `per_day_floors.flagged` → banner micro_worst_day
con chips promedio verdes (contradicción visual que el owner reportó dos veces).

Con el seed: día deficitario SIN contribuyentes + micro en `_MICRO_SEED_SOURCES` → se añade
UNA línea pequeña verificada del catálogo (~60 kcal, dentro del presupuesto kcal/día,
alergia/dislike-aware con escalera frutos-secos→semillas) y el loop richest-first la escala
hacia el piso. Exclusiones clínicas heredadas del closer (renal/K-med/anticoag intactas).
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
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)  # aisla el path promedio
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
    """omega3 solo en linaza/nueces; vit E solo en maní/linaza. kcal fija por línea."""

    @staticmethod
    def _norm(s):
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                       if unicodedata.category(c) != "Mn")

    def micros_from_ingredient_string(self, s):
        low = self._norm(s)
        out = {}
        if "linaza" in low:
            out = {"omega3_g": 0.6, "vit_e_mg": 0.1}
        elif "nueces" in low:
            out = {"omega3_g": 0.9, "vit_e_mg": 0.2}
        elif "mani" in low:
            out = {"vit_e_mg": 0.8, "omega3_g": 0.0}
        return out or {"omega3_g": 0.0, "vit_e_mg": 0.0}

    def macros_from_ingredient_string(self, s):
        return {"kcal": 55.0}


def _mk_report(key, piso):
    return {
        "panel": [{"key": key, "piso": piso, "valor": 0.0, "status": "bajo"}],
        "gaps": [{"key": key, "piso": piso, "status": "bajo"}],
        "coverage": 1.0,
        "per_day_floors": {"flagged": False},
    }


def _mk_plan():
    return {"days": [{
        "day": 1,
        "meals": [
            {"meal": "Desayuno", "name": "Avena", "ingredients": ["40 g de avena"],
             "ingredients_raw": ["40 g de avena"], "recipe": ["Cocina."]},
            {"meal": "Merienda", "name": "Fruta", "ingredients": ["1 manzana"],
             "ingredients_raw": ["1 manzana"], "recipe": ["Sirve."]},
        ],
    }]}


def _run(go, monkeypatch, key="omega3_g", piso=1.0, form=None):
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report",
                        lambda *a, **kw: _mk_report(key, piso))
    plan = _mk_plan()
    n = go._close_micro_gaps_for_plan(plan, form or {}, _FakeDB())
    return n, plan


# ---------------------------------------------------------------------------

def test_knob_and_sources():
    assert '_env_bool("MEALFIT_MICRO_SEED", True)' in _GO
    assert '"omega3_g": ("10 g de semillas de linaza"' in _GO
    assert '"vit_e_mg": ("10 g de maní"' in _GO


def test_seeds_when_day_has_no_carrier(go, monkeypatch):
    n, plan = _run(go, monkeypatch, key="omega3_g", piso=1.0)
    assert n >= 1
    merienda = plan["days"][0]["meals"][1]  # prefiere la merienda para el seed
    joined = " ".join(merienda["ingredients"])
    assert "linaza" in joined, "sin portadores de omega-3 el closer debe sembrar linaza"
    assert merienda["_micro_seed_applied"] == "omega3_g"
    # raw se mantiene alineado posicionalmente (contrato del panel/lista).
    assert len(merienda["ingredients"]) == len(merienda["ingredients_raw"])


def test_allergy_ladder_nuts_to_seeds(go, monkeypatch):
    # alergia a maní → para vit E cae al segundo candidato (linaza).
    monkeypatch.setattr(
        go, "_scan_allergen_violations",
        lambda plan, allergies: (["maní"] if any("maní" in str(i) for d in plan["days"]
                                                 for m in d["meals"] for i in m["ingredients"]) else []),
    )
    n, plan = _run(go, monkeypatch, key="vit_e_mg", piso=0.5, form={"allergies": ["maní"]})
    assert n >= 1
    joined = " ".join(plan["days"][0]["meals"][1]["ingredients"])
    assert "maní" not in joined and "linaza" in joined


def test_dislike_ladder(go, monkeypatch):
    n, plan = _run(go, monkeypatch, key="omega3_g", piso=1.0, form={"dislikes": ["linaza"]})
    assert n >= 1
    joined = " ".join(plan["days"][0]["meals"][1]["ingredients"])
    assert "linaza" not in joined and "nueces" in joined


def test_knob_off_no_seed(go, monkeypatch):
    monkeypatch.setattr(go, "MICRO_SEED_ENABLED", False)
    n, plan = _run(go, monkeypatch, key="omega3_g", piso=1.0)
    joined = " ".join(plan["days"][0]["meals"][1]["ingredients"])
    assert "linaza" not in joined and "nueces" not in joined


def test_no_seed_when_carrier_exists(go, monkeypatch):
    """Con portador presente, el closer ESCALA (comportamiento original) — no siembra doble."""
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report",
                        lambda *a, **kw: _mk_report("omega3_g", 1.0))
    plan = _mk_plan()
    plan["days"][0]["meals"][0]["ingredients"].append("10 g de semillas de linaza")
    plan["days"][0]["meals"][0]["ingredients_raw"].append("10 g de semillas de linaza")
    go._close_micro_gaps_for_plan(plan, {}, _FakeDB())
    _all = " | ".join(i for d in plan["days"] for m in d["meals"] for i in m["ingredients"])
    assert _all.count("linaza") == 1, "no debe sembrar si ya existe un portador (solo escalar)"


def test_marker_anchored_in_source():
    assert "P1-MICRO-SEED" in _GO
