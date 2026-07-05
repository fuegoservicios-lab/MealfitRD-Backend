"""[P1-CLOSER-BARIATRIC-GUARD · 2026-07-05] Skip bariátrico del micro-closer/seeder.

Gap del audit quirúrgico solver+seeder 2026-07-05: `_close_micro_gaps_for_plan` (y su seed de
semillas/frutos secos) no tenía NINGUNA conciencia bariátrica. La capa clínica aplica los caps
de porción post-op (BARIATRIC_NUT_CAP_G=20g, textura por fase) ANTES del re-fire P2-2 del closer
→ una semilla de "10 g de nueces" sembrada (y luego escalada hasta 40g por el line-clamp) o un
contribuyente escalado 1.6× entraban al plan de un paciente post-op SIN re-validación bariátrica.
Clínicamente, la adecuación de micros post-bariátrica va por protocolo de SUPLEMENTACIÓN de por
vida (ASMBS) — no por "comer más volumen": el skip es el comportamiento correcto (mismo racional
del skip pantry_strict ya documentado en el docstring del closer).

Knob: MEALFIT_MICRO_CLOSER_BARIATRIC_SKIP (default ON). El guard corre ANTES de instanciar el
catálogo (cero costo para perfiles bariátricos).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ───────────────────────── parser-based ─────────────────────────

def test_marker_and_knob_default_on():
    assert "P1-CLOSER-BARIATRIC-GUARD" in _GO
    m = re.search(r'MICRO_CLOSER_BARIATRIC_SKIP\s*=\s*_env_bool\(\s*"MEALFIT_MICRO_CLOSER_BARIATRIC_SKIP"\s*,\s*(\w+)\)', _GO)
    assert m and m.group(1) == "True"


def test_guard_runs_before_catalog_instantiation():
    """El skip corre ANTES de `if db is None:` (cero instanciación del catálogo para bariátricos)."""
    i_fn = _GO.index("def _close_micro_gaps_for_plan")
    body = _GO[i_fn:i_fn + 6000]  # el docstring del closer es largo (~3k chars)
    i_guard = body.index("MICRO_CLOSER_BARIATRIC_SKIP and _is_bariatric_condition")
    i_db = body.index("if db is None:")
    assert i_guard < i_db, "el guard debe evitar el costo del catálogo"


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
    """apio: potasio proporcional a los gramos del prefijo 'NN g'."""

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "apio" in low:
            m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
            g = float(m.group(1).replace(",", ".")) if m else 40.0
            return {"potassium_mg": 2.6 * g}
        return {"potassium_mg": 0.0}

    def macros_from_ingredient_string(self, s):
        return {"kcal": 10.0}

    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
        return float(m.group(1).replace(",", ".")) if m else None

    def rescale_ingredient_string(self, s, factor):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s))
        if not m:
            return s
        g = float(m.group(1).replace(",", ".")) * factor
        return re.sub(r"^\s*\d+(?:[.,]\d+)?\s*g", f"{g:.0f} g", s, count=1)


def _mk_plan():
    return {"days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Ensalada",
         "ingredients": ["100 g de apio", "50 g de lechuga"],
         "ingredients_raw": ["100 g de apio", "50 g de lechuga"],
         "recipe": ["Sirve."]},
    ]}]}


def _run(go, monkeypatch, plan, form_data):
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: {
        "panel": [{"key": "potassium_mg", "piso": 500.0, "valor": 0.0, "status": "bajo"}],
        "gaps": [{"key": "potassium_mg", "piso": 500.0, "status": "bajo"}],
        "coverage": 1.0, "per_day_floors": {"flagged": False}})
    return go._close_micro_gaps_for_plan(plan, form_data, _FakeDB())


def test_bariatric_profile_skips_closer_entirely(go, monkeypatch):
    plan = _mk_plan()
    n = _run(go, monkeypatch, plan, {"medicalConditions": ["Cirugía bariátrica (bypass gástrico)"]})
    assert n == 0
    assert plan["days"][0]["meals"][0]["ingredients"] == ["100 g de apio", "50 g de lechuga"], \
        "el plan de un post-op bariátrico queda INTACTO (ni seed ni escala)"


def test_bariatric_skip_works_with_db_none(go, monkeypatch):
    """El guard corre antes de instanciar el catálogo → db=None jamás llega a IngredientNutritionDB()."""
    n = go._close_micro_gaps_for_plan(_mk_plan(), {"medicalConditions": ["manga gástrica"]}, db=None)
    assert n == 0


def test_non_bariatric_profile_unaffected(go, monkeypatch):
    """Control: mismo harness sin bariátrica → el closer escala hacia el piso (comportamiento previo)."""
    plan = _mk_plan()
    n = _run(go, monkeypatch, plan, {"medicalConditions": ["diabetes tipo 2"]})
    assert n >= 1, "el closer debe seguir actuando para perfiles no-bariátricos"
    assert plan["days"][0]["meals"][0]["ingredients"] != ["100 g de apio", "50 g de lechuga"]


def test_knob_off_restores_previous_behavior(go, monkeypatch):
    monkeypatch.setattr(go, "MICRO_CLOSER_BARIATRIC_SKIP", False)
    plan = _mk_plan()
    n = _run(go, monkeypatch, plan, {"medicalConditions": ["cirugía bariátrica"]})
    assert n >= 1, "con el knob OFF el closer vuelve al comportamiento previo (rollback sin redeploy)"
