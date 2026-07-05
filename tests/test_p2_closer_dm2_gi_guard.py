"""[P2-CLOSER-DM2-GI-GUARD · 2026-07-05] (audit solver+seeder P2-4)

El micro-closer era DM2-ciego al escalar: para cerrar potasio/vit C, el contribuyente más rico
del día suele ser FRUTA — y en un diabético escalar 1.6× guineo maduro/mango/piña añade azúcar
de carga glucémica directa. Mismo patrón del guard dislip/HTA (`_ceiling_risky_contributor`):
bajo DM2 los contribuyentes alto-IG se SALTAN y el siguiente richest-first toma su lugar; si no
hay alternativa, el residual se loguea como siempre (jamás se rompe la banda glucémica para
cerrar un piso de micros).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_and_tokens_anchored():
    assert "P2-CLOSER-DM2-GI-GUARD" in _GO
    i = _GO.index("_DM2_HIGHGI_TOKENS = (")
    win = _GO[i:i + 300]
    for t in ("guineo", "platano maduro", "mango", "pina", "uva", "miel"):
        assert t in win, f"token alto-IG '{t}' ausente de la lista"


def test_guard_branch_inside_ceiling_risky_contributor():
    i = _GO.index("def _ceiling_risky_contributor")
    body = _GO[i:i + 900]
    assert "_dm2 and _dm2_gi_re.search(_ing_low)" in body


def test_tokens_applied_with_word_boundary():
    """Lección 'res'↔'fresas': 'pina' como substring matchearía dentro de 'esPINAca' y bloquearía
    justo la alternativa verde. El guard DEBE aplicar con regex \\b."""
    i = _GO.index("_DM2_HIGHGI_TOKENS = (")
    win = _GO[i:i + 500]
    assert "_dm2_gi_re = _re.compile(" in win and r")s?\b" in win


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "MICRO_SEED_ENABLED", False)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
    """guineo maduro: K denso (4.0/g); espinaca: K menos denso (2.0/g)."""

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
        g = float(m.group(1).replace(",", ".")) if m else 0.0
        if "guineo" in low:
            return {"potassium_mg": 4.0 * g}
        if "espinaca" in low:
            return {"potassium_mg": 2.0 * g}
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
        {"meal": "Almuerzo", "name": "Bowl",
         "ingredients": ["100 g de guineo maduro", "100 g de espinaca"],
         "ingredients_raw": ["100 g de guineo maduro", "100 g de espinaca"],
         "recipe": ["Sirve."]},
    ]}]}


def _run(go, monkeypatch, plan, conditions):
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: {
        "panel": [{"key": "potassium_mg", "piso": 900.0, "valor": 0.0, "status": "bajo"}],
        "gaps": [{"key": "potassium_mg", "piso": 900.0, "status": "bajo"}],
        "coverage": 1.0, "per_day_floors": {"flagged": False}})
    return go._close_micro_gaps_for_plan(plan, {"medicalConditions": conditions}, _FakeDB())


def _grams(plan, food):
    line = next(s for s in plan["days"][0]["meals"][0]["ingredients"] if food in s)
    return float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", line).group(1).replace(",", "."))


def test_dm2_skips_highgi_scales_alternative(go, monkeypatch):
    plan = _mk_plan()
    n = _run(go, monkeypatch, plan, ["Diabetes tipo 2"])
    assert n >= 1, "el closer debe seguir cerrando con el contribuyente alternativo"
    assert _grams(plan, "guineo") == 100.0, "fruta alto-IG jamás es target de escalado en DM2"
    assert _grams(plan, "espinaca") > 100.0, "la espinaca (no-IG) toma el lugar richest-first"


def test_non_dm2_scales_richest_as_before(go, monkeypatch):
    plan = _mk_plan()
    _run(go, monkeypatch, plan, ["Ninguna"])
    assert _grams(plan, "guineo") > 100.0, "sin DM2 el richest-first sigue siendo el guineo"
