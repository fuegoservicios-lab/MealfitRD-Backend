"""[P3-SOLVER-SEEDER-POLISH · 2026-07-07] Los P3 del audit solver+seeder v2 (pulido/observabilidad):

- P3-1: LIMITACIÓN documentada (seed-catalog cubre 3/13 micros) — comentario anclado.
- P3-2: phantom seed — resolver el micro ANTES de appendear (fail-secure vs catalog-drift).
- P3-3: savory-seed sin comida salada → NO sembrar (no zanahoria en merienda dulce).
- P3-4: 🌱 step-note — coacciona recipe a lista (no dropear la nota).
- P3-5: guard de dieta en el path de seed (fail-secure preventivo).
- P3-6: `report` del solver etiquetado `report_basis="greedy-reference"`.
- P3-8: `_trim_day_protein_to_ceiling` recompute HONESTO de C/F embebidos.
(P3-7 = no-action: el truth-up final resetea el drift delta/absoluto. P3-9 = residual display documentado.)
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "portion_solver.py"), encoding="utf-8") as f:
    _PS = f.read()


# ─────────────────────────── Parser anchors ───────────────────────────

def test_p3_1_seed_catalog_limitation_documented():
    assert "P3-1" in _GO and "SCALE-ONLY" in _GO
    i = _GO.index("_MICRO_SEED_SOURCES = {")
    assert "no_carrier_no_seed" in _GO[max(0, i - 900):i]  # cross-ref a la telemetría


def test_p3_2_resolve_before_append():
    assert "P3-2" in _GO
    # el gate `if _c_seed <= 0:` (skip) vive antes del append.
    assert "if _c_seed <= 0:" in _GO
    assert "fail-secure vs catalog-drift" in _GO


def test_p3_4_recipe_coerced_to_list():
    assert "P3-4" in _GO
    assert "if not isinstance(_rec_seed, list):" in _GO


def test_p3_5_diet_guard_on_seed_path():
    assert "P3-5" in _GO
    assert "_scan_diet_violations(" in _GO
    # dentro del loop de candidatos del seed (tras el allergen-scan).
    i = _GO.index("P3-5")
    assert "_seed_diet" in _GO[i:i + 600]


def test_p3_6_report_basis_labeled():
    assert _PS.count('"report_basis": "greedy-reference",') == 2  # ambas fns del solver


# ─────────────────────────── P3-6 funcional ───────────────────────────

class _MacroDB:
    _P100 = {"pollo": {"kcal": 165, "protein": 31, "carbs": 0, "fats": 4},
             "arroz": {"kcal": 130, "protein": 3, "carbs": 28, "fats": 0.5}}

    import re as _re

    def _g(self, s):
        m = self._re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
        return float(m.group(1).replace(",", ".")) if m else 0.0

    def macros_from_ingredient_string(self, s):
        low, g = str(s).lower(), self._g(s)
        for tok, per in self._P100.items():
            if tok in low and g > 0:
                return {k: v * g / 100.0 for k, v in per.items()}
        return None


def test_p3_6_report_basis_returned():
    import portion_solver as ps
    res = ps.solve_meal_macros(["100 g de pollo"], {"kcal": 300, "protein": 60, "carbs": 0, "fats": 6},
                               db=_MacroDB())
    assert res.get("report_basis") == "greedy-reference"


# ─────────────────────────── P3-8 funcional: recompute honesto C/F ───────────────────────────

def test_p3_8_protein_ceiling_lowers_embedded_fat():
    """Escalar el pollo hacia abajo (techo de proteína) también baja su GRASA embebida; antes `fats`
    quedaba stale (el reconcile lo leía inflado y sobre-recortaba)."""
    import graph_orchestrator as g
    meal = {"name": "Pollo", "ingredients": ["200 g de pollo"], "ingredients_raw": ["200 g de pollo"],
            "protein": 60, "carbs": 0, "fats": 8, "cals": 312}
    trimmed = g._trim_day_protein_to_ceiling([meal], 30.0, _MacroDB(), ceiling_pct=1.0)  # factor 0.5
    assert trimmed is True
    assert meal["protein"] < 60          # proteína bajó hacia el target
    assert meal["fats"] < 8, f"la grasa embebida del pollo debe bajar (honesto), no quedar en 8: {meal['fats']}"


# ─────────────────────────── P3-2 / P3-3 funcional (closer) ───────────────────────────

class _SeedDB:
    """Simula catalog-drift (`drift_tokens` → micros None) para P3-2; resto resuelve."""

    def __init__(self, drift_tokens=()):
        self._drift = tuple(drift_tokens)

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        if any(t in low for t in self._drift):
            return None  # no resuelve (drift)
        if "linaza" in low or "chia" in low or "chía" in low or "nuez" in low or "nueces" in low:
            return {"omega3_g": 0.6}
        if "zanahoria" in low or "auyama" in low or "espinaca" in low:
            return {"vit_a_mcg": 400.0}
        return {"omega3_g": 0.0, "vit_a_mcg": 0.0}

    def macros_from_ingredient_string(self, s):
        return {"kcal": 55.0, "protein": 2.0, "carbs": 4.0, "fats": 3.0}

    def grams_from_ingredient_string(self, s):
        return 10.0


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


def _report(key, floor):
    return {"panel": [{"key": key, "piso": floor, "valor": 0.0, "status": "bajo"}],
            "gaps": [{"key": key, "piso": floor, "status": "bajo"}],
            "coverage": 1.0, "per_day_floors": {"flagged": False}}


def _seeded_text(plan):
    return " | ".join(i for d in plan["days"] for m in d["meals"] for i in m["ingredients"]).lower()


def test_p3_2_phantom_seed_not_appended(go, monkeypatch):
    """El seed de omega-3 (linaza) NO resuelve (drift) → NO se appendea (antes: línea fantasma + flag)."""
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **k: _report("omega3_g", 1.6))
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Merienda", "name": "Fruta", "ingredients": ["1 manzana"],
         "ingredients_raw": ["1 manzana"], "recipe": ["Sirve."]}]}]}
    go._close_micro_gaps_for_plan(plan, {}, _SeedDB(drift_tokens=("linaza", "chia", "chía", "nuez", "nueces")))
    assert "linaza" not in _seeded_text(plan)
    assert all(not m.get("_micro_seed_applied") for d in plan["days"] for m in d["meals"])


def test_p3_3_savory_seed_skipped_without_savory_meal(go, monkeypatch):
    """vit A (seed SALADO) con el día SOLO en meriendas dulces → NO se siembra zanahoria."""
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **k: _report("vit_a_mcg", 700.0))
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Merienda", "name": "Yogurt con Fresas", "ingredients": ["1 taza de yogurt"],
         "ingredients_raw": ["1 taza de yogurt"], "recipe": ["Sirve."]}]}]}
    go._close_micro_gaps_for_plan(plan, {}, _SeedDB())
    assert "zanahoria" not in _seeded_text(plan) and "auyama" not in _seeded_text(plan)
