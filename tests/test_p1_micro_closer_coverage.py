"""[P1-MICRO-CLOSER-COVERAGE · 2026-06-29] (re-audit objetivo · P1)

El closer DETERMINISTA de micros cubría solo 3 de los 18 del panel (fibra/Mg/Ca). HIERRO (piso 18mg
mujeres / 27mg embarazo) y FOLATO (600mcg embarazo) son los déficits poblacionales de mayor
consecuencia y vivían SOLO con un nudge al prompt.

Fix: se añaden hierro/folato/zinc/vit C (food-achievable) con:
  - TECHO UL por micro (`_MICRO_CLOSER_UL`) — hierro/zinc tienen toxicidad (no sobre-escalar hígado);
  - presupuesto de kcal COMPARTIDO por día (no per-micro) — con 7 micros, un budget per-micro permitía
    hasta 7× el techo de kcal/día → rompía la banda macro;
  - renal-skip extendido (legumbres/hígado/cítricos cargan K/P/proteína contraindicados en ERC).

Tests: estructura (keys/UL/renal-skip/clamp) + funcional (cierre de hierro bajo UL, skip renal, budget
compartido) — sin Neon.
"""
from __future__ import annotations

from pathlib import Path

import micronutrients
import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Estructura
# ---------------------------------------------------------------------------
def test_new_micros_in_keys_and_mapping():
    for k in ("iron_mg", "folate_mcg", "zinc_mg", "vit_c_mg"):
        assert k in g._MICRO_CLOSER_KEYS, f"{k} debe estar en el set del closer"
        assert g._MICRO_CLOSER_INGREDIENT_KEY.get(k) == k, f"{k} debe mapear identidad al dict de micros"
    # base preservada
    for k in ("fiber_g", "magnesium_mg", "calcium_mg"):
        assert k in g._MICRO_CLOSER_KEYS


def test_ul_caps_present():
    assert g._MICRO_CLOSER_UL.get("iron_mg") == 45.0, "UL hierro 45mg (IOM)"
    assert g._MICRO_CLOSER_UL.get("zinc_mg") == 40.0, "UL zinc 40mg"
    assert g._MICRO_CLOSER_UL.get("folate_mcg") == 1000.0
    assert g._MICRO_CLOSER_UL.get("vit_c_mg") == 2000.0
    # fibra/Mg/Ca NO llevan UL de comida (escalar comida es seguro).
    assert "fiber_g" not in g._MICRO_CLOSER_UL
    assert "magnesium_mg" not in g._MICRO_CLOSER_UL


def test_renal_skip_includes_new_micros_in_source():
    # el renal-skip del closer debe listar los micros cerrables además de Mg/fibra.
    # [P2-MICRO-CLOSER-KEYS-EXT · 2026-07-01] la tupla se reformateó multilinea al añadir K/vitE/omega-3.
    assert '_renal and k in ("magnesium_mg", "fiber_g", "iron_mg", "folate_mcg", "zinc_mg", "vit_c_mg",' in _GRAPH
    assert '"potassium_mg", "vit_e_mg", "omega3_g")' in _GRAPH


def test_shared_kcal_budget_and_ul_clamp_in_source():
    assert "kcal_budget_left" in _GRAPH, "debe usar un presupuesto de kcal compartido por día"
    assert "headroom = _ul - day_total" in _GRAPH, "debe clampear por el techo UL del micro"


# ---------------------------------------------------------------------------
# 2. Funcional (sin Neon)
# ---------------------------------------------------------------------------
class _FakeDB:
    """higado: hierro denso; ostras: zinc denso. Micros escalan con la cantidad líder."""
    _IRON = {"higado": 10.0, "ostras": 0.0, "arroz": 0.0}
    _ZINC = {"higado": 0.0, "ostras": 5.0, "arroz": 0.0}
    _KCAL = {"higado": 130.0, "ostras": 80.0, "arroz": 130.0}

    def _key(self, s):
        low = s.lower()
        for k in self._KCAL:
            if k in low:
                return k
        return None

    def _f(self, s):
        import re as _re
        m = _re.match(r"\s*([\d.]+)", s)
        return (float(m.group(1)) / 100.0) if m else 1.0

    def micros_from_ingredient_string(self, s):
        k = self._key(s)
        if k is None:
            return None
        f = self._f(s)
        return {"grams": 100.0 * f, "iron_mg": self._IRON[k] * f, "zinc_mg": self._ZINC[k] * f}

    def macros_from_ingredient_string(self, s):
        k = self._key(s)
        if k is None:
            return None
        return {"kcal": self._KCAL[k] * self._f(s)}


def _report(gaps):
    return lambda *a, **kw: {"gaps": gaps, "coverage": 0.95, "panel": []}


def test_iron_closed_by_scaling_richest(monkeypatch):
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(micronutrients, "build_micronutrient_report",
                        _report([{"key": "iron_mg", "status": "bajo", "piso": 18.0, "valor": 10.0}]))
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: True)

    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "ingredients": ["100g Higado", "100g Arroz"], "recipe": ["..."]},
    ]}]}
    n = g._close_micro_gaps_for_plan(plan, {"gender": "female"}, db=_FakeDB())
    assert n == 1, "debió cerrar hierro escalando el hígado"
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert ings[0] != "100g Higado", "el ingrediente rico en hierro debió escalarse"
    # factor ×1.6 (MAX_SCALE binda antes que kcal/UL): hierro final 16mg < UL 45 → seguro.
    assert "160" in ings[0], f"escala esperada ×1.6, dio {ings[0]!r}"
    assert ings[1] == "100g Arroz", "el ingrediente sin hierro no se toca"


def test_iron_renal_skipped(monkeypatch):
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(micronutrients, "build_micronutrient_report",
                        _report([{"key": "iron_mg", "status": "bajo", "piso": 18.0, "valor": 10.0}]))
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: True)
    plan = {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "ingredients": ["100g Higado"]}]}]}
    n = g._close_micro_gaps_for_plan(plan, {"medicalConditions": ["enfermedad renal cronica"]}, db=_FakeDB())
    assert n == 0, "en ERC el hierro no se cierra (fuentes cargan K/P/proteína)"


def test_shared_daily_kcal_budget_caps_total(monkeypatch):
    """Con presupuesto de kcal compartido por día, cerrar el PRIMER micro puede agotar el budget →
    el segundo micro del MISMO día no se escala (impide 7× el techo de kcal/día)."""
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_MAX_KCAL_PER_DAY", 50)  # budget chico para forzar el corte
    # iron primero (hígado 130kcal), luego zinc (ostras). Orden de gaps = orden de floors.
    monkeypatch.setattr(micronutrients, "build_micronutrient_report",
                        _report([
                            {"key": "iron_mg", "status": "bajo", "piso": 18.0, "valor": 10.0},
                            {"key": "zinc_mg", "status": "bajo", "piso": 11.0, "valor": 5.0},
                        ]))
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: True)

    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "ingredients": ["100g Higado", "100g Ostras"], "recipe": ["..."]},
    ]}]}
    n = g._close_micro_gaps_for_plan(plan, {"gender": "female"}, db=_FakeDB())
    ings = plan["days"][0]["meals"][0]["ingredients"]
    # hierro: factor min(1.6, 1+50/130=1.38) = 1.38 → ~49.4 kcal añadidas → budget agotado (~0.6 left).
    assert n == 1, f"el budget compartido debió permitir solo 1 escala, dio {n}"
    assert ings[0] != "100g Higado", "el hígado (iron) sí debió escalarse"
    assert ings[1] == "100g Ostras", "las ostras (zinc) NO debieron escalarse: budget del día agotado"
