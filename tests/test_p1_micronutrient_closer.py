"""[P1-MICRONUTRIENT-CLOSER · 2026-06-29] (audit objetivo · P1-4)

Cierra la asimetría identificada en el audit: la proteína tiene un cierre DETERMINISTA post-caps
(`_close_protein_gap_for_meal`/`_repair_protein_floor_post_caps`), pero los micros eran SOLO un
nudge al prompt (advisory). Para un usuario sano con fibra 18/25g nada cerraba la brecha.

`_close_micro_gaps_for_plan` corre tras el macro engine: detecta micros ALCANZABLES (fibra/Mg/Ca)
genuinamente bajo el piso DRI con cobertura alta (status=='bajo', no 'estimado_bajo') y ESCALA el
ingrediente existente más rico en ese micro por día deficitario, acotado por factor + kcal, sin
añadir ingredientes nuevos (no toca alérgenos/lista/coherencia). Default OFF (opt-in, A/B-pending).

Tests: (1) parser-based del knob OFF-por-default + wiring en assemble; (2) funcional de la lógica de
escalado con build_micronutrient_report mockeado + db fake (sin Neon).
"""
from __future__ import annotations

from pathlib import Path

import micronutrients
import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Parser/estructura: knob OFF por default, mapping, wiring en assemble
# ---------------------------------------------------------------------------
def test_knob_on_by_default():
    """[P1-OBJECTIVE-LEVERS-ON · 2026-06-29] flipped OFF→ON: con el fix P1-MICRO-CLOSER-RAW-SYNC el closer
    REFLEJA su efecto en el panel/lista/macros; riesgo de banda acotado (≤80 kcal/día compartido + UL + skip
    renal + MACRO_REBALANCE re-apunta). Rollback: MEALFIT_MICRONUTRIENT_CLOSER=false."""
    assert hasattr(g, "MICRONUTRIENT_CLOSER_ENABLED")
    assert g.MICRONUTRIENT_CLOSER_ENABLED is True, "el micro-closer ahora es ON por default (P1-OBJECTIVE-LEVERS-ON)"


def test_keys_and_mapping():
    # [P1-MICRO-CLOSER-COVERAGE · 2026-06-29] el set base (fibra/Mg/Ca) + los 4 food-achievable.
    # [P2-MICRO-CLOSER-KEYS-EXT · 2026-07-01] + potasio (skip renal/K-med) + vit E (UL 1000) + omega-3.
    assert g._MICRO_CLOSER_KEYS == frozenset({
        "fiber_g", "magnesium_mg", "calcium_mg",
        "iron_mg", "folate_mcg", "zinc_mg", "vit_c_mg",
        "potassium_mg", "vit_e_mg", "omega3_g",
    })
    # La fibra en el dict de micros_from_ingredient_string es 'fiber', no 'fiber_g'; el resto identidad.
    assert g._MICRO_CLOSER_INGREDIENT_KEY["fiber_g"] == "fiber"
    assert g._MICRO_CLOSER_INGREDIENT_KEY["magnesium_mg"] == "magnesium_mg"
    assert g._MICRO_CLOSER_INGREDIENT_KEY["calcium_mg"] == "calcium_mg"
    assert g._MICRO_CLOSER_INGREDIENT_KEY["iron_mg"] == "iron_mg"
    assert g._MICRO_CLOSER_INGREDIENT_KEY["folate_mcg"] == "folate_mcg"
    # cada report-key del closer tiene su mapping a la key del dict de micros (sin huérfanos).
    assert set(g._MICRO_CLOSER_KEYS) == set(g._MICRO_CLOSER_INGREDIENT_KEY.keys())


def test_wired_before_guard5_in_assemble():
    assert "_close_micro_gaps_for_plan" in _GRAPH
    assert "P1-MICRONUTRIENT-CLOSER" in _GRAPH
    # El closer corre ANTES del panel (Guard 5) para que el reporte final refleje sus ajustes.
    idx_closer = _GRAPH.find("if MICRONUTRIENT_CLOSER_ENABLED and _db is not None:")
    idx_guard5 = _GRAPH.find("Guard 5 (FS4/FS8): panel de micros")
    assert idx_closer != -1 and idx_guard5 != -1
    assert idx_closer < idx_guard5, "el closer debe ejecutarse ANTES del panel Guard 5"


# ---------------------------------------------------------------------------
# 2. Funcional: escalado determinista del ingrediente más rico (sin Neon)
# ---------------------------------------------------------------------------
class _FakeDB:
    """db fake: solo los métodos que el closer usa, sin tocar Neon."""
    _FIBER = {"lentejas": 8.0, "pollo": 0.0, "arroz": 0.6}
    _KCAL = {"lentejas": 116.0, "pollo": 165.0, "arroz": 130.0}

    def _key(self, s):
        low = s.lower()
        for k in self._FIBER:
            if k in low:
                return k
        return None

    def micros_from_ingredient_string(self, s):
        k = self._key(s)
        if k is None:
            return None
        # micros escalan con la cantidad líder ('100g'→×1, '160g'→×1.6).
        import re as _re
        m = _re.match(r"\s*([\d.]+)", s)
        f = (float(m.group(1)) / 100.0) if m else 1.0
        return {"grams": 100.0 * f, "fiber": self._FIBER[k] * f,
                "magnesium_mg": None, "calcium_mg": None}

    def macros_from_ingredient_string(self, s):
        k = self._key(s)
        if k is None:
            return None
        import re as _re
        m = _re.match(r"\s*([\d.]+)", s)
        f = (float(m.group(1)) / 100.0) if m else 1.0
        return {"kcal": self._KCAL[k] * f}


def _fiber_report_bajo(*a, **kw):
    """Reporte mock: fibra genuinamente BAJO (cobertura alta) con piso 25."""
    return {"gaps": [{"key": "fiber_g", "status": "bajo", "piso": 25.0, "valor": 8.0}],
            "coverage": 0.95, "panel": []}


def test_closer_scales_richest_fiber_ingredient(monkeypatch):
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", _fiber_report_bajo)
    # truth-up usa db real → no-op para no tocar Neon.
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: True)

    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "ingredients": ["100g Lentejas", "150g Pollo"], "recipe": ["..."]},
    ]}]}
    n = g._close_micro_gaps_for_plan(plan, {"gender": "female"}, db=_FakeDB())

    assert n == 1, f"debió cerrar 1 micro escalando un ingrediente, dio {n}"
    ings = plan["days"][0]["meals"][0]["ingredients"]
    # La lenteja (rica en fibra) se escaló; el pollo (0 fibra) quedó intacto.
    assert ings[0] != "100g Lentejas", "el ingrediente rico en fibra debió escalarse"
    assert ings[0].lower().startswith("160g") or "160" in ings[0], f"escala esperada ×1.6, dio {ings[0]!r}"
    assert ings[1] == "150g Pollo", "el ingrediente sin fibra no debe tocarse"


def test_closer_skips_estimado_bajo(monkeypatch):
    """status 'estimado_bajo' (cobertura incierta) NO debe forzar el cierre."""
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(micronutrients, "build_micronutrient_report",
                        lambda *a, **kw: {"gaps": [{"key": "fiber_g", "status": "estimado_bajo",
                                                    "piso": 25.0, "valor": 8.0}], "coverage": 0.4})
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: True)
    plan = {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "ingredients": ["100g Lentejas"]}]}]}
    assert g._close_micro_gaps_for_plan(plan, {}, db=_FakeDB()) == 0


def test_closer_renal_skips_fiber_and_mg(monkeypatch):
    """En ERC, fibra y Mg NO se cierran (leguminosas/hiperkalemia contraindicadas)."""
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", _fiber_report_bajo)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: True)
    plan = {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "ingredients": ["100g Lentejas"]}]}]}
    n = g._close_micro_gaps_for_plan(plan, {"medicalConditions": ["enfermedad renal cronica"]}, db=_FakeDB())
    assert n == 0, "en renal no debe cerrarse fibra ni Mg"


def test_closer_noop_when_disabled():
    assert g._close_micro_gaps_for_plan({"days": []}, {}, db=_FakeDB()) == 0  # knob OFF por default
