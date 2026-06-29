"""[P1-MICRO-CLOSER-UPDATES · 2026-06-29] (re-audit objetivo · P1)

El closer DETERMINISTA de micros (`_close_micro_gaps_for_plan`) corría SOLO en form-gen (assemble).
swap / regenerate-day / chat-modify recomputaban el panel advisory pero NUNCA cerraban un déficit
re-abierto por la edición (asimetría vs el closer de proteína, que SÍ corre en updates).

Fix: se cablea el closer ANTES de `recompute_micronutrient_report_for_plan` en las 3 superficies,
con `pantry_strict` por superficie (escalar un ingrediente añade comida → contraindicado al cocinar
estrictamente desde la nevera). El closer self-guards en MICRONUTRIENT_CLOSER_ENABLED.

Tests: (1) parser-based del wiring + orden (closer antes del recompute) + paso de pantry_strict en
cada superficie; (2) funcional del skip pantry-strict (sin Neon).
"""
from __future__ import annotations

from pathlib import Path

import micronutrients
import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Parser/estructura: wiring en las 3 superficies
# ---------------------------------------------------------------------------
def test_closer_accepts_pantry_strict_param():
    import inspect
    sig = inspect.signature(g._close_micro_gaps_for_plan)
    assert "pantry_strict" in sig.parameters, "el closer debe aceptar pantry_strict"
    assert sig.parameters["pantry_strict"].default is False, "pantry_strict default False"


def _assert_closer_before_recompute(src: str, surface: str):
    assert "_close_micro_gaps_for_plan" in src, f"{surface}: no cablea el closer de micros"
    idx_close = src.find("_close_micro_gaps_for_plan(")
    idx_recompute = src.find("recompute_micronutrient_report_for_plan(")
    assert idx_close != -1 and idx_recompute != -1, f"{surface}: faltan callsites"
    assert idx_close < idx_recompute, \
        f"{surface}: el closer debe correr ANTES del recompute del panel"


def test_swap_persist_wires_closer():
    # bloque del _swap_mutator de /swap-meal/persist (delimitado por su `result = update_plan_data_atomic`).
    _s = _PLANS.find("def _swap_mutator")
    seg = _PLANS[_s: _PLANS.find("result = update_plan_data_atomic", _s)]
    _assert_closer_before_recompute(seg, "swap-persist")
    assert "pantry_strict=_ps_swap" in seg, "swap debe derivar pantry_strict de la metadata del meal"


def test_regen_day_wires_closer_pantry_strict():
    _s = _PLANS.find("def _day_mutator")
    seg = _PLANS[_s: _PLANS.find("result = update_plan_data_atomic(plan_id, _day_mutator", _s)]
    _assert_closer_before_recompute(seg, "regen-day")
    # regen-day es pantry-strict por diseño → pasa True (el closer se salta, documentado).
    assert "pantry_strict=True" in seg, "regen-day debe pasar pantry_strict=True (cocina desde la nevera)"


def test_chat_modify_wires_closer():
    assert "_close_micro_gaps_for_plan" in _TOOLS, "chat-modify no cablea el closer"
    idx_close = _TOOLS.find("_close_micro_gaps_for_plan(")
    idx_recompute = _TOOLS.find("recompute_micronutrient_report_for_plan(plan_data_fresh")
    assert idx_close != -1 and idx_recompute != -1
    assert idx_close < idx_recompute, "chat-modify: closer antes del recompute"
    assert "_ps_cm = bool(clean_ingredients) and not allow_pantry_expansion" in _TOOLS, \
        "chat-modify debe derivar pantry_strict de la señal canónica del modify"


# ---------------------------------------------------------------------------
# 2. Funcional: skip pantry-strict (sin Neon)
# ---------------------------------------------------------------------------
class _FakeDB:
    _FIBER = {"lentejas": 8.0, "pollo": 0.0}
    _KCAL = {"lentejas": 116.0, "pollo": 165.0}

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
        import re as _re
        m = _re.match(r"\s*([\d.]+)", s)
        f = (float(m.group(1)) / 100.0) if m else 1.0
        return {"grams": 100.0 * f, "fiber": self._FIBER[k] * f}

    def macros_from_ingredient_string(self, s):
        k = self._key(s)
        if k is None:
            return None
        import re as _re
        m = _re.match(r"\s*([\d.]+)", s)
        f = (float(m.group(1)) / 100.0) if m else 1.0
        return {"kcal": self._KCAL[k] * f}


def _fiber_bajo(*a, **kw):
    return {"gaps": [{"key": "fiber_g", "status": "bajo", "piso": 25.0, "valor": 8.0}],
            "coverage": 0.95, "panel": []}


def _plan():
    return {"days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "ingredients": ["100g Lentejas", "150g Pollo"], "recipe": ["..."]},
    ]}]}


def test_pantry_strict_skips_closing(monkeypatch):
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", _fiber_bajo)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: True)

    plan = _plan()
    n = g._close_micro_gaps_for_plan(plan, {"gender": "female"}, db=_FakeDB(), pantry_strict=True)
    assert n == 0, "pantry-strict debe SALTAR el closer (no se puede comprar más desde la nevera)"
    assert plan["days"][0]["meals"][0]["ingredients"][0] == "100g Lentejas", "no debe escalar nada"


def test_non_pantry_strict_closes(monkeypatch):
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", _fiber_bajo)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: True)

    plan = _plan()
    n = g._close_micro_gaps_for_plan(plan, {"gender": "female"}, db=_FakeDB(), pantry_strict=False)
    assert n == 1, "fuera de pantry-strict (ir de compras) el closer SÍ cierra el déficit"
    assert plan["days"][0]["meals"][0]["ingredients"][0] != "100g Lentejas"
