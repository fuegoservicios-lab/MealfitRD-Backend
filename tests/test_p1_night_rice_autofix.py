"""[P1-NIGHT-RICE-AUTOFIX · 2026-06-27] (audit G4) Autofix determinista del "arroz de noche": detectado en vivo
(corr=fda748d8) que la causa #1 de retries del reviewer era el gate P1-SLOT-APPROPRIATENESS rechazando arroz en la
cena → una regeneración completa por intento (tokens). El prompt ya lo prohíbe pero el LLM a veces desobedece.

Fix: `_night_rice_autofix` reescribe el ARROZ simple de la cena por un tubérculo nocturno (batata/yuca/casabe,
rotado por día) — ingrediente Y nombre — ANTES del macro engine, para que el motor lo dimensione y el reviewer no
dispare el retry. NO toca moro/locrio/morito (compuestos → al gate), ni desayuno, ni 'harina/leche de arroz'.
Carb-matched (~×1.4), idempotente, fail-safe. El gate queda como backstop. tooltip-anchor: P1-NIGHT-RICE-AUTOFIX
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(g.__file__).resolve().parent


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.search(r"(\d+)\s*g", s)
        return float(m.group(1)) if m else 150.0


def _wire(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    return g


def _cena(name, ings, day=0):
    return [{"day": day, "meals": [{"meal": "Cena", "name": name, "ingredients": ings}]}]


def test_rewrites_simple_rice_ingredient_and_name(monkeypatch):
    _wire(monkeypatch)
    days = _cena("Pollo a la Plancha con Arroz Blanco", ["120 g de Pollo", "1 taza de arroz blanco (150g)"])
    n = g._night_rice_autofix(days, db=_DB())
    assert n == 1
    meal = days[0]["meals"][0]
    assert "arroz" not in meal["name"].lower()
    assert "batata" in meal["name"].lower()  # día 0 → primer tubérculo de la rotación
    assert not any("arroz" in i.lower() for i in meal["ingredients"])
    assert any("batata" in i.lower() for i in meal["ingredients"])


def test_carb_matched_grams(monkeypatch):
    _wire(monkeypatch)
    days = _cena("Cena con Arroz", ["arroz blanco 100g"])
    g._night_rice_autofix(days, db=_DB())
    # 100g arroz × 1.4 ≈ 140g tubérculo (carb-match)
    ing = days[0]["meals"][0]["ingredients"][0].lower()
    assert "140 g" in ing


def test_does_not_touch_moro_locrio(monkeypatch):
    """Platos compuestos (moro/locrio/morito) se dejan al gate — reescribir su nombre quedaría torpe."""
    _wire(monkeypatch)
    days = _cena("Moro de Guandules con Cerdo", ["1 taza de moro de guandules (200g)", "Cerdo"])
    assert g._night_rice_autofix(days, db=_DB()) == 0
    assert "moro" in days[0]["meals"][0]["name"].lower()


def test_does_not_touch_breakfast(monkeypatch):
    _wire(monkeypatch)
    days = [{"day": 0, "meals": [{"meal": "Desayuno", "name": "Mangú con Arroz", "ingredients": ["arroz 100g"]}]}]
    assert g._night_rice_autofix(days, db=_DB()) == 0


def test_respects_rice_modifier_exclusions(monkeypatch):
    """'harina/leche de arroz' NO son arroz de noche (modificadores) → no se tocan."""
    _wire(monkeypatch)
    days = _cena("Panqueques de Harina de Arroz", ["harina de arroz 80g"])
    assert g._night_rice_autofix(days, db=_DB()) == 0


def test_idempotent(monkeypatch):
    _wire(monkeypatch)
    days = _cena("Pollo con Arroz Blanco", ["arroz blanco 150g"])
    assert g._night_rice_autofix(days, db=_DB()) == 1
    assert g._night_rice_autofix(days, db=_DB()) == 0  # 2da pasada no cambia nada


def test_rotation_varies_by_day(monkeypatch):
    """La rotación por día evita el mismo tubérculo en cenas consecutivas (variedad cross-day)."""
    _wire(monkeypatch)
    days = (_cena("Cena con Arroz", ["arroz 100g"], day=0)
            + _cena("Otra Cena con Arroz", ["arroz 100g"], day=1))
    g._night_rice_autofix(days, db=_DB())
    n0 = days[0]["meals"][0]["name"].lower()
    n1 = days[1]["meals"][0]["name"].lower()
    assert not (("batata" in n0) and ("batata" in n1))  # distintos tubérculos


def test_knob_off_disables(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(g, "NIGHT_RICE_AUTOFIX_ENABLED", False)
    days = _cena("Pollo con Arroz Blanco", ["arroz blanco 150g"])
    assert g._night_rice_autofix(days, db=_DB()) == 0


def test_callsite_runs_before_macro_engine():
    """El autofix DEBE correr ANTES de _apply_macro_engine (para que el motor dimensione el tubérculo)."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    body = src[src.index("async def assemble_plan_node"):]
    body = body[:body.index("\nasync def ", 5) if "\nasync def " in body[5:] else len(body)]
    i_fix = body.index("_night_rice_autofix(days)")
    i_engine = body.index("_apply_macro_engine(result, days")
    assert i_fix < i_engine
