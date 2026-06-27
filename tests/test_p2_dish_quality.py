"""[P2-DISH-QUALITY · 2026-06-27] (audit Fase 3 / G5) Telemetría ADVISORY de 'realness' de los platos:
mide cuántos parecen placeholder/crudo (nombre genérico, ingredientes-placeholder, receta no sustantiva)
en vez de platos reales cocinados — el objetivo más débil del owner (creatividad). Análogo a band_score
para macros: antes G5 no tenía NINGUNA medición. NUNCA es gate (no rechaza planes).

NB (verificación adversaria de la auditoría): NO se cabla `dominican_dishes.json` — está muerto POR DISEÑO
(el LLM itemiza, no emite 'locrío' como línea); la creatividad ya viene de la rotación del catálogo + las
plantillas de prompt (planner.py / day_generator.py). Este fix MIDE G5, no cambia la generación.

Cubre:
  - graph_orchestrator._meal_dish_quality_issue (detector per-comida)
  - graph_orchestrator.compute_dish_quality_report (telemetría plan-level)
  - parser-anchored: telemetría cableada en assemble + knob default ON + marca de degradación honesta
"""
from __future__ import annotations

from pathlib import Path

import pytest

import graph_orchestrator as go

_GO = Path(go.__file__).resolve()

_REAL_DISH = {
    "name": "Pollo Guisado a la Criolla con Moro de Gandules",
    "ingredients": ["200g pollo", "1 taza arroz", "1/2 taza gandules"],
    "recipe": [
        "Mise en place: Sazona el pollo con ajo, orégano y agrio; pica cebolla y ají",
        "El Toque de Fuego: Sofríe el pollo 8 min y agrega los gandules con el arroz",
        "Montaje: Sirve el moro con el pollo encima y decora con cilantro",
    ],
}


def test_real_dominican_dish_is_high_quality():
    low, why = go._meal_dish_quality_issue(_REAL_DISH)
    assert low is False and why is None


@pytest.mark.parametrize("meal,frag", [
    ({"name": "Plato del Almuerzo", "ingredients": ["200g pollo"],
      "recipe": ["Mise en place: Sazona el pollo con especias variadas",
                 "El Toque de Fuego: Cocina a fuego medio por 10 minutos"]}, "nombre placeholder"),
    ({"name": "Almuerzo Energético", "ingredients": ["Proteína magra al gusto", "Carbohidratos complejos"],
      "recipe": ["Mise en place: algo"]}, "ingredientes placeholder"),
    ({"name": "Ensalada Fresca", "ingredients": ["lechuga", "tomate"],
      "recipe": ["Mise en place: Preparar todo", "El Toque de Fuego: Cocinar", "Montaje: Servir"]}, "receta no sustantiva"),
    ({"name": "", "ingredients": ["x"], "recipe": ["a", "b"]}, "nombre vacío"),
])
def test_placeholder_or_raw_meals_flagged(meal, frag):
    low, why = go._meal_dish_quality_issue(meal)
    assert low is True
    assert frag in why


def test_report_counts_and_ratio():
    plan = {"days": [{"day": 1, "meals": [
        _REAL_DISH,
        {"name": "Cena del Día", "ingredients": ["pollo"], "recipe": ["Mise en place: cocina el pollo entero"]},
        {"name": "X", "ingredients": ["Vegetales mixtos"], "recipe": ["Mise en place: x"]},
    ]}]}
    rep = go.compute_dish_quality_report(plan)
    assert rep["total_meals"] == 3
    assert rep["low_quality_meals"] == 2
    assert rep["low_quality_ratio"] == round(2 / 3, 3)
    assert len(rep["issues"]) == 2


def test_report_failsafe_on_garbage():
    assert go.compute_dish_quality_report({}) == {"total_meals": 0, "low_quality_meals": 0,
                                                  "low_quality_ratio": None, "issues": []}


def test_knob_default_on():
    assert go.DISH_QUALITY_TELEMETRY_ENABLED is True


# ──────────────────────────── parser-anchored ────────────────────────────

def test_telemetry_wired_in_assemble():
    src = _GO.read_text(encoding="utf-8")
    assert "P2-DISH-QUALITY" in src
    assert 'plan["dish_quality_report"] = _dq' in src
    assert 'DISH_QUALITY_TELEMETRY_ENABLED = _env_bool("MEALFIT_DISH_QUALITY_TELEMETRY", True)' in src


def test_honest_degradation_marker_in_autofill():
    src = _GO.read_text(encoding="utf-8")
    assert 'm["_dish_quality_degraded"] = True' in src
