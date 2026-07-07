"""[P1-COHERENCE-OVERSUPPLY-STAPLES · 2026-07-07] Extiende el filtro pkg-noise a la
sobre-oferta de STAPLES/granos/vegetales (no solo condimentos).

Logs en vivo (post P1-COHERENCE-PACKAGING-NOISE): el COH-GUARD/warn seguía mostrando ~67
divergencias `unknown` por plan — sobre-oferta de envase de STAPLES (Arroz, Repollo, Ajo,
Avena, Cebada comprados por bolsa/cabeza) que el filtro solo-condimentos no capturaba.
Warn-only (no bloqueaba tras el fix del T2), pero ruido de telemetría.

Fix: la sobre-oferta (lista > receta) es RUIDO de granularidad de envase para TODO excepto
PROTEÍNA vendida por PESO (carne/pescado/aves), donde un over-buy (pollo 4×) sí es señal
real de costo. Enlatados/huevos (rounding de lata/cartón) y frutas/veg (caps P5/P6) NO son
proteína-por-peso → su sobre-oferta se filtra.
tooltip-anchor: P1-COHERENCE-OVERSUPPLY-STAPLES
"""
from __future__ import annotations

from pathlib import Path

import pytest

import shopping_calculator as sc

_SRC = (Path(sc.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def no_master_db(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: [])


def _plan(ingredients_raw, agg_list):
    return {
        "days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": ingredients_raw}]}],
        "aggregated_shopping_list": agg_list,
    }


def _mag(plan, mult=1.0):
    divs = sc.run_shopping_coherence_guard(plan, multiplier=mult)
    return [d for d in divs if d.get("magnitude")]


# --- Parser-based ---
def test_marker_and_protein_regex_present():
    assert "P1-COHERENCE-OVERSUPPLY-STAPLES" in _SRC
    assert "_COHERENCE_OVERSUPPLY_PROTEIN_KEEP_RE" in _SRC


# --- Funcional: staples over-supply filtrados ---
@pytest.mark.parametrize("food", ["Arroz", "Avena", "Repollo", "Ajo", "Cebada"])
def test_staple_over_supply_filtered(food, monkeypatch):
    """Sobre-oferta de un STAPLE/grano/veg (envase/cabeza entera) → filtrada (0 magnitud)."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.delenv("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", raising=False)
    plan = _plan([f"200 g {food.lower()}"],
                 [{"name": food, "market_qty_numeric": 900, "market_unit": "g"}])
    assert _mag(plan) == [], f"la sobre-oferta de envase de {food} no debe reportarse"


def test_staple_over_supply_reappears_knob_off(monkeypatch):
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.setenv("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", "false")
    plan = _plan(["200 g arroz"], [{"name": "Arroz", "market_qty_numeric": 900, "market_unit": "g"}])
    assert len(_mag(plan)) == 1, "sin el filtro, la sobre-oferta vuelve a reportarse"


# --- Regresión: proteína por peso PRESERVADA (señal de over-buy) ---
@pytest.mark.parametrize("food", ["Pollo", "Pavo", "Tilapia", "Res", "Pescado blanco"])
def test_protein_over_supply_preserved(food, monkeypatch):
    """Sobre-oferta de PROTEÍNA por peso (pollo 4×) NO se filtra — es over-buy real."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.delenv("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", raising=False)
    plan = _plan([f"200 g {food.lower()}"],
                 [{"name": food, "market_qty_numeric": 800, "market_unit": "g"}])
    assert len(_mag(plan)) >= 1, f"la sobre-oferta de {food} (proteína) debe preservarse"


def test_under_supply_of_staple_preserved(monkeypatch):
    """REGRESIÓN: la FALTA de un staple (arroz 1000→200) NO se filtra — es señal real."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.delenv("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", raising=False)
    plan = _plan(["1000 g arroz"], [{"name": "Arroz", "market_qty_numeric": 200, "market_unit": "g"}])
    mag = _mag(plan)
    assert len(mag) == 1 and mag[0]["hypothesis"] == "pantry_overdeduct"
