"""[P1-COHERENCE-PACKAGING-NOISE · 2026-07-07] El coherence guard reportaba
61-69 divergencias de magnitud warn (presence=0) por plan, dominadas por RUIDO
ESTRUCTURAL de granularidad de envase — no faltas accionables.

Review visual del plan de 30 días (log en vivo):
  🛒 [COH-GUARD/warn] 61 divergencias (presence=0, magnitude=61). Hipótesis:
     {'unknown': 33, 'unit_mismatch': 28}. Sample: Pimienta negra; Semillas de
     girasol; Aceite de oliva; Pan integral familiar; Tortilla integral ...

Forense (plan 5f80f797): receta "20 g semillas de girasol" → lista "1 funda
(200 gr)"; "3 lonjas de pan integral" → "1 paquete (745 gr)"; "Pimienta al gusto"
→ "1 sobre (0.5 Oz)". Todas son sobre-oferta por envase mínimo (la receta es
cocinable) o unit_mismatch (sobre/lonja no convertible a gramos).

Fix: filtro post-comparador junto al cap-aware que descarta del set de magnitud:
  (a) unit_mismatch (ya excluido del block; ruido en warn), y
  (b) unknown SOBRE-oferta (actual > expected).
Preserva yield_uncovered (banda diagnóstica), pantry_overdeduct y unknown por
DEBAJO (falta real), y toda la capa presence. Knob
MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER (default True).
tooltip-anchor: P1-COHERENCE-PACKAGING-NOISE
"""
from __future__ import annotations

from pathlib import Path

import pytest

import shopping_calculator
from shopping_calculator import run_shopping_coherence_guard

_SRC = (Path(shopping_calculator.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def no_master_db(monkeypatch):
    """Stub master DB → fallback sólo-reglas (mismo patrón que
    test_p1_shopping_recipe_coherence.py). Evita ERROR al pool + latencia."""
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [])


def _plan(ingredients_raw, agg_list):
    return {
        "days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": ingredients_raw}]}],
        "aggregated_shopping_list": agg_list,
    }


# ---------------------------------------------------------------------------
# 1. Parser-based: marker + knob + posición del filtro
# ---------------------------------------------------------------------------
def test_marker_and_knob_present():
    assert "P1-COHERENCE-PACKAGING-NOISE" in _SRC
    assert 'MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER' in _SRC


def test_filter_runs_after_cap_aware_before_extend():
    """El filtro debe correr DESPUÉS del cap-aware y ANTES de extend(magnitude_divs)."""
    i_cap = _SRC.index("P1-CAPS-COHERENCE-RECONCILE")
    i_pkg = _SRC.index("P1-COHERENCE-PACKAGING-NOISE")
    i_extend = _SRC.index("divergences.extend(magnitude_divs)")
    assert i_cap < i_pkg < i_extend, "El filtro pkg-noise está mal posicionado."


# ---------------------------------------------------------------------------
# 2. Funcional: se filtra el ruido, se preserva la señal
# ---------------------------------------------------------------------------
def test_condiment_over_supply_filtered(monkeypatch):
    """Receta 20g semillas de girasol, lista 200g (envase mínimo, sobre-oferta 10×).
    Con el filtro ON: 0 divergencias de magnitud + sin block."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
    monkeypatch.delenv("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", raising=False)
    plan = _plan(
        ["20 g semillas de girasol"],
        [{"name": "Semillas de girasol", "market_qty_numeric": 200, "market_unit": "g"}],
    )
    divs = run_shopping_coherence_guard(plan, multiplier=1.0)
    mag = [d for d in divs if d.get("magnitude")]
    assert mag == [], f"la sobre-oferta de un ítem de envase-mínimo no debe reportarse: {mag}"
    assert "_shopping_coherence_block" not in plan, "no debe bloquear por sobre-oferta de envase"


def test_condiment_over_supply_kept_when_knob_off(monkeypatch):
    """Con el filtro OFF, la misma sobre-oferta vuelve a reportarse (rollback).
    (También valida que semillas SÍ pasa la verificación — si no, saldría 0)."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
    monkeypatch.setenv("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", "false")
    plan = _plan(
        ["20 g semillas de girasol"],
        [{"name": "Semillas de girasol", "market_qty_numeric": 200, "market_unit": "g"}],
    )
    divs = run_shopping_coherence_guard(plan, multiplier=1.0)
    mag = [d for d in divs if d.get("magnitude")]
    assert len(mag) == 1, f"sin el filtro la sobre-oferta debe reportarse: {mag}"
    assert mag[0]["actual_qty"] > mag[0]["expected_qty"]  # confirma que era sobre-oferta


def test_real_food_over_supply_preserved(monkeypatch):
    """REGRESIÓN clave: la sobre-oferta de COMIDA REAL (pavo 200g→800g, 4×) NO se
    filtra — no es keyword de envase-mínimo; sigue siendo señal (multiplier/over-buy)."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
    monkeypatch.delenv("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", raising=False)
    plan = _plan(
        ["200 g pavo"],
        [{"name": "Pavo", "market_qty_numeric": 800, "market_unit": "g"}],
    )
    divs = run_shopping_coherence_guard(plan, multiplier=1.0)
    mag = [d for d in divs if d.get("magnitude") and d["food"] == "Pavo"]
    assert len(mag) >= 1, f"la sobre-oferta de pavo (comida real) debe preservarse: {divs}"


def test_under_supply_preserved(monkeypatch):
    """REGRESIÓN: falta real (receta 1000g, lista 200g → pantry_overdeduct) NO se
    filtra — es el peligro 'qty mitad' que el guard debe cazar."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
    monkeypatch.delenv("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", raising=False)
    plan = _plan(
        ["1000 g arroz"],
        [{"name": "Arroz", "market_qty_numeric": 200, "market_unit": "g"}],
    )
    divs = run_shopping_coherence_guard(plan, multiplier=1.0)
    mag = [d for d in divs if d.get("magnitude")]
    assert len(mag) == 1 and mag[0]["hypothesis"] == "pantry_overdeduct"


def test_yield_uncovered_preserved(monkeypatch):
    """REGRESIÓN: sobre-oferta en la banda diagnóstica yield (1.35×) NO se filtra —
    su hipótesis es yield_uncovered, no unknown."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
    monkeypatch.delenv("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", raising=False)
    plan = _plan(
        ["1 lb pollo"],
        [{"name": "Pollo", "market_qty_numeric": 1.35, "market_unit": "lb"}],
    )
    divs = run_shopping_coherence_guard(plan, multiplier=1.0)
    mag = [d for d in divs if d.get("magnitude")]
    assert len(mag) == 1 and mag[0]["hypothesis"] == "yield_uncovered"


def test_unit_mismatch_filtered_at_guard(monkeypatch):
    """unit_mismatch (alimento en la lista bajo unidad de envase no convertible)
    se filtra del warn del guard (converter OFF para forzar el caso cda↔unidad)."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
    monkeypatch.delenv("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", raising=False)
    monkeypatch.setenv("MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED", "false")
    plan = _plan(
        ["3 cda canela"],
        [{"name": "Canela", "market_qty_numeric": 1, "market_unit": "unidad"}],
    )
    divs = run_shopping_coherence_guard(plan, multiplier=1.0)
    mag = [d for d in divs if d.get("magnitude") and d.get("hypothesis") == "unit_mismatch"]
    assert mag == [], f"unit_mismatch no debe reportarse en warn: {mag}"
