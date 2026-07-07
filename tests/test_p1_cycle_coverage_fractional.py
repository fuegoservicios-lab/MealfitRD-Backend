"""[P1-CYCLE-COVERAGE-FRACTIONAL · 2026-07-06] Cobertura 28-vs-30 días del ciclo.

Bug detectado en la review visual del plan de 30 días:
  `_CYCLE_WEEKS_BY_DURATION = {monthly: 4}` era floor(30/7) → 4 semanas = 28 días.
  Los días 29-30 del ciclo quedaban SIN costear (costo sub-estimado ~2 días de
  perecederos) NI cubiertos en el mensaje ("recompra 4 veces"). biweekly igual:
  floor(15/7)=2 → día 15 sin cubrir.

Fix — modelo que separa DOS conceptos:
  - Multiplicador de COSTO = días/7 FRACCIONAL (30/7=4.286, 15/7=2.143). Honesto
    para los días reales. NO ceil: comprar 5 semanas completas para 30 días
    sobre-estima y podría disparar un banner "excedido" falso.
  - Nº de IDAS mostradas = ceil(días/7) (30d=5, 15d=3, 7d=1). Cuántas veces el
    usuario recompra perecederos; la última ida es parcial.

Espejo frontend: Dashboard.jsx (_cycleCostMultiplier + _cycleTrips).
tooltip-anchor: P1-CYCLE-COVERAGE-FRACTIONAL
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load():
    os.environ.setdefault("GEMINI_API_KEY", "dummy")
    os.environ.setdefault("SUPABASE_URL", "https://dummy.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "dummy")
    os.environ.setdefault("CRON_SECRET", "dummy")
    sys.path.insert(0, str(_BACKEND_ROOT))
    import shopping_calculator as sc
    return sc


# ---------------------------------------------------------------------------
# 1. Helpers: multiplicador fraccional + ceil trips
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("duration,expected_mult", [
    ("weekly", 1.0),
    ("biweekly", 15 / 7),
    ("monthly", 30 / 7),
])
def test_cost_multiplier_fractional(duration, expected_mult):
    sc = _load()
    assert sc._cycle_cost_multiplier(duration) == pytest.approx(expected_mult)


def test_monthly_multiplier_not_floor_4():
    """El regresión-guard central: el multiplicador de costo mensual NO debe ser
    el floor 4 (dejaba días 29-30 sin costear)."""
    sc = _load()
    assert sc._cycle_cost_multiplier("monthly") > 4.0
    assert sc._cycle_cost_multiplier("monthly") == pytest.approx(30 / 7)


@pytest.mark.parametrize("duration,expected_trips", [
    ("weekly", 1),
    ("biweekly", 3),
    ("monthly", 5),
])
def test_trip_count_ceil(duration, expected_trips):
    sc = _load()
    assert sc._cycle_trip_count(duration) == expected_trips


def test_trips_ge_cost_multiplier():
    """Las idas (ceil) siempre ≥ el multiplicador de costo (fraccional): la
    última ida es parcial, nunca se cobra de más pero se cubren todos los días."""
    sc = _load()
    for d in ("weekly", "biweekly", "monthly"):
        assert sc._cycle_trip_count(d) >= sc._cycle_cost_multiplier(d)
        assert sc._cycle_trip_count(d) == math.ceil(sc._cycle_cost_multiplier(d))


def test_unknown_duration_defaults_weekly():
    sc = _load()
    assert sc._cycle_cost_multiplier("bogus") == 1.0
    assert sc._cycle_trip_count("bogus") == 1


# ---------------------------------------------------------------------------
# 2. E2E: compute_shopping_cost_summary usa el modelo nuevo
# ---------------------------------------------------------------------------
def _items(*pairs):
    return [{"name": f"i{i}", "estimated_cost_rd": c, "is_perishable": p}
            for i, (c, p) in enumerate(pairs)]


def test_summary_monthly_covers_full_30_days():
    """El costo mensual del ciclo cubre 30 días (perecederos × 30/7), no 28."""
    sc = _load()
    monthly = _items((100.0, True), (800.0, False))  # 100 perecedero, 800 estable
    s = sc.compute_shopping_cost_summary(_items((100.0, True)), _items((100.0, True)), monthly, "monthly")
    m = s["by_duration"]["monthly"]
    assert m["cycle_trips"] == 5
    assert m["cycle_weeks"] == round(30 / 7, 3)
    # costo = 800 estable + 100 perecedero × 30/7 = 800 + 428.57 = 1228.57 (NO 1200 del floor-4)
    assert m["cycle_total_rd"] == round(800.0 + 100.0 * (30 / 7), 2)
    assert m["cycle_total_rd"] > 800.0 + 100.0 * 4  # estrictamente mayor que el floor viejo


def test_no_over_estimate_vs_ceil():
    """El costo mensual NO usa ceil (5×): eso sobre-estimaría y podría disparar
    un 'excedido' falso. Debe ser estrictamente menor que perecederos × 5."""
    sc = _load()
    monthly = _items((100.0, True), (800.0, False))
    s = sc.compute_shopping_cost_summary(_items((100.0, True)), _items((100.0, True)), monthly, "monthly")
    m = s["by_duration"]["monthly"]
    assert m["cycle_total_rd"] < 800.0 + 100.0 * 5
