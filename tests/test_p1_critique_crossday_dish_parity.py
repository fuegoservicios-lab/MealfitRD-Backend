"""[P1-CRITIQUE-CROSSDAY-DISH-PARITY · 2026-07-09] Self-critique detecta plato-base repetido cross-day.

Forense plan vivo 1273aecb (angelobrito500, 2026-07-09): el intento #1 fue RECHAZADO por el gate del revisor
`P2-CROSSDAY-VARIETY-GATE` ("'revoltillo' en 3 días") → retry completo. El self-critique detectaba staples-
INGREDIENTE repetidos (yogurt/pan/queso) pero NO platos-BASE repetidos entre días (revoltillo ×3), así que
skip-when-clean lo dejaba pasar hasta el reviewer.

Fix: extiende la paridad — el self-critique reusa el MISMO SSOT del gate (`build_variety_report.cross_day_dishes`,
umbral CROSS_DAY_DISH_GATE_MIN_DAYS) para (a) NO saltar el evaluador cuando hay repetición de plato-base, y
(b) inyectar la corrección al prompt del corrector (que ya tiene el contrato de variedad). Knob
`MEALFIT_CRITIQUE_CROSSDAY_DISH_PARITY` (default True).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ───────────────────────── parser-based ─────────────────────────

def test_marker_present():
    assert "P1-CRITIQUE-CROSSDAY-DISH-PARITY" in _GO


def test_knob_default_on():
    assert (
        'CRITIQUE_CROSSDAY_DISH_PARITY_ENABLED = _env_bool("MEALFIT_CRITIQUE_CROSSDAY_DISH_PARITY", True)'
        in _GO
    )


def test_crossday_in_skip_gate():
    """El detector cross-day-dish debe formar parte del gate skip-when-clean (no saltar si hay repetición)."""
    i_skip = _GO.index("SELF_CRITIQUE_SKIP_WHEN_CLEAN and not staple_repetitions")
    window = _GO[i_skip:i_skip + 400]
    assert "cross_day_dish_repeats" in window, "el skip gate debe considerar cross_day_dish_repeats"


def test_crossday_block_wired_into_prompt():
    assert "crossday_block" in _GO
    # el bloque debe inyectarse al human_content del evaluador
    i_hc = _GO.index("PLAN A EVALUAR")
    window = _GO[i_hc - 400:i_hc + 400]
    assert "crossday_block" in window


# ───────────────────────── funcional (parity SSOT) ─────────────────────────

@pytest.fixture()
def g():
    import graph_orchestrator as _g
    return _g


def _day(n, breakfast, lunch, dinner):
    return {"day": n, "meals": [
        {"meal": "desayuno", "name": breakfast, "ingredients": ["huevo", "cebolla"]},
        {"meal": "almuerzo", "name": lunch, "ingredients": ["pollo", "arroz"]},
        {"meal": "cena", "name": dinner, "ingredients": ["pescado", "yuca"]},
    ]}


def test_gate_detects_crossday_dish_revoltillo(g):
    """SSOT que el fix reusa: revoltillo en 3 días → cross_day_dishes lo reporta (umbral=3).
    Solo el desayuno (revoltillo) repite; almuerzo/cena varían para aislar la señal."""
    plan = {"days": [
        _day(1, "Revoltillo de huevo con tomate", "Pollo guisado con arroz", "Pescado al horno con yuca"),
        _day(2, "Revoltillo de huevo con espinaca", "Res encebollada con moro", "Tilapia frita con ensalada"),
        _day(3, "Revoltillo de huevo con cebolla", "Cerdo al horno con batata", "Camarones al ajillo con quinoa"),
    ]}
    report = g.build_variety_report(plan)
    cdd = report.get("cross_day_dishes") or {}
    assert any("revoltillo" in str(k).lower() for k in cdd), f"esperaba revoltillo en cross_day_dishes, got {cdd}"


def test_gate_clean_when_dishes_vary(g):
    """Platos-base distintos por día en TODOS los slots → cross_day_dishes vacío."""
    plan = {"days": [
        _day(1, "Revoltillo de huevo con tomate", "Pollo guisado con arroz", "Pescado al horno con yuca"),
        _day(2, "Panqueques de avena con guineo", "Res encebollada con moro", "Tilapia frita con ensalada"),
        _day(3, "Mangú con queso frito", "Cerdo al horno con batata", "Camarones al ajillo con quinoa"),
    ]}
    report = g.build_variety_report(plan)
    assert not (report.get("cross_day_dishes") or {}), f"esperaba cross_day_dishes vacío, got {report.get('cross_day_dishes')}"
