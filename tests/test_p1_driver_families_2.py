"""[P1-DRIVER-FAMILIES-2 · 2026-07-29] La convergencia por fin ACTÚA (P1-BUDGET-T2-CONVERGENCE)
pero se quedaba corta: en el plan vivo af917e29 sustituyó cangrejo→pescado blanco (−RD$2,106)
y quedó RD$4,456 sobre la referencia porque los drivers restantes no tenían familia:

  · "Queso ricotta" (tarro Sosua) — la familia de quesos solo cubría madurados
    (cheddar/gouda/parmesano/…), ricotta/requesón/mascarpone quedaban fuera.
  · "Yogurt" — ⚠️ la lista costeada usa el nombre CANÓNICO ("Yogurt"), la familia exigía
    "yogurt griego" literal → el driver #2 jamás matcheaba aunque el plan estuviera lleno
    de yogurt griego (mismatch de base de nombres entre ranking y regex).
  · Semillas de girasol / Espárragos / Kale / Granada / Uvas — sin familia alguna.

Familias nuevas (misma categoría culinaria SIEMPRE, mismos guards: ≥30% ahorro vs precio
vivo, alergias, dislikes, day-aware): ricotta→Queso blanco/cottage · yogurt (canónico o
griego)→Yogurt natural (excluye ya-natural) · girasol/calabaza/ajonjolí→Maní/Linaza
(excluye aceite) · espárragos→Vainitas · kale/col rizada→Espinacas · granada/uvas→
Guineo/Lechosa (excluye pasas). Espejo en la tabla estática backstop.

tooltip-anchor: P1-DRIVER-FAMILIES-2
"""
from __future__ import annotations

import pytest

import graph_orchestrator as go

_PRICES = {
    "queso ricotta": 400.0, "ricotta": 400.0, "queso blanco": 120.0, "queso cottage": 140.0,
    "yogurt griego": 260.0, "yogurt natural": 120.0,
    "semillas de girasol": 350.0, "mani": 130.0, "linaza": 110.0,
    "esparragos": 500.0, "vainitas": 90.0,
    "kale": 420.0, "espinacas": 140.0,
    "granada": 600.0, "uvas": 380.0, "guineo": 40.0, "lechosa": 60.0,
}


@pytest.fixture
def _go(monkeypatch):
    monkeypatch.setattr(go, "_budget_build_master_price_map", lambda: dict(_PRICES))
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_ENABLED", True)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_MAX_SUBS", 6)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_TOP_ITEMS", 8)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_MIN_SAVING_PCT", 0.30)
    return go


def _days(*ings):
    return [{"day": 1, "meals": [{
        "name": "Plato de Prueba", "ingredients": list(ings),
        "ingredients_raw": list(ings), "recipe": ["Prepara todo."],
        "protein": 20, "carbs": 30, "fats": 10, "cals": 300}]}]


def test_ricotta_family(_go):
    days = _days("3 cdas de queso ricotta", "1 taza de agua")
    subs = _go._apply_budget_driver_aware_pass(
        days, {}, [{"name": "Queso ricotta", "estimated_cost_rd": 520.0}])
    assert subs == 1
    joined = " ".join(days[0]["meals"][0]["ingredients"]).lower()
    assert "ricotta" not in joined and ("queso blanco" in joined or "cottage" in joined)


def test_yogurt_canonical_driver_name_matches(_go):
    """El bite real: la lista costeada dice 'Yogurt' (canónico), no 'yogurt griego'."""
    days = _days("1 taza de yogurt griego sin azúcar", "½ taza de yogurt natural")
    subs = _go._apply_budget_driver_aware_pass(
        days, {}, [{"name": "Yogurt", "estimated_cost_rd": 220.0}])
    assert subs >= 1
    ings = days[0]["meals"][0]["ingredients"]
    assert not any("griego" in s.lower() for s in ings), ings
    assert any("yogurt natural" in s.lower() for s in ings)
    assert ings[1] == "½ taza de yogurt natural", "el ya-natural queda intacto (exclude)"


def test_girasol_esparragos_kale(_go):
    days = _days("15 g de semillas de girasol", "½ taza de espárragos", "2 tazas de kale fresco")
    subs = _go._apply_budget_driver_aware_pass(
        days, {}, [
            {"name": "Semillas de girasol", "estimated_cost_rd": 300.0},
            {"name": "Espárragos", "estimated_cost_rd": 225.0},
            {"name": "Kale", "estimated_cost_rd": 205.0},
        ])
    assert subs == 3
    joined = " ".join(days[0]["meals"][0]["ingredients"]).lower()
    assert "girasol" not in joined and ("maní" in joined or "mani" in joined or "linaza" in joined)
    assert "espárragos" not in joined and "vainitas" in joined
    assert "kale" not in joined and "espinacas" in joined


def test_granada_uvas_family_and_pasas_excluded(_go):
    days = _days("½ granada (semillas)", "½ taza de uvas rojas", "1 cda de uvas pasas")
    subs = _go._apply_budget_driver_aware_pass(
        days, {}, [{"name": "Granada", "estimated_cost_rd": 290.0},
                   {"name": "Uva", "estimated_cost_rd": 170.0}])
    assert subs >= 2
    ings = days[0]["meals"][0]["ingredients"]
    joined = " ".join(ings).lower()
    assert "granada" not in joined
    assert ings[2] == "1 cda de uvas pasas", "pasas excluidas (despensa barata, no fruta fresca)"


def test_aceite_de_girasol_excluded(_go):
    days = _days("1 cda de aceite de girasol")
    subs = _go._apply_budget_driver_aware_pass(
        days, {}, [{"name": "Aceite de girasol", "estimated_cost_rd": 300.0}])
    assert subs == 0
    assert days[0]["meals"][0]["ingredients"][0] == "1 cda de aceite de girasol"


def test_static_backstop_mirrors():
    blk = dict.fromkeys([])  # noqa
    import pathlib
    src = pathlib.Path(go.__file__).with_suffix(".py").read_text(encoding="utf-8")
    i = src.index("_BUDGET_CHEAP_EQUIVALENTS = (")
    stat = src[i:src.index("def _budget_master_price_per_lb")]
    for tok in ("ricotta", "girasol", "esp[aá]rragos?", "kale", "granadas?"):
        assert tok in stat, f"backstop estático sin espejo: {tok}"
