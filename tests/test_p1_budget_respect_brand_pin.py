"""[P1-BUDGET-RESPECT-BRAND-PIN · 2026-07-29] Test ancla (fix round, finding 1 — lens
budget). La convergencia de presupuesto (`_apply_budget_driver_aware_pass` /
`_apply_budget_cheapen_pass`) rankeaba y sustituía ítems SIN mirar si el usuario fijó
una marca para ese alimento via 'Marcas del súper' (`user_brand_preferences`). Dos
fallos que esto cierra:

  (a) un alimento SIN familia de sustitución (ej. aceite — excluido a propósito de
      TODAS las familias) que además está pineado no puede tocarse, pero su costo
      seguía empujando la ranking de drivers → el pase gastaba sus sustituciones en
      OTRA comida que el usuario nunca eligió cambiar, sin ningún aviso de que la
      causa era una elección de marca en otro panel.
  (b) un alimento CON familia (salmón, queso, almendras...) que el usuario SÍ pineó
      se sustituía igual, pisando en silencio una decisión explícita.

Ambos se cierran con `_budget_pinned_food_keys` (oráculo de solo-lectura sobre
`fetch_brand_pref_packages`, la misma API que ya usa el costeo) + `_budget_food_is_
brand_pinned` (misma normalización/contención que `_resolve_brand_pref`), consultados
en los dos pases ANTES de rankear/sustituir.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as _f:
    _GO_SRC = _f.read()


# ---------------------------------------------------------------------------
# 1. Anclas de source
# ---------------------------------------------------------------------------

def test_marker_anchored():
    assert "P1-BUDGET-RESPECT-BRAND-PIN" in _GO_SRC
    assert "def _budget_pinned_food_keys(" in _GO_SRC
    assert "def _budget_food_is_brand_pinned(" in _GO_SRC


def test_both_passes_call_the_guard():
    cheapen_start = _GO_SRC.index("def _apply_budget_cheapen_pass(")
    cheapen_end = _GO_SRC.index("def apply_budget_convergence_for_days(")
    cheapen_block = _GO_SRC[cheapen_start:cheapen_end]
    assert "_budget_pinned_food_keys(form_data)" in cheapen_block
    assert "_budget_food_is_brand_pinned(" in cheapen_block

    driver_start = _GO_SRC.index("def _apply_budget_driver_aware_pass(")
    driver_block = _GO_SRC[driver_start:driver_start + 6000]
    assert "_budget_pinned_food_keys(form_data)" in driver_block
    assert "_budget_food_is_brand_pinned(" in driver_block


# ---------------------------------------------------------------------------
# 2. Helpers unitarios
# ---------------------------------------------------------------------------

@pytest.fixture()
def _go():
    import graph_orchestrator as go
    return go


def test_pinned_food_keys_fail_open_no_user(_go):
    assert _go._budget_pinned_food_keys({}) == set()
    assert _go._budget_pinned_food_keys({"user_id": "guest"}) == set()
    assert _go._budget_pinned_food_keys(None) == set()


def test_pinned_food_keys_wraps_fetch_brand_pref_packages(_go, monkeypatch):
    monkeypatch.setattr(
        "shopping_calculator.fetch_brand_pref_packages",
        lambda uid: {"aceite de oliva": {"grams": 5000, "price": 4500}, "aceite": {}},
    )
    assert _go._budget_pinned_food_keys({"user_id": "u1"}) == {"aceite de oliva", "aceite"}


def test_pinned_food_keys_fail_open_on_exception(_go, monkeypatch):
    def _boom(uid):
        raise RuntimeError("db down")
    monkeypatch.setattr("shopping_calculator.fetch_brand_pref_packages", _boom)
    assert _go._budget_pinned_food_keys({"user_id": "u1"}) == set()


def test_food_is_brand_pinned_exact_and_containment(_go):
    pinned = {"aceite de oliva"}
    assert _go._budget_food_is_brand_pinned("Aceite de oliva", pinned) is True
    assert _go._budget_food_is_brand_pinned("aceite de oliva extra virgen", pinned) is True
    assert _go._budget_food_is_brand_pinned("Sal", pinned) is False
    assert _go._budget_food_is_brand_pinned("", pinned) is False
    assert _go._budget_food_is_brand_pinned("Aceite de oliva", set()) is False


def test_food_is_brand_pinned_short_key_word_boundary_safe(_go):
    """'sal' NO matchea 'salsa' — misma escalera que _resolve_brand_pref."""
    pinned = {"salsa de tomate"}
    assert _go._budget_food_is_brand_pinned("sal", pinned) is False


# ---------------------------------------------------------------------------
# 3. Funcional: driver-aware pass
# ---------------------------------------------------------------------------

_PRICES = {
    "queso cheddar": 400.0,
    "queso blanco": 90.0,
    "salmon": 500.0,
    "filete de pescado blanco": 150.0,
    "aceite de oliva": 4000.0,
}


def _mk_days():
    return [
        {
            "day": 1,
            "meals": [
                {
                    "name": "Salmón a la plancha con aceite de oliva",
                    "ingredients": [
                        "120 g de salmón",
                        "40 g de queso cheddar",
                        "1 cda de aceite de oliva",
                    ],
                    "ingredients_raw": [
                        "120 g de salmón",
                        "40 g de queso cheddar",
                        "1 cda de aceite de oliva",
                    ],
                    "recipe": ["Sella el salmón 4 min por lado con aceite de oliva."],
                }
            ],
        }
    ]


@pytest.fixture()
def _go_wired(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "_budget_build_master_price_map", lambda: dict(_PRICES))
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_ENABLED", True)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_MAX_SUBS", 5)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_TOP_ITEMS", 8)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_MIN_SAVING_PCT", 0.30)
    return go


def test_pinned_expensive_oil_excluded_from_driver_pool_real_driver_still_fires(_go_wired, monkeypatch):
    """Aceite pineado (RD$4000, sin familia — no puede tocarse de todos modos) NO
    debe impedir que el driver real (salmón, SÍ tiene familia) se sustituya."""
    monkeypatch.setattr(
        "shopping_calculator.fetch_brand_pref_packages",
        lambda uid: {"aceite de oliva": {"grams": 5000, "price": 4000}},
    )
    weekly = [
        {"name": "Aceite de oliva", "estimated_cost_rd": 4000.0},
        {"name": "Salmón", "estimated_cost_rd": 850.0},
    ]
    days = _mk_days()
    subs = _go_wired._apply_budget_driver_aware_pass(days, {"user_id": "u1"}, weekly)
    assert subs == 1
    joined = " | ".join(days[0]["meals"][0]["ingredients"]).lower()
    assert "filete de pescado blanco" in joined
    assert "salmón" not in joined
    # el aceite pineado sigue intacto — nunca fue tocado.
    assert "aceite de oliva" in joined


def test_pinned_food_with_matching_family_is_never_overridden(_go_wired, monkeypatch):
    """Si el usuario pineó marca para 'salmón' (que SÍ tiene familia), la sustitución
    no debe pisar esa elección aunque salmón sea el driver #1 por costo."""
    monkeypatch.setattr(
        "shopping_calculator.fetch_brand_pref_packages",
        lambda uid: {"salmon": {"grams": 200, "price": 500}},
    )
    weekly = [{"name": "Salmón", "estimated_cost_rd": 850.0}]
    days = _mk_days()
    subs = _go_wired._apply_budget_driver_aware_pass(days, {"user_id": "u1"}, weekly)
    assert subs == 0
    joined = " | ".join(days[0]["meals"][0]["ingredients"]).lower()
    assert "salmón" in joined
    assert "filete de pescado blanco" not in joined


def test_no_pins_behaves_exactly_as_before(_go_wired, monkeypatch):
    """Sin `user_id` (invitado) o sin filas en user_brand_preferences: comportamiento
    idéntico al pre-fix (fail-open)."""
    monkeypatch.setattr("shopping_calculator.fetch_brand_pref_packages", lambda uid: {})
    weekly = [{"name": "Salmón", "estimated_cost_rd": 850.0}]
    days = _mk_days()
    subs = _go_wired._apply_budget_driver_aware_pass(days, {}, weekly)
    assert subs == 1
    joined = " | ".join(days[0]["meals"][0]["ingredients"]).lower()
    assert "filete de pescado blanco" in joined


# ---------------------------------------------------------------------------
# 4. Funcional: cheapen pass (backstop estático, force=True)
# ---------------------------------------------------------------------------

@pytest.fixture()
def _go_cheapen(monkeypatch):
    import graph_orchestrator as go
    prices = dict(_PRICES)
    prices["mani"] = 80.0
    monkeypatch.setattr(go, "_budget_build_master_price_map", lambda: prices)
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_PASS_ENABLED", True)
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_MAX_SUBS", 5)
    return go


def _mk_days_almendras():
    return [
        {
            "day": 1,
            "meals": [
                {
                    "name": "Ensalada con almendras",
                    "ingredients": ["30 g de almendras"],
                    "ingredients_raw": ["30 g de almendras"],
                }
            ],
        }
    ]


def test_cheapen_pass_respects_pin(_go_cheapen, monkeypatch):
    monkeypatch.setattr(
        "shopping_calculator.fetch_brand_pref_packages",
        lambda uid: {"almendras": {"grams": 200, "price": 450}},
    )
    days = _mk_days_almendras()
    subs = _go_cheapen._apply_budget_cheapen_pass(days, {"user_id": "u1"}, force=True)
    assert subs == 0
    assert "almendras" in days[0]["meals"][0]["ingredients"][0].lower()


def test_cheapen_pass_still_substitutes_without_pin(_go_cheapen, monkeypatch):
    monkeypatch.setattr("shopping_calculator.fetch_brand_pref_packages", lambda uid: {})
    days = _mk_days_almendras()
    subs = _go_cheapen._apply_budget_cheapen_pass(days, {}, force=True)
    assert subs == 1
    assert "maní" in days[0]["meals"][0]["ingredients"][0].lower()
