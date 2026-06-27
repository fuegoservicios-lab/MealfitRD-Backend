"""[P1-SWAP-SAME-DAY-VARIETY · 2026-06-27] El swap individual (JIT) era ciego al resto del día → cambiaba un
plato por otro que repetía la proteína ya usada ese día (vivo: 'Batata con Mozzarella' → 'Yuca con Soya'
cuando el almuerzo YA era Soya). Fix: el router pasa las otras comidas del día y el swap evita repetir la
proteína principal.
"""
from __future__ import annotations

import re
from pathlib import Path


def test_router_helper_finds_same_day_other_meals(monkeypatch):
    import db_core
    from routers import plans as plans_mod

    fake_plan = {"days": [
        {"meals": [
            {"name": "Revoltillo de Huevos"},
            {"name": "Soya Texturizada sobre Bulgur"},
            {"name": "Batido de Níspero"},
            {"name": "Batata Rellena con Queso Mozzarella"},
        ]},
        {"meals": [{"name": "Avena"}, {"name": "Ragú de Chivo"}]},  # otro día
    ]}
    monkeypatch.setattr(db_core, "execute_sql_query", lambda *a, **k: {"plan_data": fake_plan})

    out = plans_mod._same_day_other_meals_for_swap("u1", "Batata Rellena con Queso Mozzarella")
    assert "Soya Texturizada sobre Bulgur" in out
    assert "Revoltillo de Huevos" in out
    assert "Batata Rellena con Queso Mozzarella" not in out  # el propio plato excluido
    assert "Ragú de Chivo" not in out  # otro día, no se incluye


def test_router_helper_guest_and_empty_are_safe(monkeypatch):
    from routers import plans as plans_mod
    assert plans_mod._same_day_other_meals_for_swap("guest", "X") == []
    assert plans_mod._same_day_other_meals_for_swap("u1", "") == []
    assert plans_mod._same_day_other_meals_for_swap(None, "X") == []


def test_agent_injects_same_day_variety_directive():
    """agent.swap_meal arma context_extras con la directiva de variedad cuando hay same_day_other_meals."""
    src = (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")
    assert "P1-SWAP-SAME-DAY-VARIETY" in src
    assert 'form_data.get("same_day_other_meals")' in src
    assert "VARIEDAD DEL DÍA" in src


def test_router_wires_same_day_into_swap_data():
    src = (Path(__file__).resolve().parent.parent / "routers" / "plans.py").read_text(encoding="utf-8")
    assert "_same_day_other_meals_for_swap" in src
    assert 'data["same_day_other_meals"]' in src
    # se inyecta ANTES de llamar swap_meal
    i_inject = src.find('data["same_day_other_meals"]')
    i_call = src.find("result = swap_meal(data)")
    assert 0 < i_inject < i_call, "la inyección debe ocurrir antes de swap_meal(data)"
