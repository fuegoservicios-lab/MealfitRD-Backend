"""[P2-HABAS-CHEAPEN · 2026-07-06] Review #14: las Habas (RD$205/lb) eran el driver #1 del exceso
de presupuesto del plan del owner (RD$820, 4 paquetes) — 3-4× más caras que otras legumbres
(habichuelas rojas ~RD$53/lb). No estaban en la tabla premium→económico. Añadidas: habas →
Habichuelas rojas (legumbre coherente es-DO, 74% más barata), disparando SOLO cuando el plan
excede su referencia de presupuesto (`force=True` en la convergencia post-costeo).
"""
import pytest

import graph_orchestrator as go


@pytest.fixture()
def _prices(monkeypatch):
    # price-map mockeado (evita DB): habas caras, habichuelas baratas.
    monkeypatch.setattr(go, "_budget_build_master_price_map",
                        lambda: {"habichuelas rojas": 53.0, "habas": 205.0})


def _meal(name, ings, recipe):
    return {"name": name, "ingredients": list(ings), "ingredients_raw": list(ings),
            "recipe": list(recipe)}


def test_habas_cheapened_to_habichuelas(_prices):
    days = [{"day": 1, "meals": [_meal(
        "Yuca con Habas Guisadas", ["125 g de habas", "50 g de yuca"],
        ["El Toque de Fuego: guisa las habas con sofrito."])]}]
    n = go._apply_budget_cheapen_pass(days, {}, force=True)
    assert n == 1, "habas premium → habichuelas rojas en plan excedido"
    m = days[0]["meals"][0]
    ings = " ".join(m["ingredients"]).lower()
    assert "habichuelas rojas" in ings and "de habas" not in ings, f"swap en ingredientes: {ings}"
    assert "Habichuelas rojas" in m["name"], f"nombre honesto renombrado: {m['name']}"
    assert "habas" not in " ".join(m["recipe"]).lower(), "pasos reescritos"
    assert any("habas → Habichuelas rojas" in s for s in m.get("_budget_substitutions", []))


def test_habichuelas_not_swapped(_prices):
    # \b + exclude: no tocar habichuelas (ya son la legumbre barata).
    days = [{"day": 1, "meals": [_meal(
        "Habichuelas Rojas Guisadas", ["1 taza de habichuelas rojas"],
        ["Guisa las habichuelas."])]}]
    assert go._apply_budget_cheapen_pass(days, {}, force=True) == 0


def test_not_over_budget_leaves_habas(_prices):
    # sin force y sin preferencia de economía → no toca (habas es legítima si hay presupuesto).
    days = [{"day": 1, "meals": [_meal(
        "Yuca con Habas Guisadas", ["125 g de habas"], ["Guisa."])]}]
    assert go._apply_budget_cheapen_pass(days, {}, force=False) == 0, (
        "el cheapen solo toca habas en planes EXCEDIDOS, no en presupuesto"
    )


def test_marker_anchored():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P2-HABAS-CHEAPEN" in src
