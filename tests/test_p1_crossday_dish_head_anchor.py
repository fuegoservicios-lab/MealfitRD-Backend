"""[P1-CROSSDAY-DISH-HEAD-ANCHOR + P1-CROSSDAY-DISH-DIVERSIFY · 2026-07-06] El gate cross-día
"mismo plato en 3+ días" contaba el token base como SUBSTRING en cualquier parte del nombre →
una GUARNICIÓN ("Yautía al Ajillo con Queso y Ensalada", "Mejillones al Vapor y Ensalada") se
contaba como si el plato FUERA ensalada → falso "ensalada en 3 días" (3 rechazos vivos en la
sesión 2026-07-06, mismo falso-positivo por el que el gate INTRA-día nació OFF).

Fix (2 capas):
1. HEAD-ANCHOR: la detección del plato-base (intra Y cross-día) usa la CABEZA del nombre
   (antes del 1er conector 'con/y/sobre/...'), no cualquier substring → la guarnición ya no
   cuenta.
2. DIVERSIFY: para el caso GENUINO (una cabeza renombrable repetida en ≥N días), renombra la
   forma limpia "Base de X" → sinónimo honesto en los días excedentes, sin tocar
   ingredientes/macros ni adjetivos con género.
"""
import os

import pytest

import graph_orchestrator as go
from graph_orchestrator import build_variety_report


def _meal(name, ings=None):
    return {"name": name, "ingredients": ings or []}


# ─────────── 1. head-anchor: guarnición NO cuenta ───────────

def test_side_salad_not_counted_crossday():
    days = [
        {"day": 1, "meals": [_meal("Yautía al Ajillo con Queso Blanco y Ensalada", ["yautia"])]},
        {"day": 2, "meals": [_meal("Mejillones al Vapor con Finas Hierbas y Ensalada", ["mejillones"])]},
        {"day": 3, "meals": [_meal("Pollo a la Plancha con Ensalada", ["pollo"])]},
    ]
    rep = build_variety_report({"days": days})
    assert "ensalada" not in rep["cross_day_dishes"], (
        f"la ensalada es GUARNICIÓN (tras 'con/y'), no el plato → no cuenta: {rep['cross_day_dishes']}"
    )


def test_genuine_head_salad_still_counted():
    days = [
        {"day": 1, "meals": [_meal("Ensalada César con Pollo", ["pollo"])]},
        {"day": 2, "meals": [_meal("Ensalada Griega con Atún", ["atun"])]},
        {"day": 3, "meals": [_meal("Ensalada de Aguacate", ["aguacate"])]},
    ]
    rep = build_variety_report({"days": days})
    assert rep["cross_day_dishes"].get("ensalada") == 3, (
        f"3 ensaladas de CABEZA = monotonía genuina, sigue contando: {rep['cross_day_dishes']}"
    )


def test_head_dish_base_token_unit():
    assert go._head_dish_base_token("ensalada cesar con pollo") == "ensalada"
    assert go._head_dish_base_token("yautia al ajillo con queso y ensalada") is None
    assert go._head_dish_base_token("pollo a la plancha con ensalada") == "plancha"
    assert go._head_dish_base_token("revoltillo de huevo con platano") == "revoltillo"
    assert go._head_dish_base_token("mejillones al vapor y ensalada") is None


# ─────────── 2. gate existente intra-día intacto ───────────

def test_same_day_revoltillo_still_detected():
    # revoltillos de CABEZA (sin 'con' que los divida) siguen contando same-day.
    days = [{"day": 3, "meals": [
        _meal("Revoltillo de Huevos", ["huevos"]),
        _meal("Pollo Guisado", ["pollo"]),
        _meal("Revoltillo de Vegetales", ["vegetales"])]}]
    rep = build_variety_report({"days": days})
    assert rep["same_day_repeats"] >= 1
    assert any("revoltillo" in i.lower() for i in rep["issues"])


# ─────────── 3. diversificador cross-día ───────────

def test_diversify_renames_surplus_days():
    days = [
        {"day": 1, "meals": [_meal("Ensalada de Pollo", ["150 g de pollo"])]},
        {"day": 2, "meals": [_meal("Ensalada de Atún", ["120 g de atun"])]},
        {"day": 3, "meals": [_meal("Ensalada de Aguacate", ["1 aguacate"])]},
    ]
    n = go._diversify_cross_day_dishes(days)
    assert n == 1, "conserva las primeras 2 (MIN_DAYS-1), renombra el 3er día"
    names = [d["meals"][0]["name"] for d in days]
    assert names[0] == "Ensalada de Pollo" and names[1] == "Ensalada de Atún"
    assert names[2] == "Bowl de Aguacate", f"3er día → sinónimo honesto: {names[2]}"
    # ingredientes intactos
    assert days[2]["meals"][0]["ingredients"] == ["1 aguacate"]
    assert days[2]["meals"][0]["_crossday_dish_diversified"] == "ensalada->bowl"


def test_diversify_resolves_the_gate():
    days = [
        {"day": 1, "meals": [_meal("Guiso de Res", ["res"])]},
        {"day": 2, "meals": [_meal("Guiso de Pollo", ["pollo"])]},
        {"day": 3, "meals": [_meal("Guiso de Cerdo", ["cerdo"])]},
    ]
    assert build_variety_report({"days": days})["cross_day_dishes"].get("guiso") == 3
    go._diversify_cross_day_dishes(days)
    # tras el rename, el gate ya no ve 'guiso' en 3 días
    assert "guiso" not in build_variety_report({"days": days})["cross_day_dishes"]
    assert days[2]["meals"][0]["name"] == "Estofado de Cerdo"


def test_diversify_skips_adjective_form():
    # "Ensalada Griega" (adjetivo, sin ' de ') → NO renombra (riesgo de género).
    days = [
        {"day": 1, "meals": [_meal("Ensalada Griega", ["queso"])]},
        {"day": 2, "meals": [_meal("Ensalada César", ["pollo"])]},
        {"day": 3, "meals": [_meal("Ensalada Rusa", ["papa"])]},
    ]
    assert go._diversify_cross_day_dishes(days) == 0, "sin forma 'Base de X' limpia → gate/retry"
    assert days[2]["meals"][0]["name"] == "Ensalada Rusa"


def test_diversify_skips_non_renameable_base():
    # revoltillo (huevo-identidad) NO está en el mapa de rename → intacto.
    days = [
        {"day": 1, "meals": [_meal("Revoltillo de Huevo", ["2 huevos"])]},
        {"day": 2, "meals": [_meal("Revoltillo de Vegetales", ["vegetales"])]},
        {"day": 3, "meals": [_meal("Revoltillo de Queso", ["queso"])]},
    ]
    assert go._diversify_cross_day_dishes(days) == 0
    assert all("Revoltillo" in d["meals"][0]["name"] for d in days)


def test_diversify_under_threshold_noop():
    days = [
        {"day": 1, "meals": [_meal("Ensalada de Pollo", ["pollo"])]},
        {"day": 2, "meals": [_meal("Ensalada de Atún", ["atun"])]},
    ]
    assert go._diversify_cross_day_dishes(days) == 0, "solo 2 días < MIN_DAYS=3 → no toca"


def test_diversify_knob_off(monkeypatch):
    monkeypatch.setattr(go, "CROSS_DAY_DISH_DIVERSIFY_ENABLED", False)
    days = [
        {"day": 1, "meals": [_meal("Ensalada de Pollo", ["pollo"])]},
        {"day": 2, "meals": [_meal("Ensalada de Atún", ["atun"])]},
        {"day": 3, "meals": [_meal("Ensalada de Aguacate", ["aguacate"])]},
    ]
    assert go._diversify_cross_day_dishes(days) == 0


def test_marker_anchored_in_source():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "graph_orchestrator.py"), encoding="utf-8").read()
    assert "P1-CROSSDAY-DISH-HEAD-ANCHOR" in src
    assert "P1-CROSSDAY-DISH-DIVERSIFY" in src
