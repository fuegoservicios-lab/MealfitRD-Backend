"""[P1-PLAN-LOTE-890 · 2026-09-29] La descripción veraz, encendida en RD (el mercado principal) con los dos casos reales
que la pidieron.

El lote 854 limpió las fichas de los países beta y dejó RD apagado por cautela; el 858 midió RD sin IA (1 014 cambios
únicos en 780 planes y 24 de 188 fichas de los planes vivos, todos leídos) y lo encendió por defecto. La sesión de
PLATO midió el defecto en RD: ~3 % de las comidas dicen «sin lácteos» y llevan yogur, cottage o queso en la lista
(casi todas se los añadió después el cerrador de proteína), y el reparador 426 dejaba «servido con huevo duro» en una
cena que ya lleva pechuga. Este test fija esos dos casos en un plan DOMINICANO con los knobs por defecto, y que la
palanca de RD los vuelve a dejar como antes.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import descripcion_veraz as dv  # noqa: E402


@pytest.fixture(autouse=True)
def _por_defecto(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    for k in ("MEALFIT_DESCRIPTION_TRUTH", "MEALFIT_DESCRIPTION_TRUTH_DO"):
        monkeypatch.delenv(k, raising=False)


def _plan_do(desc, ingredientes, nombre, comida="Cena"):
    return {"_country": "DO", "days": [{"day": 1, "meals": [
        {"meal": comida, "name": nombre, "desc": desc, "ingredients": list(ingredientes)}]}]}


_HUEVO = ("Pechuga jugosa a la plancha, servido con huevo duro y ensalada fresca.",
          ["Pechuga de pollo", "Lechuga", "Tomate"], "Pechuga a la plancha con ensalada")


def test_rd_quita_el_huevo_duro_que_el_plato_ya_no_lleva():
    p = _plan_do(*_HUEVO)
    assert dv.aplicar_plan(p) == 1
    desc = p["days"][0]["meals"][0]["desc"]
    assert "huevo" not in desc.lower(), desc
    assert desc == "Pechuga jugosa a la plancha con ensalada fresca.", desc


def test_rd_no_deja_una_ausencia_falsa_de_lacteos():
    """La merienda real de la batería de esta mañana: la ficha dice «sin lácteos» y la lista lleva yogur griego."""
    p = _plan_do("Snack ligero y sin lácteos: casabe tostado con mantequilla de maní y fresas.",
                 ["2 casabes", "1 cda de mantequilla de maní", "½ taza de fresas", "1 taza de yogurt griego entero"],
                 "Casabe con mantequilla de maní", comida="Merienda")
    dv.aplicar_plan(p)
    desc = p["days"][0]["meals"][0]["desc"]
    assert "sin lácteos" not in desc.lower(), desc


def test_una_ausencia_verdadera_se_queda():
    p = _plan_do("Cena sin lácteos: pechuga jugosa a la plancha con ensalada fresca.",
                 ["Pechuga de pollo", "Lechuga", "Tomate"], "Pechuga a la plancha con ensalada")
    antes = copy.deepcopy(p)
    dv.aplicar_plan(p)
    assert p == antes


def test_la_palanca_de_rd_lo_deja_como_antes(monkeypatch):
    monkeypatch.setenv("MEALFIT_DESCRIPTION_TRUTH_DO", "false")
    p = _plan_do(*_HUEVO)
    antes = copy.deepcopy(p)
    assert dv.aplicar_plan(p) == 0
    assert p == antes


def test_marker():
    assert "[P1-PLAN-LOTE-890 · 2026-09-29]" in Path(__file__).read_text(encoding="utf-8")
