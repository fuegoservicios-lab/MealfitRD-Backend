# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-585 · 2026-09-27] La harina de maíz es un producto, no la «harina» del maíz.

Batería real (celíaco): el paso medía «15 g de harina de maíz precocida» y el reparador de fantasmas compraba «maíz dulce
en granos».
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

_IDX = {
    "harina de maiz precocida": "Harina de maíz precocida",
    "maiz": "Maíz dulce en granos", "maiz dulce en granos": "Maíz dulce en granos",
    "harina de trigo": "Harina de trigo",
    "guanabana": "Guanábana", "platano": "Plátano verde", "negrito": "Harina de negrito",
}


@pytest.fixture(autouse=True)
def _indice(monkeypatch):
    monkeypatch.setattr(go, "_phantom_catalog_index", lambda: dict(_IDX))


@pytest.mark.parametrize("frase, canon", [
    ("harina de maíz precocida", "Harina de maíz precocida"),
    ("harina de maíz", "Harina de maíz precocida"),
    ("harina de trigo", "Harina de trigo"),
    ("pulpa de guanábana", "Guanábana"),
    ("harina de negrito", "Harina de negrito"),
])
def test_resuelve_el_producto(frase, canon):
    r = go._phantom_resolve_food(frase)
    assert r and r[1] == canon, (frase, r)


def test_una_harina_sin_fila_no_compra_su_base():
    assert go._phantom_resolve_food("harina de plátano") is None


def test_el_paso_de_la_bateria_compra_la_harina():
    dias = [{"day": 1, "meals": [{
        "name": "Wrap criollo de maíz con ñame, lentejas y queso blanco",
        "ingredients": ["¼ pedazo de ñame (≈81 g)", "50 g de lentejas secas", "35 g de queso blanco"],
        "ingredients_raw": ["75 g de ñame", "50 g de lentejas secas", "35 g de queso blanco"],
        "recipe": ["Mise en place: mide 15 g de harina de maíz precocida, ¼ pedazo de ñame y 35 g de queso blanco.",
                   "El Toque de Fuego: forma una tortilla grande con la harina de maíz y agua; cocínala 3-4 min por lado."],
    }]}]
    hechos = go._repair_declared_but_unlisted_ingredients(dias)
    ings = dias[0]["meals"][0]["ingredients"]
    assert any("Harina de maíz precocida" in x for x in ings), (hechos, ings)
    assert not any("dulce en granos" in x for x in ings + dias[0]["meals"][0]["ingredients_raw"])


def test_ancla():
    src = (Path(go.__file__)).read_text(encoding="utf-8")
    assert '__import__("harina_es_producto").resolver(words, idx)  # [P1-PLAN-LOTE-585]' in src
