# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-610 · 2026-09-27] La proteína nueva del 592 es magra de verdad.

Validación en producción (estudiante, día 2): con el pollo ya en el almuerzo, el 592 eligió «pavo molido» y el recorte de
grasa del escudo lo bajó de 45 a 25 g (por kcal es más grasa que proteína); el día quedó a 3,5 g del piso.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import proteina_nueva as pn  # noqa: E402


class _Info:
    def __init__(self, name, protein, fats):
        self.name, self.protein, self.fats, self.kcal, self.carbs = name, protein, fats, 4 * protein + 9 * fats, 0.0


_DIA = {"meals": [{"meal": "Almuerzo", "name": "Pollo guisado", "ingredients": ["150 g de pechuga de pollo"]},
                  {"meal": "Cena", "name": "Plátano relleno de queso", "ingredients": ["20 g de queso blanco"]}]}
_FORM = {"dietType": "balanced", "cookingTime": "30min", "groceryDuration": "weekly", "country": "DO"}


def _con(monkeypatch, cands):
    monkeypatch.setattr(go, "_safe_high_density_proteins", lambda *a, **k: list(cands))


def test_el_pavo_molido_no_es_magro(monkeypatch):
    _con(monkeypatch, [(0.10, "Pavo molido", _Info("Pavo molido", 17.0, 10.0)),
                       (0.22, "Pechuga de pavo", _Info("Pechuga de pavo", 24.0, 1.0))])
    elegidas = [c[1] for c in pn._candidatas(_FORM, None, go, _DIA)]
    assert elegidas == ["Pechuga de pavo"], elegidas


def test_sin_ninguna_magra_no_hay_candidata(monkeypatch):
    _con(monkeypatch, [(0.10, "Pavo molido", _Info("Pavo molido", 17.0, 10.0))])
    assert pn._candidatas(_FORM, None, go, _DIA) == []


def test_ni_marisco_ni_legumbre(monkeypatch):
    # replay real (mes sin pescado, día 1): con pollo y res ya en el día, la primera candidata era «Cangrejo»
    _con(monkeypatch, [(0.2, "Cangrejo", _Info("Cangrejo", 18.06, 1.08)), (0.2, "Habas", _Info("Habas", 26.1, 1.53)),
                       (0.19, "Chivo", _Info("Chivo", 20.6, 2.31))])
    assert [c[1] for c in pn._candidatas(_FORM, None, go, _DIA)] == ["Chivo"]


def test_entre_iguales_la_mas_magra(monkeypatch):
    _con(monkeypatch, [(0.15, "Res", _Info("Res", 21.0, 7.0)), (0.19, "Res magra", _Info("Res magra", 22.0, 4.0))])
    assert [c[1] for c in pn._candidatas(_FORM, None, go, _DIA)][0] == "Res magra"


@pytest.mark.parametrize("p, f, magra", [(24.0, 1.0, True), (17.0, 10.0, False), (18.7, 8.3, False), (21.1, 9.47, False),
                                          (21.5, 5.7, True), (20.0, 5.55, True)])
def test_magra_es_proteina_sobre_grasa_en_kcal(p, f, magra):
    assert pn._magra((0.1, "x", _Info("x", p, f))) is magra
