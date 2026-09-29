# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-803 · 2026-09-28] En el embarazo, el cerrador de proteína no elige queso blando.

Batería real (embarazo, 28-sep): «Lechosa fresca con maní tostado y queso cottage» dos días seguidos, con el cottage del
cerrador servido frío («disfruta frío… Acompaña con queso cottage pasteurizado») y la nota del lote 193 pidiendo
calentarlo hasta que humee.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import graph_orchestrator as go
import nevera_exigida as ne

_POOL = {"Queso cottage": (11.1, 98), "Yogurt griego": (10.0, 97), "Pechuga de pollo": (31.0, 165),
         "Queso blanco": (18.0, 290), "Queso parmesano": (35.0, 431)}


class _DB:
    def lookup(self, name):
        p = _POOL.get(name)
        return SimpleNamespace(name=name, protein=p[0], kcal=p[1]) if p else None


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(go, "_country_protein_pool", lambda country=None: list(_POOL))


def _nombres(form):
    tok = ne.fijar(form)
    try:
        return [c[1] for c in go._safe_high_density_proteins([], _DB(), min_protein=9.0)]
    finally:
        ne.soltar(tok)


def test_embarazo_sin_queso_blando(pool):
    n = _nombres({"medicalConditions": ["Embarazo"]})
    assert "Queso cottage" not in n and "Queso blanco" not in n, n
    assert "Yogurt griego" in n and "Queso parmesano" in n, "el yogur pasteurizado y el queso curado siguen"


def test_lactancia_y_sin_condicion_intactos(pool):
    for form in ({"medicalConditions": ["Lactancia"]}, {"medicalConditions": []}, None):
        n = _nombres(form)
        assert "Queso cottage" in n and "Queso blanco" in n, (form, n)


def test_si_la_nevera_solo_tiene_queso_blando_gana_la_nevera(pool, monkeypatch):
    monkeypatch.setattr(ne, "lista", lambda fd=None: ["queso cottage"])
    monkeypatch.setattr(ne, "admite", lambda linea, fd=None: "cottage" in str(linea).lower())
    n = _nombres({"medicalConditions": ["Embarazo"]})
    assert n == ["Queso cottage"], n
