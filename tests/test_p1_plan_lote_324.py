# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-324 · 2026-09-25] «hasta que ablande» también dice hasta cuándo.

Batería real con el lote 323 (perfil del dueño, «Nada»): «Añade la auyama con 2 cdas de agua, tapa y cocina 3 min hasta
que ablande» — V7f exige 8 min a un víver salvo que el paso diga hasta cuándo, no reconocía «ablande» y el reparador del
lote 68 añadía «🍠 Corta Auyama en cubos pequeños (1 cm) y hiérvelos 8-10 minutos» a un plato que ya la cocía."""
from __future__ import annotations

import pathlib

import culinary_coherence as cc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_IDX = cc.build_culinary_index([
    {"name": "Auyama", "aliases": ["calabaza", "auyamas"], "category": "Víveres", "prep_methods": ["hervido", "horneado"]},
    {"name": "Repollo", "aliases": ["repollos"], "category": "Vegetales", "prep_methods": ["crudo", "salteado"]},
])


def _wok(fuego):
    return {"name": "Wok de pollo con auyama", "ingredients": ["250 g de auyama en cubos pequeños", "75 g de repollo"],
            "recipe": ["Mise en place: corta 250 g de auyama en cubos pequeños y 75 g de repollo en tiras.", fuego,
                       "Montaje: reparte el wok sobre las tortillas."]}


def test_ablande_se_deshaga_y_blandita_dicen_hasta_cuando():
    for frase in ("Añade la auyama con 2 cdas de agua, tapa y cocina 3 min hasta que ablande; agrega el repollo.",
                  "Cocina la auyama tapada 4 min, hasta que se deshaga un poco.",
                  "Cocina la auyama 4 minutos hasta que esté blandita."):
        assert cc.alimentos_sin_coccion(_wok(frase), _IDX) == [], frase


def test_sin_decir_hasta_cuando_sigue_sin_cocer():
    assert cc.alimentos_sin_coccion(_wok("Saltea la auyama 3 minutos con el repollo."), _IDX) == [("Auyama", "viver")]


def test_ancla():
    src = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    assert "cocid|ablan|blandit|deshag)" in src and "[P1-PLAN-LOTE-324" in src
