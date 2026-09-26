# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-372 · 2026-09-26] El paso ya no nombra lo que la lista perdió.

Plan real del dueño (26-sep, 04:34 UTC, bloque 2 de su plan de 30 días): la merienda «Vasito rápido de guineo con yogur,
avena y queso cottage» decía «mide … 30 g de avena y 10 g de maní fileteado» y «termina con maní fileteado» sin maní en la
lista. V5 lo acusaba, pero la retirada no reconocía «fileteado», caía a la frase entera (se llevaba el yogurt y la avena:
dos V3) y se deshacía."""
from __future__ import annotations

import pathlib

import recipe_repair as rr
from culinary_coherence import build_culinary_index

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_IDX = build_culinary_index([
    {"name": "Maní", "aliases": ["mani", "cacahuate"], "category": "Frutos secos"},
    {"name": "Yogurt natural", "aliases": ["yogurt", "yogur"], "category": "Lácteos"},
    {"name": "Avena", "aliases": [], "category": "Granos"},
    {"name": "Guineo", "aliases": ["guineos"], "category": "Frutas"},
    {"name": "Queso cottage", "aliases": ["cottage"], "category": "Lácteos"},
])


def _merienda():
    return {"name": "Vasito rápido de guineo con yogur, avena y queso cottage", "meal": "Merienda",
            "ingredients": ["½ taza de yogurt natural sin azúcar", "100 g de guineo", "30 g de avena", "65 g de queso cottage"],
            "recipe": ["Mise en place: corta 100 g de guineo en cubos; mide ½ taza de yogurt natural sin azúcar (130 g), 30 g "
                       "de avena y 10 g de maní fileteado.",
                       "Montaje: sirve el yogurt en un vaso o tazón, añade guineo y la avena, y termina con maní fileteado. "
                       "Termina con queso cottage."]}


def test_el_fantasma_se_va_sin_llevarse_lo_demas():
    m = _merienda()
    out = rr.retirar_sin_lista(m, _IDX)
    assert out["aplicado"] and not out["descartado"], out
    assert m["recipe"][0] == ("Mise en place: corta 100 g de guineo en cubos; mide ½ taza de yogurt natural sin azúcar "
                              "(130 g) y 30 g de avena."), m["recipe"][0]
    assert m["recipe"][1] == ("Montaje: sirve el yogurt en un vaso o tazón, añade guineo y la avena. Termina con queso "
                              "cottage."), m["recipe"][1]


def test_quitar_mencion_con_dos_modificadores():
    assert rr.quitar_mencion("Montaje: sirve la avena con 10 g de maní tostado picado.", "mani") == "Montaje: sirve la avena."


def test_leche_pasteurizada_y_su_sofrito():
    # corpus: «¾ ml de leche pasteurizada» (embarazo) y «guisado con su sofrito» (DM2)
    assert rr.quitar_mencion("Mise en place: mide 40 g de pan integral, 2 cdtas de mantequilla de maní (10 g) y ¾ ml de leche "
                             "pasteurizada; coloca 2 huevos en una olla.", "leche") == (
        "Mise en place: mide 40 g de pan integral y 2 cdtas de mantequilla de maní (10 g); coloca 2 huevos en una olla.")
    assert rr.quitar_mencion("Montaje: unta la mantequilla de maní sobre las tostadas, sirve con el huevo cocido y acompaña con "
                             "la leche pasteurizada.", "leche") == (
        "Montaje: unta la mantequilla de maní sobre las tostadas, sirve con el huevo cocido.")
    assert rr.quitar_mencion("Montaje: coloca encima el pescado guisado con su sofrito y termina con cilantro picado.",
                             "sofrito") == "Montaje: coloca encima el pescado guisado y termina con cilantro picado."


def test_ancla():
    assert "P1-PLAN-LOTE-372" in (_BACKEND / "recipe_repair.py").read_text(encoding="utf-8")
