# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-343 · 2026-09-25] Lo cocido del paso es el equivalente de lo SECO de la lista.

Batería real sobre el 331 (vegana en México): la lista cuenta «155 g de frijoles negros secos» (≈530 kcal) y el paso dice
«Mide 155 g de frijoles negros cocidos» (≈200 kcal). 103 de 137 comidas con legumbre seca en gramos del corpus: el
reparador de cantidades del contrato (C2) copiaba los gramos secos a la mención cocida."""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


class _DB:
    _FILAS = {"frijol": SimpleNamespace(name="Habichuelas negras", kcal=341.0),
              "garbanzo": SimpleNamespace(name="Garbanzos", kcal=378.0),
              "tomate": SimpleNamespace(name="Tomate", kcal=18.0)}

    def lookup(self, nombre):
        n = str(nombre).lower()
        return next((v for k, v in self._FILAS.items() if k in n), None)


def test_el_paso_cocido_recibe_el_equivalente_de_lo_seco():
    m = {"ingredients": ["2 tortillas de maíz", "155 g de frijoles negros secos", "115 g de garbanzos secos"],
         "recipe": ["Mise en place: pica el chile. Mide 155 g de frijoles negros cocidos y 115 g de garbanzos secos (cocidos).",
                    "El Toque de Fuego: machaca los 155 g de frijoles negros cocidos 5 minutos.", "Montaje: rellena."]}
    assert pc.cocido_de_la_lista(m, _DB()) == 3
    assert m["recipe"][0] == ("Mise en place: pica el chile. Mide 415 g de frijoles negros cocidos y 340 g de garbanzos "
                              "cocidos.")                       # 155 × 341/127 ≈ 416 → 415; 115 × 378/127 ≈ 342 → 340
    assert "machaca los 415 g de frijoles negros cocidos" in m["recipe"][1]


def test_ya_cuadra_reparto_o_sin_catalogo_no_se_tocan():
    pasos = ["Mise en place: mide 410 g de frijoles negros cocidos; reserva la mitad, 55 g de garbanzos cocidos, para mañana.",
             "Montaje: sirve."]
    m = {"ingredients": ["155 g de frijoles negros secos", "115 g de garbanzos secos", "55 g de tomate"],
         "recipe": list(pasos)}
    assert pc.cocido_de_la_lista(m, _DB()) == 0 and m["recipe"] == pasos
    m2 = {"ingredients": ["155 g de frijoles negros secos"], "recipe": ["Mide 155 g de frijoles negros cocidos."]}
    assert pc.cocido_de_la_lista(m2, None) == 0


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").cocido_de_la_lista(meal, db)' in src
