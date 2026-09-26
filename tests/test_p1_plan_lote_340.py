# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-340 · 2026-09-25] La proteína huérfana sale con su frase, no se vuelve «proteína».

Batería real sobre el 331 (perfil del dueño, día 3): «Cocina jamón de proteína a la plancha o hervido y sírvelo como
proteína del plato» y «Acompaña con jamón de proteína» — el autocorrector de coherencia cambiaba «pavo» por la palabra
«proteína» porque el plato no traía otra proteína."""
from __future__ import annotations

import pathlib

import graph_orchestrator as go

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _plan(pasos, ings):
    return {"days": [{"day": 1, "meals": [{"meal": "Cena", "name": "Arepitas de auyama con queso",
                                           "ingredients": list(ings), "recipe": list(pasos),
                                           "desc": "Arepitas con queso y jamón de pavo."}]}]}


def test_sin_otra_proteina_la_frase_sale():
    r = _plan(["Mise en place: corta la auyama.",
               "El Toque de Fuego: cocina las arepitas 3 min por lado. Cocina jamón de pavo a la plancha o hervido y "
               "sírvelo como proteína del plato.",
               "Montaje: sirve las arepitas con el queso; acompaña con agua. Acompaña con jamón de pavo."],
              ["60 g de harina de maíz precocida", "155 g de auyama", "20 g de queso blanco fresco"])
    go._run_assembly_validations(r, {}, set())
    m = r["days"][0]["meals"][0]
    texto = " ".join(m["recipe"]) + " " + m["desc"]
    assert "de proteína" not in texto and "pavo" not in " ".join(m["recipe"]), m["recipe"]
    assert m["recipe"][1] == "El Toque de Fuego: cocina las arepitas 3 min por lado."
    assert m["recipe"][2] == "Montaje: sirve las arepitas con el queso; acompaña con agua."


def test_con_proteina_real_se_sigue_sustituyendo():
    r = _plan(["El Toque de Fuego: dora el pollo 5 min. Sirve con jamón de pavo.", "Montaje: sirve."],
              ["150 g de pechuga de pollo", "60 g de harina de maíz precocida"])
    go._run_assembly_validations(r, {}, set())
    assert "pavo" not in " ".join(r["days"][0]["meals"][0]["recipe"])


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").sin_frases_de(_s, _p, _repl)  # [P1-PLAN-LOTE-340]' in src
