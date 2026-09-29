# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-921 · 2026-09-29] «sella el filete de filete de pescado blanco»: la unidad repetida tras una sustitución
se colapsa en los pasos al final de la cola. Batería real del 29-sep (el formulario del dueño, cena D1)."""
from __future__ import annotations

import pathlib
import sys

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import unidad_repetida as ur  # noqa: E402


def test_la_unidad_repetida_se_colapsa_en_los_pasos():
    m = {"name": "Arepitas con filete de pescado blanco",
         "ingredients": ["¾ filete de pescado blanco (≈211 g)"],
         "recipe": ["Mise en place: mide 75 g de harina y 211 g de filete de filete de pescado blanco.",
                    "El Toque de Fuego: sella el filete de filete de pescado blanco 4-5 minutos por lado hasta 63 °C.",
                    "Montaje: sirve 1 pechuga de pechuga de pollo en tiras y los filetes de filete de pescado.",
                    "⚠️ Seguridad alimentaria: el filete de filete de pescado a 63 °C."]}
    assert ur.limpiar(m) == 3
    assert m["recipe"][0] == "Mise en place: mide 75 g de harina y 211 g de filete de pescado blanco."
    assert m["recipe"][1].startswith("El Toque de Fuego: sella el filete de pescado blanco 4-5 minutos")
    assert m["recipe"][2] == "Montaje: sirve 1 pechuga de pollo en tiras y los filetes de pescado."
    assert m["recipe"][3].startswith("⚠️"), "las notas no se tocan"
    assert m["ingredients"] == ["¾ filete de pescado blanco (≈211 g)"] and m["name"].startswith("Arepitas")
    assert ur.limpiar(m) == 0, "idempotente"


def test_lo_que_no_se_repite_no_se_toca(monkeypatch):
    for t in ("Montaje: sirve el filete de pescado con la pechuga de pollo.", "Montaje: 1 taza de arroz y 1 taza de agua."):
        assert ur.colapsar(t) == t
    monkeypatch.setenv("MEALFIT_UNIT_REPEAT_COLLAPSE", "false")
    m = {"recipe": ["Montaje: sella el filete de filete de pescado."]}
    assert ur.limpiar(m) == 0


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("unidad_repetida").limpiar(meal)  # [P1-PLAN-LOTE-921]' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-921" in (_BACKEND / "unidad_repetida.py").read_text(encoding="utf-8")
