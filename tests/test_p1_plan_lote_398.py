# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-398 · 2026-09-26] Tras el verbo, el alimento va en minúscula.

Batería REAL sobre el 379 (adulto mayor con HTA): «Añade Filete de pescado blanco al guiso», «Cocina Filete de pescado
blanco a la plancha», «Incorpora también Queso blanco», «añade Sal al gusto» (corpus: 117)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_alimento_tras_el_verbo_va_en_minuscula():
    m = {"recipe": ["El Toque de Fuego: sofríe la cebolla 3 min. Añade Filete de pescado blanco al guiso y cocínalo "
                    "12-15 minutos. Incorpora también Queso blanco durante la preparación.",
                    "Montaje: sirve el revoltillo; añade Sal al gusto."]}
    assert pc.minuscula_tras_verbo(m) == 2
    assert m["recipe"] == ["El Toque de Fuego: sofríe la cebolla 3 min. Añade filete de pescado blanco al guiso y cocínalo "
                           "12-15 minutos. Incorpora también queso blanco durante la preparación.",
                           "Montaje: sirve el revoltillo; añade sal al gusto."], m["recipe"]


def test_lo_propio_se_queda():
    pasos = ["Montaje: sirve Corn Flakes con la leche.", "El Toque de Fuego: añade Maggi al caldo.",
             "⚠️ Seguridad alimentaria: cocina Pechuga de pollo por completo."]
    m = {"recipe": list(pasos)}
    assert pc.minuscula_tras_verbo(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").minuscula_tras_verbo(meal)  # [P1-PLAN-LOTE-398]' in src
