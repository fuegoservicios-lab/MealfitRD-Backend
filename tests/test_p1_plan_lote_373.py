# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-373 · 2026-09-26] Los nombres del catálogo van en minúscula a media frase; el participio, una vez.

Replay de la cola real: «añade el tomate y la pechuga, Sal al gusto y Pimienta negra», «¾ cdta de Aceite de oliva, Limón y
Orégano dominicano», «15 g de maní fileteado fileteado»."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_minuscula_y_participio_una_vez():
    m = {"recipe": [
        "El Toque de Fuego: añade el tomate y la pechuga, Sal al gusto y Pimienta negra al gusto.",
        "Mise en place: mide ¾ cdta de Aceite de oliva, Limón y Orégano dominicano; pica 15 g de maní fileteado fileteado.",
        "Montaje: sirve con 1 taza de Corn Flakes y Leche.",
        "⚠️ Nota: usa Sal Marina."]}
    assert pc.pasos_en_minuscula(m) == 3
    assert m["recipe"][0] == "El Toque de Fuego: añade el tomate y la pechuga, sal al gusto y pimienta negra al gusto."
    assert m["recipe"][1] == "Mise en place: mide ¾ cdta de aceite de oliva, limón y orégano dominicano; pica 15 g de maní fileteado."
    assert m["recipe"][2] == "Montaje: sirve con 1 taza de Corn Flakes y leche."              # la marca se queda
    assert m["recipe"][3] == "⚠️ Nota: usa Sal Marina."                                           # las notas no se tocan


def test_lo_que_no_se_toca():
    pasos = ["Mise en place: Lava el arroz. El Toque de Fuego: cocina. Montaje: sirve.",
             "Montaje: sirve las coles de Bruselas con mostaza de Dijon y canela de Ceilán."]
    m = {"recipe": list(pasos)}
    assert pc.pasos_en_minuscula(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").pasos_en_minuscula(meal)  # [P1-PLAN-LOTE-373]' in src
