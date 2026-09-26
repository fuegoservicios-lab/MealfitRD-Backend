# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-374 · 2026-09-26] La pista «(N g)» de una taza dice lo que mide el motor.

Replay de la cola real: «1¼ tazas de avena en hojuelas (65 g)» con 107 g en el motor (perfil del dueño), «⅓ taza de
yogurt griego natural sin azúcar (120 g)» con 79 g (bariátrica) — 78 de 313 pistas de taza."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_pista_sigue_al_motor():
    m = {"ingredients": ["1¼ tazas de avena en hojuelas", "⅓ taza de yogurt griego natural sin azúcar", "1 guineo"],
         "ingredients_raw": ["107 g de avena en hojuelas", "79.4 g de yogurt griego natural sin azúcar", "1 guineo"],
         "recipe": ["Mise en place: mide 1¼ tazas de avena en hojuelas (65 g) y ⅓ taza de yogurt griego natural sin azúcar "
                    "(120 g).", "⚠️ Nota: ⅓ taza de yogurt (120 g)."]}
    assert pc.pista_de_taza_del_motor(m) == 1
    assert m["recipe"][0] == ("Mise en place: mide 1¼ tazas de avena en hojuelas (107 g) y ⅓ taza de yogurt griego natural "
                              "sin azúcar (79 g)."), m["recipe"][0]
    assert m["recipe"][1] == "⚠️ Nota: ⅓ taza de yogurt (120 g)."                                   # las notas no se tocan


def test_lo_que_no_se_toca():
    casos = [
        (["40 g de arroz blanco crudo"], ["Montaje: sirve ½ taza de arroz cocido (100 g)."]),               # cocido vs crudo
        (["80 g de yogurt natural", "40 g de yogurt griego"], ["Mise en place: mide ⅓ taza de yogurt (120 g)."]),  # ambiguo
        (["⅓ taza de yogurt natural"], ["Mise en place: mide ⅓ taza de yogurt natural (120 g)."]),          # motor sin gramos
        (["83 g de yogurt natural"], ["Mise en place: mide ⅓ taza de yogurt natural (85 g)."]),              # ya casi igual
    ]
    for raw, pasos in casos:
        m = {"ingredients_raw": list(raw), "recipe": list(pasos)}
        assert pc.pista_de_taza_del_motor(m) == 0 and m["recipe"] == pasos, (raw, m["recipe"])


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").pista_de_taza_del_motor(meal)  # [P1-PLAN-LOTE-374]' in src
