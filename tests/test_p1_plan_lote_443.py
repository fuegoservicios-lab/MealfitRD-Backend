# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-443 · 2026-09-26] Las notas que hablan de otro plato.

Replay de la cola sobre 322 planes: «NO licúes yuca… en un batido… Elimina este batido» en un bowl, unos tacos y unas
brochetas (3); «enjuaga los enlatados (yogurt griego sin azúcar, granos)» (4)."""
from __future__ import annotations

import pathlib

import pasos_cerrador as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]

_NOTA = ("⚠️ Seguridad alimentaria: NO licúes yuca, víveres ni leguminosas CRUDOS en un batido — deben hervirse hasta "
         "ablandar. Elimina este batido o cámbialo por un ingrediente apto para licuar en crudo (fruta/avena/yogur).")


def test_la_nota_del_batido_se_va_si_nada_se_licua():
    m = {"name": "Bowl tropical de yautía enfriada con huevo duro",
         "recipe": ["El Toque de Fuego: hierve la yautía 15-18 minutos.", "Montaje: sirve el bowl.", _NOTA,
                    "⚠️ Sodio (hipertensión/riñón): elige versiones bajas en sodio y enjuaga los enlatados (yogurt griego "
                    "sin azúcar, granos) antes de usarlos."]}
    assert pc.notas_de_otro_plato(m) == 2
    assert m["recipe"] == ["El Toque de Fuego: hierve la yautía 15-18 minutos.", "Montaje: sirve el bowl.",
                           "⚠️ Sodio (hipertensión/riñón): elige versiones bajas en sodio y enjuaga los enlatados (granos) "
                           "antes de usarlos."]
    assert pc.notas_de_otro_plato(m) == 0


def test_en_un_batido_la_nota_se_queda():
    m = {"name": "Batido de yuca", "recipe": ["Montaje: licúa la yuca con la leche.", _NOTA]}
    assert pc.notas_de_otro_plato(m) == 0 and _NOTA in m["recipe"]


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cerrador").notas_de_otro_plato(meal)  # [P1-PLAN-LOTE-443]' in src
