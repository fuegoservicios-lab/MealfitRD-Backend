# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-429 · 2026-09-26] «½ de cebolla» → «½ cebolla».

Batería REAL sobre el 424 (dm2 con insulina): «corta ½ de cebolla en cubitos». Corpus de 322 planes: 512 menciones en los
pasos. «¼ de cebolla» («un cuarto de») es español y se queda."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_media_cebolla_sin_de():
    m = {"recipe": ["Mise en place: pica ½ de cebolla morada y ½ de aguacate; mide ½ de taza de agua y ½ de ají morrón.",
                    "Montaje: sirve."]}
    assert pc.medio_sin_de(m) == 1
    assert m["recipe"][0] == "Mise en place: pica ½ cebolla morada y ½ aguacate; mide ½ taza de agua y ½ ají morrón."
    assert pc.medio_sin_de(m) == 0


def test_lo_que_lleva_de_se_queda():
    pasos = ["Mise en place: corta ¼ de cebolla, 1½ de cebolla y ½ de la cebolla restante.",
             "⚠️ Nota: ½ de cebolla cruda puede repetir."]
    m = {"recipe": list(pasos)}
    assert pc.medio_sin_de(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").medio_sin_de(meal)  # [P1-PLAN-LOTE-429]' in src
