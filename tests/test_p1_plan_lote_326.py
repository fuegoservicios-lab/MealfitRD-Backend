# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-326 · 2026-09-25] El adjetivo concuerda con UNO o una fracción sola en los pasos.

Baterías guardadas: 28 pasos como «Mise en place: pela y corta ½ plátano verdes en trozos», «lava ½ hoja grandes de
lechuga», «bate 1 huevo enteros con 6 claras» — el reescritor del conteo cambia «2 plátanos» por «½ plátano» y deja el
adjetivo en plural. La lista ya decía «½ plátano verde»."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _meal(pasos):
    return {"ingredients": ["½ plátano verde", "½ hoja grande de lechuga", "1 huevo", "6 claras de huevo"],
            "recipe": list(pasos)}


def test_uno_o_medio_con_adjetivo_en_singular():
    m = _meal(["Mise en place: pela y corta ½ plátano verdes en trozos; lava ½ hoja grandes de lechuga.",
               "El Toque de Fuego: bate 1 huevo enteros con 6 claras y cuaja 3 min; pela 1 guineo medianos.",
               "Montaje: sirve."])
    assert pc.decimales_de_cocina(m) == 2
    assert m["recipe"][0] == "Mise en place: pela y corta ½ plátano verde en trozos; lava ½ hoja grande de lechuga."
    assert m["recipe"][1] == "El Toque de Fuego: bate 1 huevo entero con 6 claras y cuaja 3 min; pela 1 guineo mediano."


def test_lo_que_ya_es_plural_no_se_toca():
    pasos = ["Mise en place: pela 2 plátanos verdes y 1½ plátanos maduros; lava 3 hojas grandes.",
             "El Toque de Fuego: cocina 1 taza de frijoles rojos y 21 huevos enteros.",
             "⚠️ Seguridad alimentaria: cocina 1 huevo enteros por completo."]
    m = _meal(pasos)
    assert pc.decimales_de_cocina(m) == 0
    assert m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
    assert 'nuevo = _ADJ_TRAS_UNO_RE.sub(r"\\1\\2", nuevo)  # [P1-PLAN-LOTE-326]' in src
