# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-393 · 2026-09-26] «Escurre 355 g de garbanzos cocidos» con los secos en la lista trae su cocción previa.

Batería REAL sobre el 379 (estatina, día 1): «1 taza de garbanzos secos» en la lista y «escurre 355 g de garbanzos
cocidos… incorpora los garbanzos… y cocina 10 min» en los pasos, sin remojo ni hervor (V7c los daba por cocidos)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_NOTA = ("💡 Cocción previa: remoja los garbanzos secos 8-12 h y hiérvelos 60-90 min hasta que estén tiernos, y "
         "escúrrelos (puedes cocinar la tanda de varios días y guardarla en la nevera hasta 4 días).")


def test_los_secos_usados_cocidos_traen_su_coccion():
    m = {"ingredients": ["70 g de mapuey", "1 taza de garbanzos secos", "½ cebolla"],
         "recipe": ["Mise en place: pela y corta 70 g de mapuey en cubos; escurre 355 g de garbanzos cocidos.",
                    "El Toque de Fuego: hierve el mapuey 20-25 min. En una olla, incorpora los garbanzos, orégano y sal, "
                    "y cocina 10 min.",
                    "Montaje: sirve."]}
    assert pc.seco_usado_cocido(m) == 1
    assert m["recipe"][1] == _NOTA, m["recipe"][1]
    assert pc.seco_usado_cocido(m) == 0                                          # idempotente


def test_lo_que_ya_se_cuece_de_verdad_no_se_toca():
    casos = [
        ["Mise en place: escurre 355 g de garbanzos cocidos.", "El Toque de Fuego: remoja los garbanzos la noche antes."],
        ["El Toque de Fuego: hierve los garbanzos 60-90 min hasta que estén tiernos; mezcla los garbanzos cocidos."],
        ["El Toque de Fuego: cocina la cebada en agua según las instrucciones del paquete; sirve la cebada cocida."],
        [_NOTA, "Mise en place: escurre 355 g de garbanzos cocidos."],
        ["Mise en place: escurre los garbanzos."],                                  # no dice «cocidos»
    ]
    for pasos in casos:
        m = {"ingredients": ["1 taza de garbanzos secos", "95 g de cebada seca"], "recipe": list(pasos)}
        assert pc.seco_usado_cocido(m) == 0 and m["recipe"] == pasos, pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").seco_usado_cocido(meal)  # [P1-PLAN-LOTE-393]' in src
