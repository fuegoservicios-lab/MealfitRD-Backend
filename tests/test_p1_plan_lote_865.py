# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-865 · 2026-09-29] «…y desmenuza 5 g.» sin alimento también sale (lote 422 sólo conocía corta/pica/mide/pesa).

Batería real de embarazo (29-sep 08:18, D2 desayuno): el queso blanco se fue de la lista y del paso (lote 30) y quedó
«lava y corta 90 g de aguacate en trozos y desmenuza 5 g.». Corpus: 3 («desmenuza 50 g,», «trocea 10 g.»).
"""
from __future__ import annotations

import pasos_cantidades as pc


def _limpio(texto):
    m = {"recipe": [texto]}
    pc.migaja_sin_alimento(m)
    return m["recipe"][0]


def test_los_verbos_de_desmenuzar_sin_alimento_salen():
    assert _limpio("Mise en place: lava y corta 90 g de aguacate en trozos y desmenuza 5 g.") == (
        "Mise en place: lava y corta 90 g de aguacate en trozos.")
    assert _limpio("Mise en place: corta ½ cebolla en cubos pequeños, desmenuza 50 g, mide ½ casabe mediano.") == (
        "Mise en place: corta ½ cebolla en cubos pequeños, mide ½ casabe mediano.")
    assert _limpio("Mise en place: desmenuza 20 g de queso blanco fresco y trocea 10 g.") == (
        "Mise en place: desmenuza 20 g de queso blanco fresco.")


def test_con_su_alimento_no_se_toca():
    t = "Mise en place: desmenuza 20 g de queso blanco fresco y ralla 30 g de zanahoria."
    assert _limpio(t) == t
