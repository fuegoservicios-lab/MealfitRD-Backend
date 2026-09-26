# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-350 · 2026-09-26] «Hierve la yautía» también es cocerla.

Batería real sobre el 331 (adulto mayor con HTA, día 2): «Bowl tropical de yautía enfriada con huevo duro…» — «hierve la
yautía en agua sin sal durante 15-18 minutos, hasta que el cuchillo entre sin fuerza» — recibía «⚠️ Seguridad
alimentaria: NO licúes yuca, víveres ni leguminosas CRUDOS en un batido…»: el detector de víveres crudos reconocía
«hervida/cocida/cocina…» pero no el verbo «hierve»."""
from __future__ import annotations

import graph_orchestrator as go


def _plan(pasos):
    return {"days": [{"day": 2, "meals": [{"meal": "Almuerzo", "name": "Bowl tropical de yautía enfriada con huevo duro",
                                           "ingredients": ["½ pedazo de yautía (≈135 g)", "3 huevos"],
                                           "recipe": list(pasos)}]}]}


def test_hierve_absuelve_al_viver():
    p = _plan(["Mise en place: pela y corta la yautía en cubos.",
               "El Toque de Fuego: hierve la yautía en agua sin sal durante 15-18 minutos, hasta que el cuchillo entre sin "
               "fuerza; escúrrela y déjala enfriar.", "Montaje: combina la yautía con la ensalada."])
    assert go._scan_raw_viver_violations(p) == []


def test_sin_coccion_sigue_marcado():
    p = _plan(["Mise en place: pela y corta la yautía en cubos.", "Montaje: combina la yautía con la ensalada."])
    assert go._scan_raw_viver_violations(p)


def test_ancla():
    assert "hierv" in go._VIVER_LEGUME_COOK_INDICATORS
