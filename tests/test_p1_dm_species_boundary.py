"""[P1-DM-SPECIES-BOUNDARY · 2026-07-28] "res" ⊂ "fResco" — 11ª mordida de subcadena
del maratón, cazada EN VIVO (corr=6acd0c94, primera generación con audit-3 desplegado):
`_meal_double_main_resolve` (P1-VISUAL-AUDIT-2) eliminó 'Cilantro fresco para decorar' y
'0.75 cda de cilantro fresco picado' de dos platos de CERDO creyendo que eran "res" (2ª
proteína principal). 2 de sus 3 disparos en ese plan fueron falsos positivos.

Dos curas apiladas:
1. Léxica: `_DM_TOKEN_RX` — "res" solo como palabra completa (res/reses; fresco/fresas/
   refrescante/reservada fuera); el resto de especies por stem con inicio de palabra
   (camarones/pescados/atunes siguen matcheando).
2. Semántica: si la "2ª principal" resuelve macros con proteína <3 g, es adorno/condimento
   mal clasificado → NO se remueve (el cilantro vivo resolvió factor ×1.00).

tooltip-anchor: P1-DM-SPECIES-BOUNDARY
"""
from __future__ import annotations

import graph_orchestrator as go


def test_dm_species_word_boundary():
    """[P1-DM-SPECIES-BOUNDARY · 2026-07-28] "res" ⊂ "fResco" (11ª mordida de subcadena,
    cazada EN VIVO corr=6acd0c94): el resolver de doble principal eliminó 'Cilantro fresco
    para decorar' de dos platos de cerdo creyendo que era res. Detección por frontera de
    palabra: 'res' solo como palabra (res/reses); el resto de especies con inicio de palabra
    (camarones/pescados siguen matcheando por stem)."""
    assert go._dm_species_of("cilantro fresco para decorar") == set()
    assert go._dm_species_of("0.75 cda de cilantro fresco picado") == set()
    assert go._dm_species_of("batido refrescante de fresas") == set()
    assert go._dm_species_of("1 taza de agua (reservada)") == set()
    assert go._dm_species_of("carne de res molida") == {"res"}
    assert go._dm_species_of("filete de res") == {"res"}
    assert go._dm_species_of("65g de camarones cocido") == {"camaron"}
    assert go._dm_species_of("filete de pescado blanco fresco") == {"pescado"}


def test_double_main_keeps_garnish_and_low_protein_lines():
    """El caso vivo completo: plato de CERDO con cilantro de adorno → 0 remociones;
    el caso legítimo (pollo + camarones) sigue resolviendo."""
    days = [{"meals": [
        {"name": "Pimientos Rellenos de Cerdo Magro",
         "ingredients": ["150 g de cerdo magro", "Cilantro fresco para decorar"],
         "recipe": ["Rellena los pimientos."]},
        {"name": "Pollo Salteado al Wok",
         "ingredients": ["120 g de pechuga de pollo", "65 g de camarones cocidos"],
         "recipe": ["Saltea el pollo."]},
    ]}]
    assert go._meal_double_main_resolve(days) == 1
    m0 = days[0]["meals"][0]
    assert any("Cilantro" in s for s in m0["ingredients"]), \
        f"el adorno jamás es 2ª proteína principal: {m0['ingredients']}"
    m1 = days[0]["meals"][1]
    assert not any("camarones" in s for s in m1["ingredients"])
