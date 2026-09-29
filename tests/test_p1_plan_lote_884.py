# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-884 · 2026-09-29] La mise en place no desmenuza la carne que se cocina en la «💡 Cocción previa».

Medidores sobre las baterías del escudo actual: el aviso más frecuente (V7f, ~15 % de los planes) — «Mise en place:
desmenuza 230 g de pechuga de pollo» antes de la cocción previa que la cuece. Corpus: 54 comidas con el código actual.
"""
from __future__ import annotations

import pathlib

import mise_sin_desmenuzar as ms

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_PREVIA_AVE = ("💡 Cocción previa: cocina la pechuga de pollo en agua con sal 15-18 min, o a la plancha 5-7 min por lado, "
               "hasta que no quede rosada por dentro (74 °C al centro); déjala reposar y desmenúzala o córtala como pide la "
               "receta.")


def test_con_coccion_previa_la_mise_en_place_solo_la_tiene_a_mano():
    m = {"recipe": ["Mise en place: desmenuza 230 g de pechuga de pollo; pica la cebolla.", _PREVIA_AVE, "Montaje: sirve."]}
    assert ms.ordenar(m) == 1
    assert m["recipe"][0] == "Mise en place: mide 230 g de pechuga de pollo; pica la cebolla."
    assert ms.ordenar(m) == 0, "idempotente"
    m2 = {"recipe": ["Mise en place: desmenuza ½ pechuga de pollo (≈124 g) ya cocida (verifica 74 °C en la parte más "
                     "gruesa).", _PREVIA_AVE]}
    assert ms.ordenar(m2) == 1 and m2["recipe"][0] == "Mise en place: ten a mano ½ pechuga de pollo (≈124 g).", m2


def test_el_pescado_se_desmenuza_despues_de_su_coccion_previa():
    m = {"recipe": ["Mise en place: desmenuza 1 filete de pescado (180 g).",
                    "💡 Cocción previa: cocina el filete de pescado a la plancha o al vapor 3-4 min por lado, hasta que se "
                    "desmenuce fácilmente (63 °C al centro)."]}
    assert ms.ordenar(m) == 1
    assert m["recipe"][0] == "Mise en place: ten a mano 1 filete de pescado (180 g)."
    assert m["recipe"][1].endswith("(63 °C al centro). Luego desmenúzalo.")


def test_sin_coccion_previa_no_se_toca_y_ancla():
    t = "Mise en place: desmenuza 225 g de filete de pescado blanco."
    m = {"recipe": [t, "El Toque de Fuego: mezcla el pescado con limón; forma tortitas y hornéalas 15 minutos."]}
    assert ms.ordenar(m) == 0 and m["recipe"][0] == t, "las tortitas de pescado crudo se hornean: el desmenuzado vale"
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("mise_sin_desmenuzar").ordenar(meal)  # [P1-PLAN-LOTE-884]' in src
