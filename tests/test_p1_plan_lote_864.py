# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-864 · 2026-09-29] Lo que se le hacía al huevo no se le hace a la pechuga que lo sustituyó.

Batería real de embarazo con 806-809 vivos (29-sep 08:18): el autocorrector de proteína repetida cambió el huevo duro de
dos cenas por pechuga de pollo (`_protein_autofix_applied: huevo->pollo`) y el reparador del lote 426 dejó
«💡 Cocción previa: la pechuga de pollo debe alcanzar 74 °C…; pélalos y córtalos en mitades», «acomoda al lado la
pechuga de pollo en tiras pelados y cortados por la mitad» y «ten listos ½ pechuga de pollo bien cocidos».
"""
from __future__ import annotations

import copy

import pasos_cerrador as pc

_CENA_D1 = {
    "_protein_autofix_applied": "huevo->pollo",
    "name": "Casabe crujiente en airfryer con pechuga de pollo, aguacate y zanahoria asada",
    "ingredients": ["1½ casabes pequeños", "½ pechuga de pollo (≈100 g)", "½ zanahoria mediana", "½ aguacate maduro"],
    "recipe": ["Mise en place: corta 1½ casabes pequeños en trozos; ten listos ½ pechuga de pollo bien cocidos; pela y "
               "corta ½ zanahoria mediana en bastones; maja ½ aguacate.",
               "💡 Cocción previa: la pechuga de pollo debe alcanzar 74 °C en la parte más gruesa; pélalos y córtalos en "
               "mitades.",
               "El Toque de Fuego: cocina la zanahoria en airfryer a 200 °C 12-14 min. Cocina pechuga de pollo a la plancha "
               "con unas gotas de aceite 6-7 minutos por lado, hasta que alcance 74 °C en la parte más gruesa, y córtala "
               "en tiras.",
               "Montaje: coloca el casabe como base, encima la zanahoria asada y la pechuga de pollo en tiras."],
}
_CENA_D2 = {
    "_protein_autofix_applied": "huevo->pollo",
    "name": "Remolacha guisada con pechuga de pollo y ensalada de repollo morado al limón",
    "ingredients": ["300 g de remolacha", "½ pechuga de pollo (≈81 g)", "1 taza de repollo morado rallado"],
    "recipe": ["Mise en place: pela y corta 300 g de remolacha en cubos pequeños; ten listos ½ pechuga de pollo.",
               "El Toque de Fuego: cocina la pechuga de pollo a la plancha con unas gotas de aceite 6-7 minutos por lado, "
               "hasta que alcance 74 °C en la parte más gruesa, y córtala en tiras; cocina la remolacha 15-20 min.",
               "Montaje: sirve el guiso de remolacha en un plato, acomoda al lado la pechuga de pollo en tiras pelados y "
               "cortados por la mitad y la ensalada de repollo morado aliñada con el jugo de limón."],
}


def test_la_coccion_previa_pierde_el_pelado_del_huevo():
    m = copy.deepcopy(_CENA_D1)
    assert pc.huevo_sustituido(m) > 0
    assert m["recipe"][1] == "💡 Cocción previa: la pechuga de pollo debe alcanzar 74 °C en la parte más gruesa.", m["recipe"]
    assert "ten lista ½ pechuga de pollo;" in m["recipe"][0], m["recipe"][0]
    assert pc.huevo_sustituido(m) == 0, "idempotente"


def test_el_montaje_no_pela_ni_parte_la_pechuga():
    m = copy.deepcopy(_CENA_D2)
    assert pc.huevo_sustituido(m) > 0
    assert "acomoda al lado la pechuga de pollo en tiras y la ensalada" in m["recipe"][2], m["recipe"][2]
    assert "ten lista ½ pechuga de pollo." in m["recipe"][0], m["recipe"][0]


def test_sin_la_marca_del_sustituto_no_se_toca():
    m = copy.deepcopy(_CENA_D2)
    m.pop("_protein_autofix_applied")
    antes = copy.deepcopy(m)
    assert pc.huevo_sustituido(m) == 0 and m == antes
