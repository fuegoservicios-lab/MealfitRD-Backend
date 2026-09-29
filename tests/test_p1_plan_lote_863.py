# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-863 · 2026-09-29] Las instrucciones de seguridad (806 huevo, 807 queso) van dentro de la receta y se
reponen cada vez que corre el etiquetador.

Batería real de embarazo con 806-809 vivos (29-sep 08:18): «Nabo crujiente en airfryer con lentejas guisadas y queso
fresco» llegó con la nota «caliéntalo hasta que humee» y sin el paso de dorar el queso — la comida se reescribió después
de la primera pasada y el etiquetador del escudo, al ver la nota, salía sin reponerlo.
"""
from __future__ import annotations

import copy
import pathlib

import embarazo_seguro as es
import etiquetas_clinicas as ec
import graph_orchestrator as go
import huevo_sin_coccion as hs
import paso_de_seguridad as ps

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_NOTA_QUESO = es._NOTA_QUESO_QUE_HUMEE
_NABO = {
    "name": "Nabo crujiente en airfryer con lentejas guisadas y queso fresco",
    "ingredients": ["30 g de nabo", "1 taza de lentejas cocidas", "20 g de queso blanco fresco pasteurizado"],
    "recipe": ["Mise en place: corta el nabo en bastones y mide las lentejas y el queso.",
               "El Toque de Fuego: cocina el nabo en airfryer a 200 °C 14-16 min; calienta las lentejas 5 min.",
               "Montaje: sirve las lentejas con el nabo y desmenuza el queso blanco fresco pasteurizado por encima.",
               _NOTA_QUESO],
}


def test_con_la_nota_ya_puesta_el_escudo_repone_el_paso_del_queso():
    plan = {"days": [{"day": 1, "meals": [copy.deepcopy(_NABO)]}]}
    ec.etiquetar(plan, {"medicalConditions": ["Embarazo"]})
    rec = plan["days"][0]["meals"][0]["recipe"]
    assert rec[1].endswith("Dora el queso blanco pasteurizado en la sartén caliente, 1-2 minutos por lado, hasta que humee "
                           "y esté bien caliente por dentro (74 °C)."), rec
    assert sum(p == _NOTA_QUESO for p in rec) == 1, "la nota no se duplica"
    antes = copy.deepcopy(plan)
    ec.etiquetar(plan, {"medicalConditions": ["Embarazo"]})
    assert plan == antes, "idempotente"


def test_con_la_nota_de_huevo_ya_puesta_se_repone_el_paso_de_cuajar():
    tortilla = {"name": "Tortilla de tomate", "ingredients": ["2 huevos", "1 tomate"],
                "recipe": ["Mise en place: bate 2 huevos y corta el tomate.",
                           "El Toque de Fuego: cocina el tomate 3 minutos. Añade huevos al lado para acompañar.",
                           "Montaje: sirve la tortilla.", go._FOOD_SAFETY_NOTE_NOCOOK]}
    plan = {"days": [{"day": 1, "meals": [tortilla]}]}
    assert go._scan_raw_egg_violations(plan) == [], "con la nota puesta el detector ya no la marca: por eso se repone aparte"
    assert hs.reasegurar(plan, go._FOOD_SAFETY_NOTE_NOCOOK) == 1
    assert "Vierte los huevos batidos en la sartén caliente" in plan["days"][0]["meals"][0]["recipe"][1]
    assert hs.reasegurar(plan, go._FOOD_SAFETY_NOTE_NOCOOK) == 0, "idempotente"
    src = (_BACKEND / "etiquetas_clinicas.py").read_text(encoding="utf-8")
    assert '__import__("huevo_sin_coccion").reasegurar(plan, __import__("graph_orchestrator")._FOOD_SAFETY_NOTE_NOCOOK)' in src


def test_sin_toque_de_fuego_la_instruccion_es_su_propio_paso():
    rec = ps.poner(["Mise en place: corta el queso.", "Montaje: sirve."], "Dora el queso en la sartén.")
    assert rec == ["Mise en place: corta el queso.", "El Toque de Fuego: dora el queso en la sartén.", "Montaje: sirve."]
    rec = ps.poner(["El Toque de Fuego: tuesta el pan", "Montaje: sirve."], "Dora el queso en la sartén.")
    assert rec[0] == "El Toque de Fuego: tuesta el pan. Dora el queso en la sartén."
