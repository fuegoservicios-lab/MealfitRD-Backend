# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-868 · 2026-09-29] El queso blanco del embarazo se CALIENTA hasta que humee, no se «dora».

Segunda batería real de embarazo (860-864 vivos): el escáner culinario (contrato V1, «dorar» a un listo-para-comer) marcó
las 3 comidas con el paso del 807/863 «Dora el queso blanco pasteurizado… hasta que humee»; con «Calienta…», 0 avisos. El
reparador del lote 426 ya lo escribía así («calienta el queso blanco en la sartén… hasta que humee»).
"""
from __future__ import annotations

import copy

import embarazo_seguro as es
import queso_que_humee as q

_EMBARAZO = {"medicalConditions": ["Embarazo"]}
_MERIENDA = {
    "name": "Casabe crujiente con queso blanco fresco y guayaba",
    "ingredients": ["1 torta pequeña de casabe", "20 g de queso blanco fresco pasteurizado", "60 g de guayaba"],
    "recipe": ["Mise en place: corta la guayaba.", "El Toque de Fuego: tuesta el casabe 1-2 min por lado.",
               "Montaje: coloca el queso blanco y la guayaba sobre el casabe."],
}


def _etiquetar(meal):
    plan = {"days": [{"day": 1, "meals": [copy.deepcopy(meal)]}]}
    es.etiquetar(plan, _EMBARAZO)
    return plan["days"][0]["meals"][0]["recipe"]


def test_el_queso_blanco_se_calienta_no_se_dora():
    rec = _etiquetar(_MERIENDA)
    assert rec[1].endswith("Calienta el queso blanco pasteurizado en la sartén caliente, 1-2 minutos por lado, hasta que "
                           "humee y esté bien caliente por dentro (74 °C)."), rec[1]
    assert not any("Dora el queso" in p for p in rec)


def test_la_frase_vieja_de_un_plan_guardado_pasa_al_verbo_del_contrato():
    viejo = copy.deepcopy(_MERIENDA)
    viejo["recipe"][1] += (" Dora el queso blanco pasteurizado en la sartén caliente, 1-2 minutos por lado, hasta que humee y "
                           "esté bien caliente por dentro (74 °C).")
    rec = _etiquetar(viejo)
    assert "Calienta el queso blanco pasteurizado en la sartén caliente" in rec[1] and "Dora el queso" not in rec[1], rec[1]
    assert sum("hasta que humee y esté bien caliente" in p for p in rec) == 1, "no se añade otro"
    assert _etiquetar({**viejo, "recipe": rec}) == rec, "idempotente"


def test_el_queso_de_freir_si_se_dora():
    frito = {**_MERIENDA, "ingredients": ["1 torta pequeña de casabe", "40 g de queso de freir", "60 g de guayaba"]}
    meal = copy.deepcopy(frito)
    assert q.insertar_paso(meal)
    assert "Dora el queso de freír en la sartén caliente" in meal["recipe"][1], meal["recipe"]
