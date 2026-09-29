# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-883 · 2026-09-29] V4 no compara el arroz crudo de la lista con el arroz cocido del paso.

Batería real RD (mujer que pierde grasa, código 868): los dos únicos avisos del escáner eran V4 «Arroz integral:
ingrediente declara 45 g, pasos declaran 125 g» — «45 g de arroz integral crudo» y «mide 125 g de arroz integral
cocido», la MISMA cantidad en dos bases. La regla (a) del V4: nunca inventar una conversión, sólo gramos de la misma forma.
"""
from __future__ import annotations

import culinary_coherence as cc

_CAT = [{"name": "Arroz integral", "prep_methods": ["hervir"], "ready_to_eat": False},
        {"name": "Queso de hoja", "prep_methods": ["ninguno", "crudo"], "ready_to_eat": True}]


def _plan(pasos, ingredientes):
    return {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Plato", "ingredients": ingredientes,
                                           "recipe": pasos}]}]}


def _v4(pasos, ingredientes):
    return [x for x in cc.culinary_contract_scan(_plan(pasos, ingredientes), _CAT) if x["check"] == "V4"]


def test_crudo_en_la_lista_y_cocido_en_el_paso_no_es_v4():
    assert not _v4(["Mise en place: mide 125 g de arroz integral cocido.",
                    "💡 Cocción previa: enjuaga el arroz integral crudo y cuécelo en agua 35-45 min."],
                   ["45 g de arroz integral crudo"])


def test_la_misma_forma_sigue_comparandose():
    assert _v4(["Mise en place: mide 125 g de arroz integral cocido."], ["60 g de arroz integral cocido"])
    assert _v4(["Mise en place: pesa 45 g de queso de hoja."], ["30 g de queso de hoja"]), "el caso real del V4 sigue"
