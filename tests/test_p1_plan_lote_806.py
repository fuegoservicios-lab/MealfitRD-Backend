# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-806 · 2026-09-29] El huevo que ningún paso cuaja recibe su paso de cocción, no sólo una nota.

Batería real (embarazo, 29-sep): «Tortilla bien cuajada con tomate y aguacate…» batía los huevos y nunca los cuajaba
(«Añade huevos al lado para acompañar»); el detector de seguridad lo marcaba `no_cook` y sólo añadía la nota.
"""
from __future__ import annotations

import copy

import graph_orchestrator as go
import huevo_sin_coccion as hs

_TORTILLA = {
    "name": "Tortilla bien cuajada con tomate y aguacate, guayaba fresca, yogurt griego entero",
    "ingredients": ["3 huevos", "3 tomates medianos", "½ cebolla", "3 dientes de ajo", "½ cdta de aceite de oliva",
                    "½ aguacate", "60 g de guayaba", "¼ taza de yogurt griego entero pasteurizado", "1 clara de huevo"],
    "recipe": ["Mise en place: lava y corta el tomate en cubitos, pica ½ cebolla y 3 dientes de ajo; bate 3 huevos y 1 clara "
               "de huevo y corta ½ aguacate y 1 guayaba.",
               "El Toque de Fuego: calienta ½ cdta de aceite de oliva en una sartén a fuego medio; cocina el tomate, la "
               "cebolla y el ajo durante 3 minutos. Añade huevos al lado para acompañar.",
               "Montaje: sirve la tortilla con el aguacate y la guayaba fresca al lado. Acompaña con yogurt griego entero."]}


def _plan(meal):
    return {"days": [{"day": 1, "meals": [copy.deepcopy(meal)]}]}


def test_la_tortilla_sin_cuajar_recibe_su_paso():
    plan = _plan(_TORTILLA)
    go._apply_food_safety_fixes(plan)
    rec = plan["days"][0]["meals"][0]["recipe"]
    i = next(i for i, p in enumerate(rec) if p.startswith("El Toque de Fuego"))
    assert rec[i].endswith("Añade huevos al lado para acompañar. Vierte los huevos y la clara batidos en la sartén caliente "
                           "con un poco de aceite y cocínalos, removiendo, 3-4 minutos, hasta que cuajen por completo (sin "
                           "partes líquidas)."), rec[i]   # [P1-PLAN-LOTE-863] dentro del Toque de Fuego, no un paso «💪»
    assert rec[i + 1].startswith("Montaje"), rec
    assert any("Seguridad alimentaria" in p for p in rec), "la nota sigue"


def test_si_un_paso_ya_lo_cocina_solo_la_nota():
    guiso = copy.deepcopy(_TORTILLA)
    guiso["name"] = "Guiso ligero de huevos con tomate"
    guiso["recipe"][1] = ("El Toque de Fuego: calienta el aceite; cocina el tomate 3 minutos. Vierte los huevos batidos en la "
                          "salsa y deja que se cocinen 5 minutos, hasta que estén firmes.")
    plan = _plan(guiso)
    go._apply_food_safety_fixes(plan)
    rec = plan["days"][0]["meals"][0]["recipe"]
    assert not any("Vierte los huevos y la clara batidos" in p for p in rec), rec


def test_los_falsos_positivos_del_corpus_no_reciben_paso():
    """Las cinco formas en que el corpus SÍ cocina el huevo y el primer criterio no lo veía (8 de 27 inserciones)."""
    cocidos = (
        ["Mise en place: prepara 2 huevos bien cocidos (10 minutos en agua hirviendo) y pélalos.", "Montaje: sirve."],
        ["Mise en place: corta 3 huevos bien cocidos en mitades.",
         "💡 Cocción previa: hierve los huevos 10-12 min, pásalos a agua fría y pélalos.", "Montaje: sirve."],
        ["Mise en place: bate 6 claras de huevo.",
         "El Toque de Fuego: sazona las claras con ajo y sal; cocínalas en una sartén 3 minutos.", "Montaje: sirve."],
        ["El Toque de Fuego: mezcla el pescado con el huevo y el pan rallado; forma tortitas y hornéalas 15 minutos.",
         "Montaje: sirve."],
        ["El Toque de Fuego: mezcla la quinoa y las vainitas con el huevo y las claras. Forma bocaditos y cocínalos en el "
         "airfryer 10 minutos.", "Montaje: sirve."],
    )
    for rec in cocidos:
        assert hs.cocido_en_pasos({"recipe": rec}), rec
    # «pan integral» no es una mezcla: el huevo de las tostadas sigue sin cocer
    assert not hs.cocido_en_pasos({"recipe": ["Mise en place: prepara 1 huevo y 1 rebanada de pan integral.",
                                              "El Toque de Fuego: calienta el aceite; cocina el tomate 3 minutos. Añade "
                                              "huevo al lado para acompañar.",
                                              "Montaje: coloca el huevo sobre las tostadas integrales."]})


def test_el_criterio_estricto():
    assert not hs.cocido_en_pasos(_TORTILLA)
    assert hs.cocido_en_pasos({"recipe": ["Añade los huevos batidos; remueve hasta que cuajen."]})
    assert hs.cocido_en_pasos({"recipe": ["Hierve los huevos 10 minutos y pélalos."]})
    assert not hs.cocido_en_pasos({"recipe": ["⚠️ Seguridad alimentaria: cocina el huevo por completo."]})
    assert hs.paso({"ingredients": ["2 claras de huevo"], "recipe": []}).startswith(
        "Bate las claras, viértelas en la sartén caliente con un poco de aceite y cocínalas")
