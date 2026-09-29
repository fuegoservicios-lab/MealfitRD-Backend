# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-809 · 2026-09-29] «el huevo duros» → «el huevo duro»: el participio sigue al número del huevo.

Corpus (5.334 comidas): 23 pasos así, ya presentes ANTES del escudo (el pase que baja «los huevos» a «el huevo» no toca
el adjetivo que los seguía).
"""
from __future__ import annotations

import pathlib

import huevo_concuerda as hc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_cadena_pegada_al_huevo_toma_su_numero():
    casos = {
        "Montaje: acompaña con el huevo duros y las claras cocidas.":
            "Montaje: acompaña con el huevo duro y las claras cocidas.",
        "Montaje: sirve el mangú con la cebolla salteada y el huevo pelados y cortados; acompaña con aguacate fresco.":
            "Montaje: sirve el mangú con la cebolla salteada y el huevo pelado y cortado; acompaña con aguacate fresco.",
        "Montaje: acompaña con el huevo cocido pelados.": "Montaje: acompaña con el huevo cocido pelado.",
        "Montaje: agrega el jugo de limón y el huevo bien cocidos cortados en cuartos.":
            "Montaje: agrega el jugo de limón y el huevo bien cocido cortado en cuartos.",
        "Mise en place: desmenuza 1 huevo bien cocidos; mide ½ cdta de aceite de oliva.":
            "Mise en place: desmenuza 1 huevo bien cocido; mide ½ cdta de aceite de oliva.",
        "El Toque de Fuego: hierve el plátano y el huevo 15 min, hasta que el plátano esté tierno y el huevo estén cocidos.":
            "El Toque de Fuego: hierve el plátano y el huevo 15 min, hasta que el plátano esté tierno y el huevo esté cocido.",
        "El Toque de Fuego: remueve 3-4 minutos hasta que el huevo estén cuajados.":
            "El Toque de Fuego: remueve 3-4 minutos hasta que el huevo esté cuajado.",
        "Mise en place: prepara 3 huevos bien cocido y mide ¼ cdta de canela en polvo.":
            "Mise en place: prepara 3 huevos bien cocidos y mide ¼ cdta de canela en polvo.",
        "El Toque de Fuego: incorpora el huevo bien cocidos para calentarlos 1 min.":
            "El Toque de Fuego: incorpora el huevo bien cocido para calentarlo 1 min.",
    }
    for antes, despues in casos.items():
        assert hc.concordar(antes) == despues, hc.concordar(antes)


def test_lo_que_no_va_pegado_no_se_toca():
    for t in ("Montaje: acompaña con el huevo y las claras cocidas.",
              "El Toque de Fuego: vierte el huevo y revuelve 3-4 min, hasta que estén cuajados.",
              "El Toque de Fuego: hierve 2 huevos bien cocidos y pélalos.",
              "Montaje: sirve el huevo cocido y cortado por la mitad.",
              "Mise en place: mezcla 2 huevos batidos con la avena."):
        assert hc.concordar(t) == t


def test_pasos_si_notas_y_lista_no():
    nota = "⚠️ Seguridad alimentaria: sirve el huevo duros solo si está bien cocido."
    m = {"ingredients": ["1 huevo", "Huevos duros pelados"],
         "recipe": ["Mise en place: pela el huevo cocido.", "Montaje: sirve el huevo duros al lado.", nota]}
    assert hc.concordar_pasos(m) == 1
    assert m["recipe"][1] == "Montaje: sirve el huevo duro al lado."
    assert m["recipe"][2] == nota, "las notas se reconocen por su texto exacto: no se tocan"
    assert m["ingredients"] == ["1 huevo", "Huevos duros pelados"], "la lista es un identificador"
    assert hc.concordar_pasos(m) == 0, "idempotente"


def test_ancla_en_la_cola_del_contrato():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i = src.index('__import__("huevo_concuerda").concordar_pasos(meal)  # [P1-PLAN-LOTE-809]')
    assert src.index('__import__("concordancia").concordar_masculinos(meal)  # [P1-PLAN-LOTE-447]') < i
    assert i < src.index('__import__("doble_punto").limpiar(meal)')
