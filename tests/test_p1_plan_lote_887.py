# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-887 · 2026-09-29] La mise en place no pela ni corta el huevo que hierve la «💡 Cocción previa».

Corpus + replays del escudo (5.281 comidas únicas): 6 así, casi todas mangú con huevo — «Mise en place: … lava aguacate y
corta 3 huevos bien cocidos en mitades» y DESPUÉS «💡 Cocción previa: hierve los huevos 10-12 min, pásalos a agua fría y
pélalos» (batería real de embarazo del 28-sep, rdv801). Un huevo crudo no se pela ni se corta.
"""
from __future__ import annotations

import pathlib

import mise_sin_desmenuzar as msd

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_PREVIA = "💡 Cocción previa: hierve los huevos 10-12 min, pásalos a agua fría y pélalos."


def _meal(mise, previa=_PREVIA):
    return {"recipe": [mise, previa, "El Toque de Fuego: hierve el plátano verde 15-18 min y májalo.",
                       "Montaje: sirve el mangú con los huevos y el aguacate."]}


def test_el_huevo_se_tiene_a_mano_y_el_corte_va_tras_hervirlo():
    m = _meal("Mise en place: pela y corta el plátano verde en trozos; lava aguacate y corta 3 huevos bien cocidos en "
              "mitades.")
    assert msd.huevo(m) == 1
    assert m["recipe"][0] == "Mise en place: pela y corta el plátano verde en trozos; lava aguacate y ten a mano 3 huevos."
    assert m["recipe"][1] == _PREVIA + " Luego córtalos en mitades."
    assert msd.huevo(m) == 0, "idempotente"


def test_rebana_y_claras():
    previa = ("💡 Cocción previa: hierve los huevos 10-12 min, pásalos a agua fría y pélalos; para las claras de huevo de "
              "la lista, hierve también un huevo por cada clara y quítale la yema al pelarlo.")
    m = _meal("Mise en place: corta la yautía en cubos pequeños, pela y rebana 3 huevos y 1 clara de huevo duros ya "
              "cocidos, pica el ajo y exprime el limón.", previa)
    assert msd.huevo(m) == 1
    assert m["recipe"][0] == ("Mise en place: corta la yautía en cubos pequeños, ten a mano 3 huevos y 1 clara de huevo, "
                              "pica el ajo y exprime el limón.")
    assert m["recipe"][1] == previa + " Luego rebánalos."


def test_solo_pelar_no_añade_corte():
    m = _meal("Mise en place: pica ½ cebolla; pela 3 huevos y 2 claras de huevo cocidos.")
    assert msd.huevo(m) == 1
    assert m["recipe"][0] == "Mise en place: pica ½ cebolla; ten a mano 3 huevos y 2 claras de huevo."
    assert m["recipe"][1] == _PREVIA


def test_sin_coccion_previa_del_huevo_o_con_el_knob_apagado_nada(monkeypatch):
    mise = "Mise en place: corta 2 huevos duros en rodajas."
    m = _meal(mise, "💡 Cocción previa: hierve la yuca 20 min.")
    assert msd.huevo(m) == 0 and m["recipe"][0] == mise
    monkeypatch.setenv("MEALFIT_MISE_HUEVO_SIN_CORTAR", "false")
    m = _meal(mise)
    assert msd.huevo(m) == 0 and m["recipe"][0] == mise


def test_ancla_despues_del_409():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i409 = src.index('__import__("pasos_cantidades").huevo_duro_de_la_lista(meal)  # [P1-PLAN-LOTE-409]')
    i887 = src.index('__import__("mise_sin_desmenuzar").huevo(meal)  # [P1-PLAN-LOTE-887]')
    assert i409 < i887
    assert "tooltip-anchor: P1-PLAN-LOTE-887" in (_BACKEND / "mise_sin_desmenuzar.py").read_text(encoding="utf-8")
