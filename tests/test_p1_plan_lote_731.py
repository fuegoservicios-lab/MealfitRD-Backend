# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-731 · 2026-09-28] Lo que sólo se le hace a un huevo o a un queso no se le hace a su sustituto.

Batería real sobre el 659 (adulto mayor con HTA): «Wok criollo … con pechuga de pollo bien cuajado», «vierte la
pechuga de pollo en tiras», «desmenuza los 40 g de yogurt griego» y «… por encima para que se funda con el calor
residual», todos en platos cuya lista ya no tenía huevo ni queso.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import restos_de_sustitucion as rs  # noqa: E402


def _wok():
    return {
        "name": "Wok criollo de yautía en trozos con pechuga de pollo bien cuajado, brócoli y palmito",
        "ingredients": ["½ pedazo de yautía", "¾ pechuga de pollo (≈150 g)", "2⅔ tazas de brócoli"],
        "recipe": [
            "Mise en place: pela la yautía y córtala en trozos.",
            "El Toque de Fuego: Empuja los vegetales a un lado, vierte la pechuga de pollo en tiras y cocina 8-10 "
            "minutos, hasta que el pollo alcance 74 °C por dentro y quede bien cuajado.",
            "Montaje: sirve caliente.",
        ],
    }


def test_el_pollo_no_se_cuaja_ni_se_vierte():
    m = _wok()
    assert rs.limpiar(m) == 2
    assert m["name"] == "Wok criollo de yautía en trozos con pechuga de pollo, brócoli y palmito"
    assert m["recipe"][1] == ("El Toque de Fuego: Empuja los vegetales a un lado, añade la pechuga de pollo en tiras y "
                              "cocina 8-10 minutos, hasta que el pollo alcance 74 °C por dentro y quede bien cocido.")
    assert m["recipe"][0] == _wok()["recipe"][0] and m["recipe"][2] == _wok()["recipe"][2]


def test_el_yogur_no_se_desmenuza_ni_se_funde():
    m = {
        "name": "Arepitas finas de trigo con salteado de brócoli, palmito y yogur griego",
        "ingredients": ["40 g de yogurt griego sin azúcar", "15 g de harina de trigo"],
        "recipe": [
            "Mise en place: desmenuza los 40 g de yogurt griego sin azúcar bajo en sodio.",
            "El Toque de Fuego: saltea el brócoli 5 minutos. Retira del fuego y desmenuza yogurt griego sin azúcar "
            "por encima para que se funda con el calor residual.",
        ],
    }
    assert rs.limpiar(m) == 2
    # en la mise en place se MIDE (replay: «Mide el aceite…; añade los 40 g de yogurt» no se lee)
    assert m["recipe"][0] == "Mise en place: mide los 40 g de yogurt griego sin azúcar bajo en sodio."
    assert m["recipe"][1] == ("El Toque de Fuego: saltea el brócoli 5 minutos. Retira del fuego y añade yogurt griego "
                              "sin azúcar por encima.")


def test_con_huevo_en_la_lista_no_se_toca():
    m = {"name": "Tortilla de huevo bien cuajada", "ingredients": ["3 huevos", "½ taza de espinaca"],
         "recipe": ["El Toque de Fuego: vierte los huevos batidos y cocina hasta que estén bien cuajados."]}
    antes = {"name": m["name"], "recipe": list(m["recipe"])}
    assert rs.limpiar(m) == 0
    assert m["name"] == antes["name"] and m["recipe"] == antes["recipe"]


def test_con_queso_en_la_lista_el_queso_se_sigue_desmenuzando():
    m = {"name": "Mangú con queso de freír", "ingredients": ["60 g de queso de freír", "½ taza de yogurt natural"],
         "recipe": ["El Toque de Fuego: desmenuza el queso por encima para que se funda con el calor residual.",
                    "Montaje: sirve el yogurt natural al lado."]}
    antes = list(m["recipe"])
    assert rs.limpiar(m) == 0 and m["recipe"] == antes


def test_lo_que_se_funde_de_verdad_se_sigue_fundiendo():
    m = {"name": "Batata asada con yogur", "ingredients": ["1 batata", "½ taza de yogurt griego", "5 g de mantequilla"],
         "recipe": ["El Toque de Fuego: añade la mantequilla y el yogurt griego para que se funda con el calor residual."]}
    antes = list(m["recipe"])
    assert rs.limpiar(m) == 0 and m["recipe"] == antes


def test_las_notas_no_se_tocan():
    nota = "⚠️ Seguridad alimentaria: no desmenuces el yogurt; sírvelo frío."
    m = {"name": "Yogur con fruta", "ingredients": ["1 taza de yogurt natural"], "recipe": [nota]}
    assert rs.limpiar(m) == 0 and m["recipe"] == [nota]


def test_enganchado_en_la_cola_del_contrato():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i = src.index('__import__("restos_de_sustitucion").limpiar(meal)')
    assert i < src.index('__import__("doble_punto").limpiar(meal)')
    assert i > src.index('__import__("pasos_cerrador").huevo_sustituido(meal)')
