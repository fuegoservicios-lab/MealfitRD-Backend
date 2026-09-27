# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-441 · 2026-09-26] El ave que la lista compra cruda no se «calienta»: se cocina hasta 74 °C.

Replay de la cola sobre 322 planes: «Incorpora el pollo desmenuzado y las habichuelas, cocina 5 minutos y comprueba que el
pollo esté bien caliente» (EMBARAZO, pechuga cruda en la lista, ningún paso que la cocine); «guisa pechuga de pollo … 6-8
min, hasta que … pechuga de pollo bien caliente» (el atún en lata pasó a pollo)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_pollo_desmenuzado_crudo_trae_su_coccion_previa():
    m = {"ingredients": ["¾ pechuga de pollo (≈150 g)", "½ taza de habichuelas rojas cocidas"],
         "recipe": ["Mise en place: pica la cebolla.",
                    "El Toque de Fuego: sofríe la cebolla. Incorpora el pollo desmenuzado y las habichuelas, cocina 5 "
                    "minutos y comprueba que el pollo esté bien caliente.",
                    "Montaje: sirve."]}
    assert pc.ave_desmenuzada_cruda(m) == 1
    assert m["recipe"][1].startswith("💡 Cocción previa: cocina la pechuga de pollo en agua con sal 15-18 min")
    assert pc.ave_hasta_74(m) == 0, "con su cocción previa, calentarla basta"
    assert pc.ave_desmenuzada_cruda(m) == 0


def test_el_pollo_crudo_que_solo_se_calienta_llega_a_74():
    m = {"ingredients": ["½ pechuga de pollo (≈100 g)", "½ tomate mediano"],
         "recipe": ["El Toque de Fuego: guisa pechuga de pollo con el tomate a fuego medio durante 6-8 min, hasta que el "
                    "sofrito esté tierno y pechuga de pollo bien caliente. Aparte, saltea la tayota.",
                    "Montaje: sirve."]}
    assert pc.ave_hasta_74(m) == 1
    assert m["recipe"][0] == ("El Toque de Fuego: guisa pechuga de pollo con el tomate a fuego medio durante 6-8 min, "
                              "hasta que el sofrito esté tierno y el pollo alcance 74 °C por dentro. Aparte, saltea la "
                              "tayota.")
    w = {"ingredients": ["¾ pechuga de pollo (≈140 g)"],
         "recipe": ["El Toque de Fuego: añade la cebolla y el pollo, y cocina 4-5 minutos removiendo hasta que el pollo "
                    "esté bien caliente."]}
    assert pc.ave_hasta_74(w) == 1
    assert w["recipe"][0] == ("El Toque de Fuego: añade la cebolla y el pollo, y cocina 8-10 minutos removiendo hasta que "
                              "el pollo alcance 74 °C por dentro.")


def test_lo_cocido_o_con_su_punto_no_se_toca():
    pasos = ["El Toque de Fuego: añade el pollo cocido y calienta hasta que esté bien caliente."]
    m = {"ingredients": ["1 taza de pollo cocido desmenuzado"], "recipe": list(pasos)}
    assert pc.ave_hasta_74(m) == 0 and pc.ave_desmenuzada_cruda(m) == 0 and m["recipe"] == pasos
    p74 = ["El Toque de Fuego: cocina el pollo 8 min hasta 74 °C. Luego caliéntalo bien caliente con la salsa."]
    q = {"ingredients": ["1 pechuga de pollo"], "recipe": list(p74)}
    assert pc.ave_hasta_74(q) == 0 and q["recipe"] == p74


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").ave_desmenuzada_cruda(meal)  # [P1-PLAN-LOTE-441]' in src
    assert '__import__("pasos_cantidades").ave_hasta_74(meal)  # [P1-PLAN-LOTE-441]' in src
