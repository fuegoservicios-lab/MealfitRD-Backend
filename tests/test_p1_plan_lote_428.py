# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-428 · 2026-09-26] El agua de la avena va donde se nombra el líquido, no al final del paso.

Batería REAL sobre el 424 (familia de 4, desayuno del día 2): «…lleva la leche descremada con la canela…; añade la avena y
cocina 8-10 minutos… saltea la pera y el mango… Completa el líquido con 160 ml de agua para que la avena se cocine». Corpus
de 322 planes: 36 frases así, 24 del plan de emergencia con «avena COCIDA» a la que se sumaban 600 ml de agua."""
from __future__ import annotations

import pathlib

import avena_liquido as al

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _desayuno():
    return {"name": "Avena cremosa con pera caramelizada",
            "ingredients": ["¾ taza de avena en hojuelas", "80 ml de leche descremada", "60 g de pera"],
            "recipe": ["Mise en place: mide la avena, la leche descremada y la canela; corta la pera en cubos.",
                       "El Toque de Fuego: en una olla pequeña lleva la leche descremada con la canela a fuego medio-bajo; "
                       "añade la avena y cocina 8-10 minutos hasta que espese. En una sartén saltea la pera 4 minutos.",
                       "Montaje: sirve la avena con la pera."]}


def test_el_agua_entra_con_la_leche_que_el_paso_nombra():
    m = _desayuno()
    agua = al.completar(m)
    assert agua > 0 and m["ingredients"][-1] == f"{agua} ml de agua"
    assert m["recipe"][1].startswith(
        f"El Toque de Fuego: en una olla pequeña lleva la leche descremada y {agua} ml de agua con la canela")
    assert "Completa el líquido" not in " ".join(m["recipe"])


def test_la_frase_que_ya_estaba_al_final_se_mueve_a_su_sitio():
    m = _desayuno()
    m["ingredients"].append("160 ml de agua")
    m["recipe"][1] += " Completa el líquido con 160 ml de agua para que la avena se cocine."
    assert al.ubicar_agua(m) == 1
    assert m["recipe"][1] == ("El Toque de Fuego: en una olla pequeña lleva la leche descremada y 160 ml de agua con la "
                              "canela a fuego medio-bajo; añade la avena y cocina 8-10 minutos hasta que espese. En una "
                              "sartén saltea la pera 4 minutos.")
    assert al.ubicar_agua(m) == 0
    # renal del 302b: «cocina la avena con agua… deja que se enfríe. Completa el líquido con 200 ml de agua…»
    r = {"name": "Bowl fresco de avena", "ingredients": ["½ taza de avena", "200 ml de agua"],
         "recipe": ["El Toque de Fuego: en una olla pequeña, cocina la avena con agua a fuego medio durante 5-7 minutos, "
                    "removiendo hasta que esté suave; deja que se enfríe. Completa el líquido con 200 ml de agua para que "
                    "la avena se cocine."]}
    assert al.ubicar_agua(r) == 1
    assert r["recipe"][0] == ("El Toque de Fuego: en una olla pequeña, cocina la avena con 200 ml de agua a fuego medio "
                              "durante 5-7 minutos, removiendo hasta que esté suave; deja que se enfríe.")


def test_con_el_liquido_y_la_taza_de_agua():
    m = {"name": "Avena con frutas", "ingredients": ["1 taza de avena", "frutas variadas"],
         "recipe": ["El Toque de Fuego: cocina la avena con el líquido a fuego medio 5 minutos, removiendo hasta cremosa."]}
    agua = al.completar(m)
    assert agua > 0 and m["recipe"][0].startswith(f"El Toque de Fuego: cocina la avena con {agua} ml de agua a fuego medio")
    t = {"name": "Avena cremosa", "ingredients": ["100 g de avena", "1 taza de agua"],
         "recipe": ["El Toque de Fuego: lleva 1 taza de agua a hervor; añade la avena y cocina 5-7 minutos hasta que espese."]}
    assert al.completar(t) == 160
    assert t["ingredients"] == ["100 g de avena", "400 ml de agua"]
    assert t["recipe"][0].startswith("El Toque de Fuego: lleva 400 ml de agua a hervor")


def test_la_avena_cocida_no_recibe_agua_y_se_calienta():
    m = {"name": "Huevos y Avena", "ingredients": ["2 huevos", "1¾ tazas de avena cocida", "1 fruta de temporada"],
         "recipe": ["Mise en place: lava, pica y pesa cada ingrediente según las cantidades listadas.",
                    "El Toque de Fuego: cocina la avena con el líquido a fuego medio 5 minutos, removiendo hasta cremosa.",
                    "Montaje: sirve en bowl."]}
    assert al.completar(m) == 0 and len(m["ingredients"]) == 3
    assert al.ubicar_agua(m) == 1
    assert m["recipe"][1] == ("El Toque de Fuego: calienta la avena cocida a fuego medio 2-3 minutos, removiendo hasta que "
                              "esté cremosa.")


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("avena_liquido").ubicar_agua(meal)  # [P1-PLAN-LOTE-428]' in src
