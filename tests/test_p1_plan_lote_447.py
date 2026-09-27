# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-447 · 2026-09-27] «aguacate fresca» → «aguacate fresco» (≈120 frases en 322 planes)."""
from __future__ import annotations

import pathlib

import concordancia as co

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_adjetivo_pegado_al_alimento_pasa_a_masculino():
    casos = {
        "Montaje: coloca las tostadas con el huevo revuelto y acompaña con aguacate fresca.":
            "Montaje: coloca las tostadas con el huevo revuelto y acompaña con aguacate fresco.",
        "Mise en place: lava y desinfecta ½ aguacate mediana y córtala en mitades.":
            "Mise en place: lava y desinfecta ½ aguacate mediano y córtalo en mitades.",
        "Montaje: sirve el mangú con aguacate fresca cortada en cubos.":
            "Montaje: sirve el mangú con aguacate fresco cortado en cubos.",
        "Montaje: añade la lechosa en cubos y guineo troceada.": "Montaje: añade la lechosa en cubos y guineo troceado.",
        "Mise en place: mide 170 g de arroz integral cocida.": "Mise en place: mide 170 g de arroz integral cocido.",
        "Montaje: espolvorea maní fileteado tostadas y sirve.": "Montaje: espolvorea maní fileteado tostado y sirve.",
        "Montaje: sirve mango fresca al lado.": "Montaje: sirve mango fresco al lado.",
    }
    for antes, despues in casos.items():
        assert co.concordar(antes) == despues, co.concordar(antes)


def test_lo_que_va_con_de_habla_de_otro_sustantivo():
    for t in ("Montaje: sirve los huevos sobre las tortas de casabe tostadas.",
              "Montaje: añade las rodajas de aguacate frescas.",
              "Mise en place: corta 230 g de pechuga de pollo cocida en trozos."):
        assert co.concordar(t) == t


def test_nombre_y_pasos_pero_no_la_lista():
    m = {"name": "Revoltillo Criollo de Huevo con Pan Integral Tostada",
         "ingredients": ["⅔ taza de arroz integral cocida", "½ aguacate mediano"],
         "recipe": ["Mise en place: corta ½ aguacate mediana.", "El Toque de Fuego: bate los huevos.",
                    "Montaje: sirve con aguacate fresca."]}
    assert co.concordar_masculinos(m) == 3
    assert m["name"] == "Revoltillo Criollo de Huevo con Pan Integral Tostado"
    assert m["recipe"][0] == "Mise en place: corta ½ aguacate mediano."
    assert m["ingredients"][0] == "⅔ taza de arroz integral cocida", "la lista es un identificador: no se toca"
    assert co.concordar_masculinos(m) == 0, "idempotente"


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("concordancia").concordar_masculinos(meal)  # [P1-PLAN-LOTE-447]' in src
