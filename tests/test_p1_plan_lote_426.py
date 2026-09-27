# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-426 · 2026-09-26] Lo que el tope de huevo cambió por pollo, queso o yogur ya no se trata como huevo.

Batería REAL sobre el 424 (familia de 4, cena del día 2): «Aparte, hierve pechuga de pollo 10-12 minutos, pélalos y
desmenúzalos». Corpus de 322 planes: 71 comidas con la marca «huevo->X», con «casca los 2 queso blanco», «bate 1 pechuga
de pollo entero con 1 pechuga de pollo», «corona con las mitades de pechuga de pollo duro», «hasta que el agua salga
pechuga de pollo» y nombres como «Tostadas… con queso blanco fresco, aguacate, queso blanco cuajado y queso blanco»."""
from __future__ import annotations

import pathlib

import pasos_cerrador as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_pollo_que_era_huevo_duro_se_cocina_como_pollo():
    m = {"_protein_autofix_applied": "huevo->pollo", "name": "Yuca guisada con pechuga de pollo",
         "ingredients": ["100 g de yuca", "¾ pechuga de pollo (≈150 g)", "½ tomate mediano"],
         "recipe": ["Mise en place: pela y corta la yuca en trozos; corta el tomate en cubos.",
                    "El Toque de Fuego: hierve la yuca 20-25 minutos. Aparte, hierve pechuga de pollo 10-12 minutos, "
                    "pélalos y desmenúzalos. Sofríe el tomate 8 minutos. Incorpora la yuca cocida y pechuga de pollo "
                    "desmenuzado y cocina 3 minutos.",
                    "Montaje: sirve la yuca con pechuga de pollo en su salsa.",
                    "⚠️ Seguridad alimentaria: cocina pechuga de pollo por completo antes de servir; evita consumirlo crudo "
                    "o poco cocido."]}
    assert pc.huevo_sustituido(m) > 0
    assert m["recipe"][1] == (
        "El Toque de Fuego: hierve la yuca 20-25 minutos. Aparte, cocina la pechuga de pollo a la plancha con unas gotas "
        "de aceite 6-7 minutos por lado, hasta que alcance 74 °C en la parte más gruesa, y desmenúzala. Sofríe el tomate 8 "
        "minutos. Incorpora la yuca cocida y pechuga de pollo desmenuzada y cocina 3 minutos.")
    assert m["recipe"][3] == ("⚠️ Seguridad alimentaria: cocina la pechuga de pollo por completo antes de servir; evita "
                              "consumirla cruda o poco cocida.")
    assert pc.huevo_sustituido(m) == 0


def test_el_queso_que_era_huevo_no_se_casca_ni_se_cuaja():
    m = {"_protein_autofix_applied": "huevo->queso", "name": "Guiso criollo de frijoles rojos con queso blanco",
         "ingredients": ["50 g de frijoles rojos secos", "30 g de queso blanco"],
         "recipe": ["Mise en place: mide 135 g de frijoles rojos cocidos; casca los 2 queso blanco; pica ½ tomate.",
                    "El Toque de Fuego: guisa los frijoles 8-10 minutos. Cocina queso blanco aparte, bien cuajados "
                    "(queso blanco firme), 6-8 minutos.",
                    "Montaje: sirve los frijoles coronados con queso blanco bien cocidos, junto a la ensalada."]}
    assert pc.huevo_sustituido(m) > 0
    assert m["recipe"] == [
        "Mise en place: mide 135 g de frijoles rojos cocidos; corta el queso blanco en cubos; pica ½ tomate.",
        "El Toque de Fuego: guisa los frijoles 8-10 minutos.",
        "Montaje: sirve los frijoles coronados con el queso blanco en cubos, junto a la ensalada."]
    # fuera del embarazo el queso blanco no se cocina (el contrato marca «dorar» un listo-para-comer); con embarazo, se
    # calienta hasta que humee
    e = {"_protein_autofix_applied": "huevo->queso", "name": "Casabe con queso blanco bien cocido",
         "ingredients": ["2 tortas pequeñas de casabe", "30 g de queso blanco pasteurizado"],
         "recipe": ["El Toque de Fuego: Tuesta el casabe. Aparte, hierve 4 queso blanco pasteurizado 10-12 minutos hasta "
                    "que queden bien cocidos (queso blanco pasteurizado firme), enfría en agua y pélalos.",
                    "Montaje: Sirve el casabe con queso blanco pasteurizado duros cortados en rodajas encima.",
                    "🤰 Seguridad alimentaria (embarazo/lactancia): cocina queso blanco pasteurizado POR COMPLETO (queso "
                    "blanco pasteurizado firmes, sin puntos líquidos); lava y desinfecta las frutas antes de usarlas."]}
    assert pc.huevo_sustituido(e) > 0
    assert e["recipe"][0] == ("El Toque de Fuego: Tuesta el casabe. Aparte, calienta el queso blanco en la sartén 1-2 "
                              "minutos por lado, hasta que humee por dentro.")
    assert e["recipe"][1] == "Montaje: Sirve el casabe con el queso blanco caliente encima."
    assert e["recipe"][2] == ("🤰 Seguridad alimentaria (embarazo/lactancia): lava y desinfecta las frutas antes de "
                              "usarlas.")
    assert e["name"] == "Casabe con queso blanco"
    assert pc.huevo_sustituido(e) == 0


def test_el_pollo_batido_entra_en_tiras_y_llega_a_74():
    m = {"_protein_autofix_applied": "huevo->pollo", "name": "Bulgur salteado con pechuga de pollo",
         "ingredients": ["35 g de bulgur", "1¼ pechugas de pollo (≈270 g)"],
         "recipe": ["Mise en place: enjuaga 35 g de bulgur; bate 1 pechuga de pollo entero con 1 pechuga de pollo; mide "
                    "1 cdta de jugo de limón.",
                    "El Toque de Fuego: saltea el bulgur 1 minuto; incorpora pechuga de pollo batidos y remueve hasta que "
                    "cuajen, unos 2 minutos.",
                    "Montaje: sirve.",
                    "🌱 Nota del Nutricionista AI: esta receta usa solo pechuga de pollo — NO botes pechuga de pollo: "
                    "guárdalas tapadas en la nevera."]}
    assert pc.huevo_sustituido(m) > 0
    assert m["recipe"][0] == ("Mise en place: enjuaga 35 g de bulgur; corta la pechuga de pollo en tiras; mide 1 cdta de "
                              "jugo de limón.")
    assert m["recipe"][1] == ("El Toque de Fuego: saltea el bulgur 1 minuto; incorpora la pechuga de pollo en tiras y "
                              "cocina 8-10 minutos, hasta que el pollo alcance 74 °C por dentro.")
    assert not any("NO botes" in p for p in m["recipe"])


def test_el_agua_de_enjuagar_vuelve_a_salir_clara_y_el_nombre_pierde_el_huevo():
    m = {"_protein_autofix_applied": "huevo->pollo", "name": "Yautía majada con pechuga de pollo bien cocido y ensalada",
         "ingredients": ["35 g de lentejas", "½ pechuga de pollo (≈91 g)"],
         "recipe": ["Mise en place: enjuaga las lentejas hasta que el agua salga pechuga de pollo.",
                    "El Toque de Fuego: cocina la pechuga de pollo a la plancha 7 minutos por lado hasta 74 °C.",
                    "Montaje: sirve."]}
    assert pc.huevo_sustituido(m) > 0
    assert m["recipe"][0] == "Mise en place: enjuaga las lentejas hasta que el agua salga clara."
    assert m["name"] == "Yautía majada con pechuga de pollo y ensalada"
    q = {"_protein_autofix_applied": "huevo->queso",
         "name": "Tostadas integrales con queso blanco fresco, aguacate, queso blanco cuajado y queso blanco",
         "ingredients": ["15 g de queso blanco"],
         "recipe": ["El Toque de Fuego: dora el queso blanco 3-4 min, hasta que estén firmes.", "Montaje: sirve."]}
    assert pc.huevo_sustituido(q) > 0
    assert q["name"] == "Tostadas integrales con queso blanco fresco, aguacate y queso blanco"


def test_sin_la_marca_del_tope_de_huevo_no_se_toca():
    pasos = ["El Toque de Fuego: hierve los huevos 10-12 minutos, pélalos y córtalos en mitades."]
    m = {"name": "Huevos duros", "ingredients": ["2 huevos"], "recipe": list(pasos)}
    assert pc.huevo_sustituido(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cerrador").huevo_sustituido(meal)  # [P1-PLAN-LOTE-426]' in src
