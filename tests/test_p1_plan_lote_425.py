# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-425 · 2026-09-26] La legumbre del cerrador se prepara según lo que compra la lista, y una sola vez.

Batería REAL sobre el 424 (dm2 con insulina): «Cocina edamame en agua hasta que ablanden e incorpóralo al plato» con
«215 g de edamame cocido» en la lista y «Acompaña con edamame» en el montaje; «Añade berenjena en agua hasta que ablanden
e incorpóralo al plato»; «Incorpora lentejas secas al guiso y cocínalos… hasta que estén cocidos por dentro» con el 💡 que
ya las hirvió. Corpus de 322 planes: 214 comidas con la frase del cerrador, las 214 con su «Acompaña con»."""
from __future__ import annotations

import pathlib

import pasos_cerrador as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_edamame_cocido_se_calienta_y_se_sirve_al_lado():
    m = {"ingredients": ["½ plátano verde", "40 g de queso blanco fresco", "215 g de edamame cocido"],
         "recipe": ["Mise en place: lava el plátano verde.",
                    "El Toque de Fuego: hornea el plátano 18-20 min. Cocina edamame en agua hasta que ablanden e "
                    "incorpóralo al plato.",
                    "Montaje: sirve las canoas calientes. Acompaña la cena con agua. Acompaña con edamame."]}
    assert pc.legumbre_del_cerrador(m) == 1
    assert m["recipe"][1] == ("El Toque de Fuego: hornea el plátano 18-20 min. Calienta el edamame cocido en agua "
                              "hirviendo 2-3 minutos y escúrrelo.")
    assert m["recipe"][2].endswith("Acompaña con edamame.")
    assert pc.legumbre_del_cerrador(m) == 0


def test_la_verdura_que_quedo_se_cocina_y_el_edamame_crudo_del_montaje_tambien():
    m = {"ingredients": ["30 g de quinoa", "30 g de berenjena", "½ cebolla", "170 g de edamame"],
         "recipe": ["Mise en place: enjuaga 30 g de quinoa; corta 30 g de berenjena en cubos.",
                    "💡 Cocción previa: enjuaga la quinoa cruda y cuécela en agua 12-15 min hasta que esté tierna.",
                    "El Toque de Fuego: calienta el aceite en una olla a fuego medio y cocina la cebolla 2 minutos. "
                    "Añade berenjena en agua hasta que ablanden e incorpóralo al plato.",
                    "Montaje: sirve la quinoa con los vegetales. Acompaña con agua. Acompaña con edamame."]}
    assert pc.legumbre_del_cerrador(m) == 2
    assert m["recipe"][2] == ("El Toque de Fuego: calienta el aceite en una olla a fuego medio y cocina la cebolla 2 "
                              "minutos. Añade la berenjena y cocina 5-6 minutos, hasta que esté tierna. Cocina el edamame "
                              "en agua hirviendo 4-5 minutos y escúrrelo.")


def test_las_lentejas_que_hirvio_el_consejo_entran_cocidas_al_guiso():
    m = {"ingredients": ["20 g de lentejas secas", "½ tomate mediano"],
         "recipe": ["Mise en place: mide 55 g de lentejas cocidas.",
                    "💡 Cocción previa: enjuaga las lentejas secas y hiérvelas 20-25 min hasta que estén tiernas.",
                    "El Toque de Fuego: añade el tomate y cocina 4 min. Incorpora lentejas secas al guiso y cocínalos a "
                    "fuego medio 12-15 minutos, hasta que estén cocidos por dentro; incorpóralos con cuidado para no "
                    "deshacer el resto.",
                    "Montaje: sirve el guiso tibio."]}
    assert pc.legumbre_del_cerrador(m) == 1
    assert m["recipe"][2] == ("El Toque de Fuego: añade el tomate y cocina 4 min. Incorpora las lentejas cocidas al guiso "
                              "y cocínalas a fuego medio 5 minutos para que tomen el sabor; remuévelas con cuidado para "
                              "no deshacer el resto.")


def test_la_soya_se_hidrata_y_lo_que_otro_paso_cocina_sobra():
    m = {"ingredients": ["85 g de soya texturizada", "1 taza de quinoa"],
         "recipe": ["El Toque de Fuego: cocina la quinoa 12 min. Cocina soya texturizada en agua hasta que ablanden e "
                    "incorpórala al plato.",
                    "Montaje: sirve. Acompaña con soya texturizada."]}
    assert pc.legumbre_del_cerrador(m) == 1
    assert m["recipe"][0].endswith("Hidrata la soya texturizada en agua caliente 10 minutos y escúrrela bien.")
    q = {"ingredients": ["½ taza de quinoa seca", "100 g de edamame cocido"],
         "recipe": ["El Toque de Fuego: cocina la quinoa en agua 12-15 minutos. Incorpora quinoa seca en agua hasta que "
                    "ablanden e incorpóralo al plato.",
                    "Montaje: sirve la quinoa."]}
    assert pc.legumbre_del_cerrador(q) == 1
    assert q["recipe"][0] == "El Toque de Fuego: cocina la quinoa en agua 12-15 minutos."


def test_lo_que_el_cerrador_mete_en_la_preparacion_se_sirve_una_sola_vez():
    """85 «Escurre e incorpora X (ya viene cocido) a la preparación», 46 «Agrega X a la licuadora», 15 «Incorpora X a la
    preparación y mézclalo» y 17 «Sirve X al lado» con su «Acompaña con X» en el montaje (corpus de 322 planes)."""
    atun = {"ingredients": ["85 g de atún en agua"],
            "recipe": ["El Toque de Fuego: revuelve el huevo 3 minutos. Escurre e incorpora atún en agua (ya viene cocido) "
                       "a la preparación antes de servir.",
                       "Montaje: sirve el mangú. Acompaña con atún en agua."]}
    assert pc.legumbre_del_cerrador(atun) == 1
    assert atun["recipe"][0].endswith("Escurre el atún en agua (ya viene cocido).")
    batido = {"ingredients": ["40 g de queso cottage", "1 guineo"],
              "recipe": ["El Toque de Fuego: licúa el guineo con el agua. 💪 Agrega queso cottage a la licuadora y licúa "
                         "hasta integrar.",
                         "Montaje: sirve frío. Acompaña con queso cottage."]}
    assert pc.legumbre_del_cerrador(batido) == 1
    assert batido["recipe"][1] == "Montaje: sirve frío."
    yogur = {"ingredients": ["¾ taza de yogurt griego entero", "1 torta de casabe"],
             "recipe": ["El Toque de Fuego: tuesta el casabe 1-2 minutos. Incorpora yogurt griego entero a la preparación "
                        "y mézclalo antes de servir.",
                        "Montaje: coloca el queso sobre el casabe. Acompaña con yogurt griego entero."]}
    assert pc.legumbre_del_cerrador(yogur) == 1
    assert yogur["recipe"][0] == "El Toque de Fuego: tuesta el casabe 1-2 minutos."
    bol = {"ingredients": ["67 g de yogurt griego entero", "1 torta pequeña de casabe"],
           "recipe": ["El Toque de Fuego: tuesta el casabe 2-3 min. Sirve yogurt griego entero al lado para acompañar.",
                      "Montaje: arma las tostadas y acompaña con el bol de yogurt griego con almendras."]}
    assert pc.legumbre_del_cerrador(bol) == 1 and bol["recipe"][0] == "El Toque de Fuego: tuesta el casabe 2-3 min."


def test_sin_la_frase_no_se_toca():
    m = {"ingredients": ["100 g de edamame cocido"],
         "recipe": ["El Toque de Fuego: saltea el edamame 3 minutos.", "Montaje: sirve. Acompaña con edamame."]}
    antes = [list(m["recipe"]), list(m["ingredients"])]
    assert pc.legumbre_del_cerrador(m) == 0 and [m["recipe"], m["ingredients"]] == antes


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cerrador").legumbre_del_cerrador(meal)  # [P1-PLAN-LOTE-425]' in src
