# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-540 · 2026-09-27] La verdura de la lista que ningún paso cocina recibe su cocción.

Batería real del 27-sep (DM2 con insulina, celíaco), leída entera: vainitas cortadas que el guiso nunca recibe, brócoli
«separado» en unas lentejas guisadas, espinacas lavadas en un pollo guisado que nunca las lleva.
"""
from __future__ import annotations

import pathlib
import sys

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import verdura_sin_coccion as vsc  # noqa: E402


def _quinoa_guisada():
    return {"name": "Quinoa guisada ligera con berenjena, vainitas, pimentón y camarones",
            "ingredients": ["40 g de quinoa seca", "40 g de berenjena", "120 g de vainitas", "½ pimentón", "½ cebolla",
                            "¾ cdta de aceite de oliva", "70 g de camarones cocidos"],
            "recipe": ["Mise en place: enjuaga 40 g de quinoa; corta 40 g de berenjena y 120 g de vainitas; pica ½ pimentón "
                       "y ½ cebolla.",
                       "El Toque de Fuego: cocina la quinoa en agua a fuego medio-bajo durante 15-18 min, hasta que esté "
                       "tierna. En una sartén, calienta ¾ cdta de aceite de oliva y cocina la cebolla y el pimentón 3 min. "
                       "Añade berenjena al guiso y cocínala a fuego medio 8-10 minutos, hasta que esté tierna.",
                       "Montaje: sirve la quinoa guisada en un plato hondo. Acompaña con camarones."]}


def test_las_vainitas_que_el_guiso_nunca_recibe():
    m = _quinoa_guisada()
    assert vsc.cocer(m) == 1
    assert m["recipe"][1] == ("💡 Cocción previa: hierve las vainitas 6-8 minutos en agua con sal, hasta que estén "
                              "tiernas, y escúrrelas."), m["recipe"]
    assert vsc.cocer(m) == 0                                   # idempotente; la berenjena ya se cocinaba


def test_brocoli_separado_en_las_lentejas_guisadas():
    m = {"name": "Lentejas guisadas con plátano verde, vegetales y sardinas en lata",
         "ingredients": ["½ taza de lentejas secas", "½ taza de brócoli", "½ cebolla", "¾ cdta de aceite de oliva"],
         "recipe": ["Mise en place: pica ½ cebolla; separa ½ taza de brócoli y mide 280 g de lentejas cocidas.",
                    "💡 Cocción previa: enjuaga las lentejas secas y hiérvelas 20-25 min hasta que estén tiernas.",
                    "El Toque de Fuego: en una olla, calienta el aceite de oliva y cocina la cebolla 5 minutos. Calienta "
                    "las lentejas cocidas 2-3 minutos.",
                    "Montaje: sirve el guiso de lentejas."]}
    assert vsc.cocer(m) == 1
    assert m["recipe"][2].startswith("💡 Cocción previa: hierve el brócoli"), m["recipe"]


def test_espinacas_lavadas_en_el_pollo_guisado():
    m = {"name": "Pollo guisado en salsa natural de vegetales con majado de yautía",
         "ingredients": ["1¼ pechugas de pollo (≈230 g)", "½ tomate", "½ taza de espinacas", "½ cucharada de aceite de oliva"],
         "recipe": ["Mise en place: corta el pollo en trozos; pica ½ tomate; lava ½ taza de espinacas.",
                    "El Toque de Fuego: calienta el aceite en una sartén y cocina el tomate 4 minutos; agrega el pollo y "
                    "guísalo en esa salsa 15 minutos, hasta 74 °C.",
                    "Montaje: sirve el pollo con su salsa."]}
    assert vsc.cocer(m) == 1
    assert m["recipe"][2].startswith("🥬 Añade las espinacas"), m["recipe"]


def test_lo_que_ya_se_cocina_o_va_crudo_no_se_toca():
    m = {"name": "Bulgur meloso con vainitas y repollo al ajo",
         "ingredients": ["100 g de bulgur seco", "2½ tazas de vainitas", "1 taza de repollo"],
         "recipe": ["Mise en place: lava y corta las vainitas y el repollo.",
                    "El Toque de Fuego: calienta el aceite en una olla. Añade las vainitas, el repollo y el tomillo; "
                    "cocina 4-5 min.",
                    "Montaje: sirve el bulgur meloso."]}
    antes = list(m["recipe"])
    assert vsc.cocer(m) == 0 and m["recipe"] == antes
    m = {"name": "Ensalada crujiente de brócoli y atún",
         "ingredients": ["1 taza de brócoli", "130 g de atún en agua"],
         "recipe": ["Mise en place: corta el brócoli en arbolitos.", "El Toque de Fuego: calienta el atún 1 minuto.",
                    "Montaje: mezcla el brócoli crudo con el atún."]}
    assert vsc.cocer(m) == 0
    m = {"name": "Bowl de espinacas y huevo", "ingredients": ["2 tazas de espinacas", "2 huevos"],
         "recipe": ["Mise en place: lava las espinacas.", "El Toque de Fuego: hierve los huevos 10-12 minutos.",
                    "Montaje: sirve las espinacas con los huevos."]}
    assert vsc.cocer(m) == 0                                   # sin guiso ni salsa: la espinaca va cruda


def test_la_linea_huerfana_no_es_una_coccion_que_falta():
    # corpus (perfil con hipotiroidismo): «75g de espinacas» en cada comida, que ningún paso nombra, y un desayuno que sólo
    # tuesta el pan «en una sartén seca» — antes de la guarda recibía «Añade las espinacas a la sartén o al guiso…»
    m = {"name": "Tostadas integrales con yogurt, guineo y mantequilla de maní",
         "ingredients": ["1 rebanada de pan integral familiar", "¾ taza de yogurt natural entero", "75g de espinacas"],
         "recipe": ["Mise en place: mide el yogurt y corta el guineo.",
                    "El Toque de Fuego: tuesta el pan integral en una sartén seca a fuego medio durante 2-3 minutos por lado.",
                    "Montaje: unta la mantequilla de maní y sirve con el yogurt."]}
    antes = list(m["recipe"])
    assert vsc.cocer(m) == 0 and m["recipe"] == antes
    m = {"name": "Casabe con queso blanco y vegetales asados", "ingredients": ["½ taza de brócoli", "1 torta de casabe"],
         "recipe": ["Mise en place: mide el casabe.", "El Toque de Fuego: calienta el casabe 1-2 min por lado.",
                    "Montaje: sirve el casabe."]}
    assert vsc.cocer(m) == 0


def test_replay_del_corpus_asado_en_bandeja_y_espinacas_frescas():
    m = {"name": "Casabe con queso blanco, vegetales asados al limón y edamame",
         "ingredients": ["½ calabacín mediano", "½ taza de brócoli", "½ cebolla", "1 cdta de aceite de oliva"],
         "recipe": ["Mise en place: corta el calabacín, divide el brócoli en floretes y pica la cebolla.",
                    "El Toque de Fuego: mezcla el calabacín, el brócoli y la cebolla con el aceite. Asa en una bandeja a "
                    "220 °C durante 15-18 minutos, hasta que los vegetales estén tiernos.",
                    "Montaje: sirve los vegetales asados."]}
    assert vsc.cocer(m) == 0
    m = {"name": "Casabe con guiso de habichuelas rojas, calabacín y queso fresco",
         "ingredients": ["½ taza de habichuelas rojas cocidas", "30 g de calabacín", "15 g de espinacas"],
         "recipe": ["Mise en place: corta el calabacín.",
                    "El Toque de Fuego: sofríe la cebolla 2 minutos; añade el calabacín y guisa 12-15 minutos.",
                    "Montaje: sirve el guiso. Acompaña con las espinacas frescas."]}
    assert vsc.cocer(m) == 0


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("verdura_sin_coccion").cocer(meal)  # [P1-PLAN-LOTE-540]' in src
