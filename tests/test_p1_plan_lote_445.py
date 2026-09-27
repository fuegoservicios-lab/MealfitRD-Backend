# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-445 · 2026-09-27] La proteína que la lista compra CRUDA y ningún paso cuece.

Replay de la cola sobre 322 planes (proteína animal de la lista, nombrada en un paso, que ninguna cláusula cuece): pollo
«ya cocido» que sólo se calienta 2 minutos, «desmenuza 1 filete de pescado» servido en una «mezcla fresca», huevos
«cocidos listos para consumir» que nadie hierve —en perfiles de lactancia y embarazo—. Los reparadores 407/409/441 existían
y callaban: cinco preguntaban «¿ya tiene su cocción previa?» mirando el texto ENTERO desde la primera nota (la del arroz
contaba), y los verbos no incluían «prepara», «corta», «calienta», «sirve» ni el imperativo «desmenuza». Y V7f no veía
dos cocciones reales («Cocínalo… hasta 63 °C» con el pronombre de dos cláusulas atrás; «Añade el pavo…; cocina 8-10 min»)."""
from __future__ import annotations

import json
import pathlib

import culinary_coherence as cc
import pasos_cantidades as pq

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_ARROZ = "💡 Cocción previa: enjuaga el arroz blanco crudo y cuécelo en agua 15-20 min hasta que esté tierno."


def _catalogo():
    return json.loads((_BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json")
                      .read_text(encoding="utf-8"))["catalogo_filas"]


_IDX = cc.build_culinary_index(_catalogo())


# ─────────────────────────────── (1) la cocción previa ya puesta se busca en la NOTA

def test_la_nota_del_arroz_no_es_la_del_pollo():
    m = {"ingredients": ["½ pechuga de pollo (≈114 g)", "¼ taza de arroz blanco crudo"],
         "recipe": ["Mise en place: corta ½ pechuga de pollo (≈114 g) en tiras y mide el arroz.", _ARROZ,
                    "El Toque de Fuego: saltea la cebolla 3 minutos; añade el pollo cocido y calienta 2 minutos.",
                    "Montaje: sirve el pollo con el arroz."]}
    assert pq.proteina_cocida_de_la_lista(m) == 1
    assert m["recipe"][1].startswith("💡 Cocción previa: cocina la pechuga de pollo en agua con sal"), m["recipe"]
    assert pq.proteina_cocida_de_la_lista(m) == 0, "idempotente: ahora sí la encuentra en SU nota"


def test_desmenuza_ya_cocida_con_habichuelas_delante():
    m = {"ingredients": ["1½ pechugas de pollo (≈300 g)", "40 g de habichuelas rojas secas"],
         "recipe": ["Mise en place: Desmenuza 300 g de pechuga de pollo ya cocida (de la tanda de la semana).",
                    "💡 Cocción previa: remoja las habichuelas rojas secas 8-12 h y hiérvelas 60-90 min.",
                    "El Toque de Fuego: saltea el repollo 3-4 minutos y añade el pollo.", "Montaje: arma el wrap."]}
    assert pq.proteina_cocida_de_la_lista(m) == 1
    assert any("cocina la pechuga de pollo" in p for p in m["recipe"])


# ─────────────────────────────── (2) los verbos que faltaban

def test_prepara_la_pechuga_cocida():
    m = {"ingredients": ["115 g de pechuga de pollo"],
         "recipe": ["Mise en place: prepara 115 g de pechuga de pollo cocida y desmenuzada.",
                    "El Toque de Fuego: añade la pechuga de pollo y el orégano; calienta 2 minutos.", "Montaje: sirve."]}
    assert pq.proteina_cocida_de_la_lista(m) == 1


def test_prepara_los_huevos_cocidos():
    m = {"ingredients": ["35 g de avena", "3 huevos", "2 claras de huevo"],
         "recipe": ["Mise en place: mide 35 g de avena y prepara 3 huevos y 2 claras de huevo cocido listo para consumir.",
                    "El Toque de Fuego: calienta la avena con la leche en el microondas 2-3 minutos.",
                    "Montaje: añade el guineo a la avena y acompáñala con el huevo cocido."]}
    assert pq.huevo_duro_de_la_lista(m) == 1
    assert m["recipe"][1].startswith("💡 Cocción previa: hierve los huevos 10-12 min"), m["recipe"]


def test_el_cocido_es_del_huevo_y_un_huevo_cuajado_no_se_hierve():
    """«sirve los huevos con la yuca cocida»: el «cocida» es de la yuca; y unos huevos que el paso cuaja ya están hechos
    (plan de emergencia bariátrico «Huevos Revueltos con Yuca», que recibía «hierve los huevos 10-12 min»)."""
    m = {"ingredients": ["2 unidades huevo", "100g yuca cocida"],
         "recipe": ["Mise en place: pesa los ingredientes.",
                    "El Toque de Fuego: bate los huevos y cuájalos en una sartén 3-4 minutos, hasta que estén firmes.",
                    "Montaje: sirve los huevos con la yuca cocida."]}
    assert pq.huevo_duro_de_la_lista(m) == 0, m["recipe"]
    m2 = {"ingredients": ["3 huevos"], "recipe": ["Mise en place: pesa.", "El Toque de Fuego: tuesta el pan 2 min.",
                                                   "Montaje: sirve el pan con los huevos enteros bien cocidos."]}
    assert pq.huevo_duro_de_la_lista(m2) == 1


def test_desmenuza_el_pescado_crudo():
    m = {"ingredients": ["1 filete de pescado (≈250 g)", "75 g de espinacas"],
         "recipe": ["Mise en place: desmenuza 1 filete de pescado (250 g); pica 75 g de espinacas.",
                    "El Toque de Fuego: mezcla filete de pescado blanco con espinacas y el jugo de limón.",
                    "Montaje: sirve la mezcla fresca."]}
    assert pq.ave_desmenuzada_cruda(m) == 1
    assert "(63 °C al centro)" in m["recipe"][1], m["recipe"]


def test_si_un_paso_lo_cuece_no_hay_nota():
    m = {"ingredients": ["1 filete de pescado (≈250 g)"],
         "recipe": ["Mise en place: desmenuza 1 filete de pescado (250 g).",
                    "El Toque de Fuego: cocina el filete de pescado a la plancha 3-4 min por lado.", "Montaje: sirve."]}
    assert pq.ave_desmenuzada_cruda(m) == 0


def test_lo_que_un_paso_cocina_con_su_punto_no_lleva_coccion_previa():
    """Replay: 8 de 25 notas nuevas eran a proteínas que el paso siguiente ya cocina hasta su punto."""
    dora = {"ingredients": ["½ pechuga de pollo (≈120 g)"],
            "recipe": ["Mise en place: Corta la pechuga de pollo cocida en tiras.",
                       "El Toque de Fuego: Añade el ajo, la cebolla y el pollo; dóralos 3-4 min con el jugo de limón, "
                       "hasta que el pollo alcance 74 °C en la parte más gruesa.", "Montaje: sirve."]}
    assert pq.proteina_cocida_de_la_lista(dora) == 0, dora["recipe"]
    horno = {"ingredients": ["¾ pechuga de pollo (≈150 g)"],
             "recipe": ["Mise en place: corta 150 g de pechuga de pollo en cubos pequeños.",
                        "El Toque de Fuego: coloca la pechuga de pollo con la cebolla en una fuente; hornea durante 18-22 "
                        "min y verifica con termómetro que el centro de la pieza más gruesa alcance 74 °C. Desmenuza el "
                        "pollo e intégralo con la harina.", "Montaje: sirve."]}
    assert pq.ave_desmenuzada_cruda(horno) == 0, horno["recipe"]
    salsa = {"ingredients": ["1¼ filetes de pescado (≈185 g)"],
             "recipe": ["Mise en place: corta 185 g de filete de pescado blanco en piezas.",
                        "El Toque de Fuego: añade el tomate y el pescado con sal y cocina tapado a fuego medio 8-10 min; "
                        "desmenuza el pescado dentro de la salsa.", "Montaje: sirve."]}
    assert pq.ave_desmenuzada_cruda(salsa) == 0, salsa["recipe"]
    tortitas = {"ingredients": ["225 g de filete de pescado blanco"],
                "recipe": ["Mise en place: desmenuza 225 g de filete de pescado blanco.",
                           "El Toque de Fuego: mezcla el pescado blanco con el limón; forma tortitas compactas. Colócalas "
                           "en una bandeja y hornéalas a 200 °C 12-15 min, hasta que el centro esté completamente cocido.",
                           "Montaje: sirve."]}
    assert pq.ave_desmenuzada_cruda(tortitas) == 0, tortitas["recipe"]
    huevo = {"ingredients": ["1 huevo", "20 g de avena"],
             "recipe": ["Mise en place: lava 1 huevo.",
                        "El Toque de Fuego: coloca el huevo en agua y hiérvelo a fuego medio durante 9-10 min.",
                        "Montaje: sirve la avena y acompaña con el huevo duro."]}
    assert pq.huevo_duro_de_la_lista(huevo) == 0, huevo["recipe"]
    nocturno = {"ingredients": ["1 huevo"],     # plan bariátrico real: ya en producción recibía la nota
                "recipe": ["Mise en place: saca 1 huevo de la nevera.",
                           "El Toque de Fuego: coloca el huevo en una olla pequeña con agua fría que lo cubra, lleva a hervor "
                           "a fuego medio-alto y cocina 10 minutos exactos desde que empieza a hervir.",
                           "Montaje: corta el huevo duro en mitades."]}
    assert pq.huevo_duro_de_la_lista(nocturno) == 0, nocturno["recipe"]
    meta = {"ingredients": ["2½ pechugas de pollo (≈425 g)"],
            "recipe": ["Mise en place: corta 300 g de pechuga de pollo cocida en porciones.",
                       "El Toque de Fuego: agrega las espinacas y calienta la pechuga de pollo cocida hasta que el pollo "
                       "alcance 74 °C por dentro.", "Montaje: sirve."]}
    assert pq.proteina_cocida_de_la_lista(meta) == 0, "calentar HASTA 74 °C es cocinarla"
    guiso = {"ingredients": ["¾ taza de lentejas secas", "95 g de auyama"],
             "recipe": ["Mise en place: corta 95 g de auyama en cubos.",
                        "El Toque de Fuego: sofríe la cebolla 3 minutos; agrega la auyama, las lentejas cocidas y agua. "
                        "Cocina 15-20 minutos a fuego medio hasta que la auyama esté tierna.", "Montaje: sirve."]}
    assert pq.viver_cocido_de_la_lista(guiso) == 0, "el «cocidas» es de las lentejas, y el guiso ya cuece la auyama"
    # «calienta 2 minutos (74 °C si no estaba previamente cocido)» NO cocina: el pollo crudo sigue recibiendo su nota
    quesadilla = {"ingredients": ["¼ pechuga de pollo (≈82 g)"],
                  "recipe": ["Mise en place: desmenuza ¼ pechuga de pollo (≈82 g).",
                             "El Toque de Fuego: saltea el puerro 2-3 minutos; incorpora el pollo y calienta 2 minutos "
                             "(74 °C si no estaba previamente cocido).", "Montaje: sirve."]}
    assert pq.ave_desmenuzada_cruda(quesadilla) == 1


# ─────────────────────────────── (3) V7f: dos cocciones reales que no veía

def _sin(ingredientes, pasos):
    return cc.alimentos_sin_coccion({"name": "Plato", "ingredients": ingredientes, "recipe": pasos}, _IDX)


def test_el_punto_de_su_clase_cuece_lo_del_pronombre():
    pasos = ["Mise en place: corta el pescado en porciones y seca bien la superficie.",
             "El Toque de Fuego: sazona el pescado con ajo en polvo, orégano y sal; cúbrelo con el pan rallado presionando "
             "y rocíalo con la mitad del aceite. Cocínalo en el airfryer a 200 °C durante 10-14 minutos, volteándolo a "
             "mitad del tiempo, hasta que el centro alcance 63 °C.",
             "Montaje: sirve el filete con el arroz."]
    assert _sin(["1 filete de pescado (≈160 g)", "30 g de pan rallado"], pasos) == []
    # 74 °C es el punto del ave, no del pescado: no lo da por cocido
    pasos[1] = pasos[1].replace("63 °C", "74 °C")
    assert _sin(["1 filete de pescado (≈160 g)", "30 g de pan rallado"], pasos) != []


def test_lo_que_se_anade_a_la_sarten_al_fuego_se_cuece_con_ella():
    pasos = ["Mise en place: corta 140 g de pechuga de pavo en trozos y 35 g de cundeamor en rodajas.",
             "El Toque de Fuego: calienta 1 cdta de aceite de oliva en una sartén; cocina la cebolla y el ajo a fuego "
             "medio 2 min. Añade pechuga de pavo, el cundeamor, el maíz y el orégano; cocina 8-10 min, removiendo hasta "
             "que el cundeamor esté tierno.",
             "Montaje: sirve caliente."]
    ings = ["140 g de pechuga de pavo", "35 g de cundeamor", "270 g de maíz dulce en granos", "½ cebolla"]
    assert _sin(ings, pasos) == []
    # sin fuego antes en el paso, «añade» no une nada
    frio = ["Mise en place: corta 140 g de pechuga de pavo en trozos.",
            "Montaje: añade pechuga de pavo, el cundeamor y el maíz; sirve 8-10 min después."]
    assert _sin(ings, frio) == [("Pechuga de pavo", "proteina")]


def test_anclas():
    src_cc = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-445" in src_cc and "def _v7f_punto(" in src_cc
    src_pq = (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
    assert src_pq.count("_en_coccion_previa(rec,") == 6, "los cinco reparadores preguntan a la NOTA"
    assert '.find("coccion previa"):]:' not in src_pq
