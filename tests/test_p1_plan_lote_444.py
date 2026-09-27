# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-444 · 2026-09-26] El víver que ningún paso cuece: sin falsos «crudo», y su cocción donde toca.

Replay de la cola sobre 322 planes: el detector del lote 68 (`alimentos_sin_coccion`, el mismo recorrido de V7f) acusaba
19 víveres y 7 SÍ se cocían —bandeja al horno, microondas tapado, sartén que sigue, casabe de yuca, «guineítos»—; el
reparador les añadía un SEGUNDO paso de cocción. Y en 10 de los 17 platos con ese paso no había guiso: «Añade plátano
verde al guiso y cocínalo 15-20 minutos» en unos tostones, DESPUÉS del paso que ya los doraba y aplastaba."""
from __future__ import annotations

import json
import pathlib

import culinary_coherence as cc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _catalogo():
    return json.loads((_BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json")
                      .read_text(encoding="utf-8"))["catalogo_filas"]


_IDX = cc.build_culinary_index(_catalogo())


def _sin(ingredientes, pasos):
    return cc.alimentos_sin_coccion({"name": "Plato", "ingredients": ingredientes, "recipe": pasos}, _IDX)


# ─────────────────────────────── (1) lo que SÍ se cuece no es «sin cocción»

def test_la_bandeja_al_horno_cuece_lo_que_lleva():
    pasos = ["Mise en place: pela y corta ½ plátano maduro en rodajas; lava 70 g de espárragos.",
             "El Toque de Fuego: precalienta el horno a 200 °C. Coloca el pollo, el plátano maduro y los espárragos en "
             "una bandeja; distribuye el aceite de oliva, el ajo y la sal. Hornea 18-22 min, hasta que el pollo alcance "
             "74 °C en la parte más gruesa y los espárragos estén tiernos.",
             "Montaje: sirve el pollo con el plátano maduro asado y los espárragos."]
    assert _sin(["1 pechuga de pollo (≈177 g)", "½ plátano maduro", "70 g de espárragos"], pasos) == []


def test_la_bandeja_no_cuece_lo_que_se_cocina_aparte():
    pasos = ["El Toque de Fuego: coloca la yuca en una bandeja; cocina la pechuga de pollo 8 minutos por lado a la "
             "plancha, hasta 74 °C.",
             "Montaje: sirve la yuca con el pollo."]
    assert _sin(["200 g de yuca", "1 pechuga de pollo (≈177 g)"], pasos) == [("Yuca", "viver")]


def test_el_microondas_tapado_cuece_al_vapor():
    ings = ["250 g de auyama en cubos pequeños", "½ cebolla"]
    pasos = ["Mise en place: corta 250 g de auyama en cubos pequeños.",
             "El Toque de Fuego: coloca la auyama en un recipiente apto para microondas con una cucharada de agua, tapa y "
             "cocina a potencia alta durante 5-7 min. Calienta el aceite y cocina la cebolla 2 min.",
             "Montaje: sirve la auyama con la cebolla."]
    assert _sin(ings, pasos) == []
    pasos[1] = "El Toque de Fuego: calienta la auyama en el microondas 2 minutos. Cocina la cebolla 2 min."
    assert _sin(ings, pasos) == [("Auyama", "viver")], "destapada y 2 minutos no la cuece"


def test_el_fuego_del_recipiente_vale_para_la_clausula_siguiente():
    pasos = ["Mise en place: corta 250 g de auyama en cubos pequeños.",
             "El Toque de Fuego: coloca la auyama en un recipiente apto para microondas con 1 cda de agua; tápala y "
             "caliéntala durante 4-5 minutos, hasta que esté tierna. Cocina los huevos 3-4 minutos.",
             "Montaje: sirve el huevo con la auyama."]
    assert _sin(["250 g de auyama", "2 huevos"], pasos) == []


def test_lo_que_sigue_en_la_sarten_suma_minutos():
    ings = ["1 pechuga de pollo (≈189 g)", "250 g de auyama en cubos pequeños", "½ cebolla"]
    pasos = ["Mise en place: corta 250 g de auyama en cubos pequeños y 189 g de pechuga de pollo en tiras.",
             "El Toque de Fuego: calienta el aceite en una sartén a fuego medio-alto. Cocina la auyama y la cebolla 4-5 "
             "min, añade el ajo y las tiras de pollo sazonadas con sal y jugo de limón, y cocina 5-6 min; mide 74 °C en "
             "la parte más gruesa del pollo.",
             "Montaje: reparte el pollo y la auyama entre las tortillas."]
    assert _sin(ings, pasos) == []
    pasos[1] = "El Toque de Fuego: cocina la auyama 4 minutos, retírala y cocina el pollo 8 minutos hasta 74 °C."
    assert _sin(ings, pasos) == [("Auyama", "viver")], "fuera de la sartén, lo que sigue ya no la cuece"


def test_el_casabe_de_yuca_es_casabe():
    pasos = ["El Toque de Fuego: tuesta el casabe de yuca en una sartén seca 1-2 min por lado, hasta que esté crujiente.",
             "Montaje: unta la mantequilla de maní sobre el casabe de yuca."]
    assert _sin(["1 pieza de casabe de yuca", "1¾ cdtas de mantequilla de maní"], pasos) == []


def test_los_guineitos_son_el_guineo_de_la_lista():
    pasos = ["Mise en place: pela y corta 55 g de guineo verde en rodajas; pica ½ cebolla.",
             "El Toque de Fuego: hierve los guineítos verdes en agua con sal 15-18 min hasta que un cuchillo entre sin "
             "fuerza; escúrrelos. En una sartén con el aceite sofríe la cebolla 2 min, añade los guineítos y guisa 3-4 "
             "min.",
             "Montaje: sirve los guineítos guisados."]
    assert _sin(["55 g de guineo verde pelado", "½ cebolla"], pasos) == []


def test_lo_que_de_verdad_no_se_cuece_se_sigue_viendo():
    pasos = ["Mise en place: mide ½ plátano verde (120 g) ya cocido.",
             "El Toque de Fuego: calienta el plátano en el microondas y májalo con el jugo de limón.",
             "Montaje: sirve el mangú."]
    assert _sin(["½ plátano verde", "½ limón"], pasos) == [("Plátano verde", "viver")]
    dora = ["El Toque de Fuego: dora la auyama 3 minutos en la sartén con la cebolla.", "Montaje: sirve."]
    assert _sin(["250 g de auyama", "½ cebolla"], dora) == [("Auyama", "viver")]


def test_v7f_ve_lo_mismo_que_el_reparador():
    plan = {"days": [{"day": 1, "meals": [{"name": "Casabe con maní", "ingredients": ["1 pieza de casabe de yuca"],
                                           "recipe": ["El Toque de Fuego: tuesta el casabe de yuca 1-2 min por lado.",
                                                      "Montaje: sirve."]}]}]}
    assert [v for v in (cc.culinary_contract_scan(plan, _catalogo()) or []) if v["check"] == "V7f"] == []


# ─────────────────────────────── (2) el paso que lo cuece: al guiso si lo hay; si no, antes del fuego

def _tostones():
    """El plato real (bowl de 322 planes, «owner_like»): el Toque dora las rodajas 2 min por lado y las aplasta."""
    return {"meal": "Almuerzo", "name": "Bowl tropical de pollo y tostones rápidos",
            "ingredients": ["1 pechuga de pollo (≈200 g)", "½ plátano verde mediano", "90 g de maíz dulce en granos",
                            "185 g de coliflor en floretes pequeños", "¼ cebolla picada"],
            "recipe": ["Mise en place: corta la pechuga de pollo en tiras finas; pela y corta ½ plátano verde en "
                       "rodajas, separa 185 g de coliflor en floretes pequeños y pica ¼ de cebolla.",
                       "El Toque de Fuego: calienta una sartén amplia a fuego medio-alto con el aceite de oliva; cocina "
                       "la coliflor, la cebolla y el ajo durante 4 minutos. Añade el pollo y el maíz dulce y cocina 2 "
                       "minutos, hasta que el pollo alcance 74 °C en la parte más gruesa. En otra sartén caliente, dora "
                       "las rodajas de plátano verde 2 minutos por lado y aplástalas para formar tostones.",
                       "Montaje: coloca los tostones en el bowl y añade la mezcla de pollo, coliflor y maíz."]}


def test_sin_guiso_la_coccion_previa_va_antes_del_fuego_y_hierve_lo_que_el_mise_corto():
    import coccion_viver as cv
    m = _tostones()
    paso, previa = cv.paso_viver_sin_coccion(m, "Plátano verde", False, False)
    assert previa and paso == ("💡 Cocción previa: hierve las rodajas de plátano verde en agua 10-12 minutos, hasta "
                               "que estén tiernas; escúrrelas."), paso
    rec = cv.tras_la_mise(m["recipe"], paso)
    assert rec[0].startswith("Mise en place") and rec[1] == paso and rec[2].startswith("El Toque de Fuego")
    entero = {"recipe": ["Mise en place: mide ½ plátano verde.", "El Toque de Fuego: májalo.", "Montaje: sirve."]}
    assert cv.paso_viver_sin_coccion(entero, "Plátano verde", False, False)[0] == (
        "💡 Cocción previa: hierve el plátano verde pelado en agua 20-25 min, hasta que el cuchillo entre sin fuerza.")


def test_en_un_guiso_va_al_guiso_con_su_genero():
    import coccion_viver as cv
    paso, previa = cv.paso_viver_sin_coccion({"name": "Habichuelas guisadas"}, "Auyama", False, True)
    assert not previa and paso == ("🍠 Añade la auyama al guiso y cocínala 15-20 minutos, hasta que esté tierna por "
                                   "dentro, antes de servir.")
    # con «Nada» de tiempo, jamás 15-20 minutos: cubos pequeños antes del fuego (lote 220)
    assert cv.paso_viver_sin_coccion({"name": "Habichuelas guisadas"}, "Auyama", True, True)[1] is True


def test_sin_tiempo_cubos_pequenos_antes_del_fuego():
    import coccion_viver as cv
    paso, previa = cv.paso_viver_sin_coccion({"name": "Mangú"}, "Yautía", True, False)
    assert previa and paso == ("💡 Cocción previa: corta la yautía en cubos pequeños (1 cm) y hiérvelos 10-12 minutos, "
                               "hasta que estén tiernos; escúrrelos.")
    assert cv.paso_viver_sin_coccion({"name": "Mangú"}, "Yuca", True, False)[0].endswith(
        "escúrrelos y desecha el agua de cocción (cruda no se come).")
    paso, _ = cv.paso_viver_sin_coccion({"name": "Mangú"}, "Plátano maduro", False, False)
    assert "hierve el plátano maduro pelado en agua 10-15 min" in paso


def test_el_texto_del_408_es_el_mismo():
    import pasos_cantidades as pq
    assert pq.nota_hervor_viver("yuca") == ("💡 Cocción previa: hierve la yuca pelada en agua 20-25 min, hasta que el "
                                            "cuchillo entre sin fuerza, y desecha el agua de cocción (cruda no se come).")
    assert pq.nota_hervor_viver("guineito").startswith("💡 Cocción previa: hierve los guineítos pelados en agua 15-20 min")


def test_el_reparador_del_68_pone_la_coccion_previa_en_los_tostones():
    import graph_orchestrator as go
    plan = {"days": [{"day": 1, "meals": [_tostones()]}]}
    assert go._auto_patch_uncooked_foods(plan, _catalogo(), form_data={"cookingTime": "30min"}) == 1
    rec = plan["days"][0]["meals"][0]["recipe"]
    assert rec[1].startswith("💡 Cocción previa: hierve las rodajas de plátano verde"), rec
    assert not any("al guiso" in str(s) for s in rec), rec
    assert go._auto_patch_uncooked_foods(plan, _catalogo(), form_data={"cookingTime": "30min"}) == 0


def test_anclas():
    src_cc = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-444" in src_cc and "def _v7f_candidatos(" in src_cc
    assert src_cc.count("_v7f_candidatos(meal, index)") == 2, "V7f y el reparador recorren los MISMOS candidatos"
    assert "tooltip-anchor: P1-PLAN-LOTE-444" in (_BACKEND / "coccion_viver.py").read_text(encoding="utf-8")
    src_go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "_cv.paso_viver_sin_coccion(_m, food, _sin_tiempo, _stewy)" in src_go
