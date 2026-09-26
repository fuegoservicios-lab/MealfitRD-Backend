# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-378 · 2026-09-26] El wrap es el plato que se envuelve.

Batería REAL sobre el 376 (perfil del dueño, día 3): «Wok rápido de pollo y auyama con tortilla integral» recibía un
SEGUNDO Montaje —«rellena la casabe con lo que cierra (unos 150 g del relleno) y sirve el resto… un wrap que se puede
cerrar»—: la tortilla que ACOMPAÑA hacía wrap al plato y el soporte era el casabe que el cerrador añadió de guarnición.
El corpus traía además «rellena la almendras tostadas…» sobre «Sandía fresca con almendras tostadas»."""
from __future__ import annotations

import pathlib

import dish_structure as ds
import recipe_repair as rr

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_PASO_WRAP = ("Montaje: rellena la {pan} con lo que cierra (unos 150 g del relleno) y sirve el resto del relleno (~65 g) "
              "al lado, como ensalada: misma compra, un wrap que se puede cerrar.")


def test_la_tortilla_que_acompana_no_hace_wrap():
    assert ds.familia({"name": "Wok rápido de pollo y auyama con tortilla integral"}) == "otro"
    assert ds.familia({"name": "Revoltillo Criollo de Huevo con Tortilla Integral Dorada"}) == "tortilla_revuelto"
    assert ds.familia({"name": "Queso fresco guisado con repollo y tortilla integral"}) == "guiso"
    assert ds.familia({"name": "Bowl tropical de pollo con auyama asada, repollo al limón y tortilla integral"}) == "bowl"
    assert ds.familia({"name": "Crema suave de calabacín con tortilla de maíz"}) == "batido_crema"
    assert ds.familia({"name": "Sandía fresca con almendras tostadas, queso fresco y yogurt griego entero"}) == "otro"
    # la vasija: al empezar el nombre, tras «en» o tras «de»
    assert ds.familia({"name": "Tortilla integral rellena de pollo al limón"}) == "tostada_wrap"
    assert ds.familia({"name": "Lentejas suaves con espinacas en tortilla integral"}) == "tostada_wrap"
    assert ds.familia({"name": "Wrap de Tortilla Integral con Habichuelas Negras Guisadas y Aguacate"}) == "tostada_wrap"
    assert ds.familia({"name": "Wraps de Pavo Salteado al Wok con Repollo y Tortilla Integral"}) == "tostada_wrap"


def test_el_wok_del_dueno_no_se_acusa_ni_con_el_casabe_en_gramos():
    wok = {"name": "Wok rápido de pollo y auyama con tortilla integral",
           "ingredients": ["134 g de pechuga de pollo", "1 tortilla integral", "145 g de auyama en cubos pequeños",
                           "1 ají morrón en tiras", "45 g de casabe"],
           "recipe": ["El Toque de Fuego: saltea el pollo 5-6 minutos; añade la auyama y el ají.",
                      "🫓 Acompaña con el casabe de tus ingredientes: está listo para comer, sin cocción.",
                      "Montaje: sirve el wok de pollo y auyama con la tortilla integral caliente."]}
    assert ds.relaciones(wok) == []
    antes = list(wok["recipe"])
    assert rr.reparar_estructura(wok)["aplicado"] == [] and wok["recipe"] == antes


def test_el_soporte_es_el_pan_que_el_nombre_dice():
    m = {"name": "Tortilla integral rellena de pollo",
         "ingredients": ["1 tortilla integral", "45 g de casabe", "200 g de pechuga de pollo", "80 g de lechuga"],
         "recipe": ["Montaje: rellena la tortilla con el pollo y la lechuga."]}
    assert not any(r["tipo"] == "wrap_desproporcionado" for r in ds.relaciones(m))
    assert ds.componentes({"ingredients": ["15 g de almendras tostadas", "200 g de sandía"]})["soporte"] is None
    # sin nombrar pan, el wrap de verdad sigue acusado (lote 344)
    w = {"name": "Wrap de pollo con lechuga", "ingredients": ["1 tortilla de trigo (40 g)", "200 g de pechuga de pollo",
                                                              "100 g de lechuga"],
         "recipe": ["Montaje: rellena la tortilla con el pollo y la lechuga."]}
    assert any(r["tipo"] == "wrap_desproporcionado" for r in ds.relaciones(w))


def test_el_paso_que_ya_estaba_escrito_se_retira():
    casos = [
        ("Wok rápido de pollo y auyama con tortilla integral", "casabe"),
        ("Sandía fresca con almendras tostadas, queso fresco y yogurt griego entero", "almendras tostadas"),
        ("Tostadas integrales con yogurt natural, lechosa y queso cottage", "pan integral"),
        ("Tortilla integral rellena de pollo", "casabe"),
    ]
    for nombre, pan in casos:
        m = {"name": nombre, "ingredients": ["1 unidad de algo"],
             "recipe": ["Montaje: sirve.", "💡 Ajustamos ligeramente las porciones.", _PASO_WRAP.format(pan=pan)],
             "_display": {"en-US": {}}}
        assert rr.retirar_wrap_que_no_es(m) == 1, nombre
        assert m["recipe"] == ["Montaje: sirve.", "💡 Ajustamos ligeramente las porciones."] and "_display" not in m


def test_el_wrap_de_verdad_conserva_su_paso():
    m = {"name": "Wrap criollo de casabe con queso fresco, ensalada crujiente y edamame",
         "ingredients": ["1½ tortas pequeñas de casabe"],
         "recipe": ["Montaje: rellena el casabe.", _PASO_WRAP.format(pan="casabe")]}
    antes = list(m["recipe"])
    assert rr.retirar_wrap_que_no_es(m) == 0 and m["recipe"] == antes


def test_ancla():
    src = (_BACKEND / "recipe_repair.py").read_text(encoding="utf-8")
    assert 'out["retirado"] = retirar_wrap_que_no_es(meal)' in src
