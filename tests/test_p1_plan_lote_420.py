# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-420 · 2026-09-26] El pan de un «Wrap integral» es la tortilla, no el casabe del cerrador.

Batería REAL sobre el 409 (perfil del dueño, día 3): «Wrap integral de queso blanco y auyama al microondas con huevo»
recibía «Montaje: rellena la casabe con lo que cierra… un wrap que se puede cerrar»: el nombre no nombra su pan («wrap»
no es un pan del 378) y el soporte volvía a ser el casabe que añadió el cerrador."""
from __future__ import annotations

import dish_structure as ds
import recipe_repair as rr

_WRAP = {"name": "Wrap integral de queso blanco y auyama al microondas con huevo",
         "ingredients": ["2 tortillas integrales", "31 g de casabe", "250 g de auyama", "20 g de queso blanco", "2 huevos"],
         "recipe": ["Montaje: reparte el queso y la auyama entre las tortillas y dobla cada tortilla como wrap."]}


def test_el_wrap_integral_no_se_rellena_de_casabe():
    assert ds.familia(_WRAP) == "tostada_wrap"
    assert not any(r["tipo"] == "wrap_desproporcionado" for r in ds.relaciones(_WRAP))
    m = dict(_WRAP, recipe=list(_WRAP["recipe"]) + [
        "Montaje: rellena la casabe con lo que cierra (unos 90 g del relleno) y sirve el resto del relleno (~270 g) al lado, "
        "como ensalada: misma compra, un wrap que se puede cerrar."])
    assert rr.retirar_wrap_que_no_es(m) == 1 and m["recipe"] == _WRAP["recipe"]


def test_el_wrap_con_su_tortilla_sigue_acusado():
    w = {"name": "Wrap de pollo con lechuga", "ingredients": ["1 tortilla de trigo (40 g)", "200 g de pechuga de pollo",
                                                              "100 g de lechuga"],
         "recipe": ["Montaje: rellena la tortilla con el pollo y la lechuga."]}
    assert any(r["tipo"] == "wrap_desproporcionado" for r in ds.relaciones(w))
    c = {"name": "Wrap criollo de casabe con queso", "ingredients": ["40 g de casabe", "250 g de queso blanco"],
         "recipe": ["Montaje: rellena el casabe."]}
    assert any(r["tipo"] == "wrap_desproporcionado" for r in ds.relaciones(c))              # el nombre dice casabe
