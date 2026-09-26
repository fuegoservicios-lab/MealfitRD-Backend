# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-330 · 2026-09-25] Una pizca de la lista no es «½ g» en el paso.

Batería de cierre del 25-sep: «mide … y ½ g de semillas de girasol» con «1 pizca de semillas de girasol sin sal» en la
lista, «⅔ g de ajonjolí» con «1 pizca de ajonjolí»: el pulido convierte las migajas de la lista en pizcas (lote 181) y el
paso se quedaba con el gramo de máquina."""
from __future__ import annotations

import pasos_cantidades as pc


def test_la_migaja_del_paso_pasa_a_pizca():
    m = {"ingredients": ["60 g de guineo", "1 pizca de semillas de girasol sin sal", "1 pizca de ajonjolí"],
         "recipe": ["Mise en place: mide 60 g de guineo y ½ g de semillas de girasol; ten ⅔ g de ajonjolí.",
                    "Montaje: espolvorea."]}
    assert pc.pizcas_de_la_lista(m) == 1
    assert m["recipe"][0] == ("Mise en place: mide 60 g de guineo y 1 pizca de semillas de girasol; ten 1 pizca de "
                              "ajonjolí.")


def test_sin_pizca_en_la_lista_no_se_toca():
    pasos = ["Mise en place: mide ½ g de canela.", "⚠️ Seguridad alimentaria: ½ g de sal como máximo."]
    m = {"ingredients": ["2 g de canela", "1 pizca de sal"], "recipe": list(pasos)}
    assert pc.pizcas_de_la_lista(m) == 0 and m["recipe"] == pasos
    assert "pizcas_de_la_lista(meal)" in __import__("inspect").getsource(pc.lo_que_dice_la_lista)
