# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-866 · 2026-09-29] «No calientes el queso» no es calentarlo; en el embarazo, la orden se va y el queso se
dora. Y el paso nombra el queso de la LISTA.

Segunda batería real de embarazo del 29-sep (860-864 vivos), cena D1: «Nabo asado a la parrilla con queso blanco fresco…»
— «El Toque de Fuego: …ásalas a la parrilla… No calientes el queso blanco fresco pasteurizado.» con la nota «caliéntalo
hasta que humee». El detector leía «calient» + «queso» y daba el queso por calentado: el plato salía con las dos órdenes.
En la merienda D2 la lista decía «queso fresco pasteurizado» y el paso «Dora el queso blanco pasteurizado».
"""
from __future__ import annotations

import copy

import embarazo_seguro as es
import queso_que_humee as q

_EMBARAZO = {"medicalConditions": ["Embarazo"]}
_NABO = {
    "name": "Nabo asado a la parrilla con queso blanco fresco, aguacate y edamame",
    "ingredients": ["250 g de nabo", "20 g de queso blanco fresco pasteurizado", "½ aguacate"],
    "recipe": ["Mise en place: pela y corta 250 g de nabo en rodajas de 1 cm.",
               "El Toque de Fuego: unta las rodajas de nabo con el aceite y ásalas a la parrilla 8-10 min. No calientes el "
               "queso blanco fresco pasteurizado. Calienta el edamame cocido en agua hirviendo 2-3 minutos.",
               "Montaje: sirve el nabo asado con el queso blanco fresco pasteurizado y el aguacate."],
}


def _etiquetar(meal):
    plan = {"days": [{"day": 1, "meals": [copy.deepcopy(meal)]}]}
    es.etiquetar(plan, _EMBARAZO)
    return plan["days"][0]["meals"][0]["recipe"]


def test_la_orden_de_no_calentar_el_queso_se_va_y_el_queso_se_dora():
    rec = _etiquetar(_NABO)
    assert not any("No calientes el queso" in p for p in rec), rec
    assert "Dora el queso blanco pasteurizado en la sartén caliente" in rec[1], rec[1]
    assert "Calienta el edamame cocido en agua hirviendo 2-3 minutos." in rec[1], "lo demás del paso se queda"
    assert _etiquetar({**_NABO, "recipe": rec}) == rec, "idempotente"


def test_la_negacion_no_cuenta_como_calentar():
    assert not q._calentado(["El Toque de Fuego: asa el nabo. No calientes el queso blanco fresco."])
    assert q._calentado(["El Toque de Fuego: dora el queso blanco en la sartén 2 minutos."])


def test_el_paso_nombra_el_queso_de_la_lista():
    merienda = {"name": "Casabe tostado con queso fresco y lechosa",
                "ingredients": ["½ porción de casabe (15 g)", "25 g de queso fresco pasteurizado", "60 g de lechosa fresca"],
                "recipe": ["Mise en place: corta el queso fresco pasteurizado en láminas.",
                           "El Toque de Fuego: tuesta el casabe 1-2 min por lado.",
                           "Montaje: coloca sobre el casabe el queso fresco pasteurizado y la lechosa."]}
    rec = _etiquetar(merienda)
    assert "Dora el queso fresco pasteurizado en la sartén caliente" in rec[1], rec[1]
