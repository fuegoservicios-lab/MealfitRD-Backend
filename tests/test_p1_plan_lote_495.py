# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-495 · 2026-09-27] Con alergia al pescado, la reserva de proteína de la compra única son las claras.

Batería real del 27-sep (hombre de 38 años, alérgico al pescado, pierde grasa, 30 días sin congelador; días 4-7): atún y
sardinas no son seguros y toda proteína que no aguanta caía en garbanzos, que el band-closer trata como carbohidrato y
recorta. La cena «Pollo a la plancha con maíz dulce» (1¾ pechugas, 71 g de proteína) salió con 70 g de garbanzos (6 g):
el día al 43 % de su proteína. Las claras pasteurizadas (botella de 400 g del súper, 35 días en nevera) van antes de los
garbanzos; se baten y se cuajan. Con «Nada» de tiempo no (hay que cocinarlas) y con alergia al huevo tampoco.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402

_REQ = {"need_days": 5, "allow_frozen": False}


def test_alergico_al_pescado_recibe_claras():
    r = cu.sustituir_linea("1¾ pechugas de pollo (≈279 g)", 5, _REQ, alergias=["Pescado"], semilla=0,
                           gramos_de=lambda _t: None, listo=False)
    assert r and r[1] == "claras de huevo", r
    assert r[0] == "6 claras de huevo", "279 g → tope de 6 claras por comida"
    r = cu.sustituir_linea("90 g de pechuga de pollo", 5, _REQ, alergias=["Pescado"], semilla=0,
                           gramos_de=lambda _t: None, listo=False)
    assert r and r[0] == "3 claras de huevo", r


def test_con_pescado_la_rueda_sigue_siendo_de_lata():
    r = cu.sustituir_linea("150 g de pechuga de pollo", 5, _REQ, semilla=0, gramos_de=lambda _t: None, listo=False)
    assert r and r[1] in ("atun en agua", "sardinas en lata"), r


def test_sin_tiempo_o_alergia_al_huevo_van_garbanzos():
    r = cu.sustituir_linea("150 g de pechuga de pollo", 5, _REQ, alergias=["Pescado"], semilla=0,
                           gramos_de=lambda _t: None, listo=True)
    assert r and r[1] == "garbanzos cocidos", r
    r = cu.sustituir_linea("150 g de pechuga de pollo", 5, _REQ, alergias=["Pescado", "Huevo"], semilla=0,
                           gramos_de=lambda _t: None, listo=False)
    assert r and r[1] == "garbanzos cocidos", r


def test_las_claras_se_baten_y_se_cuajan():
    m = {"name": "Pollo a la plancha con maíz dulce y ajíes morrones salteados",
         "ingredients": ["1¾ pechugas de pollo (≈279 g)", "80 g de maíz dulce en granos"],
         "ingredients_raw": ["1¾ pechugas de pollo (≈279 g)", "80 g de maíz dulce en granos"],
         "recipe": ["Mise en place: corta 1¾ pechugas de pollo en filetes finos y sazónalos con sal al gusto.",
                    "El Toque de Fuego: sella los filetes de pollo 5-6 minutos por lado, hasta que alcancen 74 °C en "
                    "el centro, y retíralos. En la misma sartén saltea el maíz 2 minutos.",
                    "⚠️ Seguridad alimentaria: cocina el pollo por completo (74 °C) antes de servir."]}
    sf.sustituir_en_plato(m, 0, "1¾ pechugas de pollo (≈279 g)", "6 claras de huevo", "claras de huevo")
    assert m["name"] == "Claras a la plancha con maíz dulce y ajíes morrones salteados", m["name"]
    assert m["recipe"][0] == "Mise en place: bate 6 claras de huevo y sazónalas con sal al gusto.", m["recipe"][0]
    assert m["recipe"][1] == ("El Toque de Fuego: cocina las claras 2-3 minutos, hasta que cuajen, y retíralas. "
                              "En la misma sartén saltea el maíz 2 minutos."), m["recipe"][1]
    assert len(m["recipe"]) == 2, "la nota del pollo crudo sale"


def test_el_verbo_de_una_lista_se_queda_con_los_demas():
    """Rechain de la batería real: «corta 1¾ pechugas de pollo, ½ ají morrón en tiras y ½ cebolla en plumas» salía «bate
    6 claras de huevo, ½ ají morrón en tiras…»: nadie bate un ají."""
    m = {"name": "Pollo con ajíes", "ingredients": ["1¾ pechugas de pollo (≈279 g)"],
         "ingredients_raw": ["1¾ pechugas de pollo (≈279 g)"],
         "recipe": ["Mise en place: corta 1¾ pechugas de pollo en tiras, ½ ají morrón en tiras y ½ cebolla en plumas."]}
    sf.sustituir_en_plato(m, 0, "1¾ pechugas de pollo (≈279 g)", "6 claras de huevo", "claras de huevo")
    assert m["recipe"][0] == ("Mise en place: bate 6 claras de huevo; corta ½ ají morrón en tiras y ½ cebolla en "
                              "plumas."), m["recipe"][0]
