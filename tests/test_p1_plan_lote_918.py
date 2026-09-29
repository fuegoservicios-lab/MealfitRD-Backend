# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-918 · 2026-09-29] En un plato de base DULCE la fruta está en su sitio, aunque lleve huevo al lado.

Baterías reales del 28/29-sep: «Avena cremosa con aguacate, huevo bien cocido y yogurt griego entero» (rdv809, con la ficha
«endulzada con aguacate maduro en cubos y coronada con aguacate fresco») y «Avena cremosa con aguacate y huevos
sancochados» (rdv868). El cerrador de proteína le pone huevo a la avena, el nombre pasa a decir «…guayaba y huevo» y el
detector de pareo fruta + salado lo lee como «huevo + guayaba»: el autocorrector cambia la fruta por aguacate. Corpus del
VPS (8.574 comidas): 680 con la marca `fruit_savory`; 120 empiezan por una base dulce y en 114 el único salado del nombre
es el huevo; 23 en baterías desde el 27-sep.
"""
from __future__ import annotations

import copy
import pathlib

import base_dulce as bd
import culinary_context as cc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _clash(nombre: str) -> bool:
    return cc._meal_has_sweet_savory_clash({"name": nombre})


def test_la_fruta_de_la_avena_no_choca_con_el_huevo_de_al_lado():
    for nombre in ("Avena cremosa con guayaba y huevo bien cocido",
                   "Avena cremosa con lechosa, huevo bien cocido y yogurt griego entero",
                   "Avena cremosa de canela con mango y huevos revueltos",
                   "Yogur griego con mango, maní y huevo duro",
                   "Panqueques de avena con lechosa y huevos revueltos",
                   "Bowl fresco de yogur, guayaba y huevo cocido",
                   "Batido de mango con avena y huevo cocido al lado"):
        assert _clash(nombre) is False, nombre


def test_el_pareo_de_verdad_sigue_siendo_choque():
    for nombre in ("Revoltillo de huevo con mango",                       # la base es el huevo
                   "Mangú de plátano verde con huevo y guayaba",
                   "Tostadas integrales con huevo y guayaba",            # la tostada no es base dulce
                   "Arroz blanco con mango",
                   "Avena cremosa con mango y arroz",                    # base dulce, pero el salado no es el huevo
                   "Avena salada con huevo y mango"):                    # la avena salada es un plato salado
        assert _clash(nombre) is True, nombre


def test_el_autocorrector_deja_la_fruta_de_la_avena(monkeypatch):
    import graph_orchestrator as go
    plato = {"meal": "Desayuno", "name": "Avena cremosa con guayaba y huevo bien cocido",
             "ingredients": ["30 g de avena", "60 g de guayaba", "1 huevo"],
             "ingredients_raw": ["30 g de avena", "60 g de guayaba", "1 huevo"],
             "recipe": ["El Toque de Fuego: cocina la avena 7-9 min; hierve el huevo 10-12 min.",
                        "Montaje: sirve la avena con la guayaba por encima y el huevo al lado."]}
    antes = copy.deepcopy(plato)
    go._fruit_savory_autofix([{"day": 1, "meals": [plato]}], form_data={})
    assert plato["name"] == antes["name"] and plato["ingredients"] == antes["ingredients"]
    assert plato.get("_slot_autofix_applied") != "fruit_savory"
    salado = {"meal": "Desayuno", "name": "Revoltillo de huevo con mango",
              "ingredients": ["2 huevos", "80 g de mango"], "ingredients_raw": ["2 huevos", "80 g de mango"],
              "recipe": ["El Toque de Fuego: revuelve los huevos 3 min.", "Montaje: sirve con el mango."]}
    go._fruit_savory_autofix([{"day": 1, "meals": [salado]}], form_data={})
    assert "mango" not in salado["name"].lower(), "el revoltillo con mango sigue corrigiéndose"


def test_la_fruta_de_agua_con_carne_sigue_siendo_choque_tambien_en_base_dulce():
    """La exención es sólo para el huevo: la segunda regla del detector (fruta de agua + carne o pescado) sigue mirando."""
    assert _clash("Yogur con lechosa, huevo cocido y pollo desmenuzado") is True


def test_con_el_knob_apagado_vuelve_la_conducta_anterior(monkeypatch):
    monkeypatch.setenv("MEALFIT_SWEET_BASE_KEEPS_FRUIT", "false")
    assert _clash("Avena cremosa con guayaba y huevo bien cocido") is True


def test_la_regla_sola():
    assert bd.fruta_en_su_sitio("avena cremosa con guayaba y huevo bien cocido", ["huevo"]) is True
    assert bd.fruta_en_su_sitio("avena cremosa con guayaba", []) is False, "sin salado no hay nada que eximir"
    assert bd.fruta_en_su_sitio("tostadas integrales con huevo y guayaba", ["huevo"]) is False
    assert bd.fruta_en_su_sitio("avena cremosa con mango y arroz", ["arroz"]) is False


def test_ancla_en_el_detector():
    src = (_BACKEND / "culinary_context.py").read_text(encoding="utf-8")
    cuerpo = src[src.index("def _meal_has_sweet_savory_clash("):src.index("def ceil_div(")]
    assert 'if not __import__("base_dulce").fruta_en_su_sitio(name_low, _salados):  # [P1-PLAN-LOTE-918]' in cuerpo
    modulo = _BACKEND / "base_dulce.py"
    assert "tooltip-anchor: P1-PLAN-LOTE-918" in modulo.read_text(encoding="utf-8") and b"\x08" not in modulo.read_bytes()
