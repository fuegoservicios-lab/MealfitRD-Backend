# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-783 · 2026-09-28] Embarazo: el sustituto del pescado de LATA se cocina.

Con el tope de pescado (lote 187), «Mezcla el atún en agua (ya viene cocido)… sirve frío» quedaba «mezcla pechuga de pavo
en agua (ya viene cocido)… sirve frío» y la batería real de lactancia servía «Acompaña con pechuga de pavo en agua»:
carne cruda, fría, a una embarazada.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import embarazo_pescado as ep  # noqa: E402


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+)\s*g\b", s)
        return float(m.group(1)) if m else 100.0

    def lookup(self, s):
        return None


def _comida(nombre, lineas, pasos, franja="Almuerzo"):
    return {"meal": franja, "name": nombre, "ingredients": list(lineas), "ingredients_raw": list(lineas),
            "recipe": list(pasos)}


def _plan(comida_pescado):
    dias = []
    for d in range(1, 4):
        dias.append({"day": d, "meals": [
            _comida("Revoltillo con casabe", ["2 huevos", "1 casabe"], ["Montaje: sirve."], "Desayuno"),
            comida_pescado(),
            _comida("Pollo guisado con arroz", ["150 g de pechuga de pollo", "½ taza de arroz"],
                    ["El Toque de Fuego: guisa el pollo 20 min."], "Cena")]})
    return {"days": dias}


_FD = {"medicalConditions": ["Embarazo"], "dietType": "balanced", "allergies": ["Ninguna"], "dislikes": ["Ninguno"]}


def _ensalada_atun():
    return _comida("Ensalada de atún en agua con aguacate y tomate",
                   ["150 g de atún en agua", "50 g de aguacate", "1 tomate"],
                   ["Mise en place: escurre el atún en agua y pica el tomate.",
                    "Montaje: mezcla el atún en agua (ya viene cocido) con el tomate y el aguacate; sirve frío."])


def test_el_sustituto_del_atun_de_lata_se_cocina():
    plan = _plan(_ensalada_atun)
    assert ep.limitar_pescado(plan, _FD, db=_DB()) == 1
    m = plan["days"][2]["meals"][1]
    assert m["_embarazo_pescado_cap"] and "pechuga" in m["ingredients"][0]
    texto = " ".join([m["name"]] + [p for p in m["recipe"] if not p.startswith("💡")]).lower()
    assert "en agua" not in texto, m
    assert "ya viene cocid" not in texto, m["recipe"]
    assert not re.search(r"escurre[^.;]{0,20}pechuga", texto), m["recipe"]
    previa = [p for p in m["recipe"] if p.startswith("💡 Cocción previa")]
    assert previa and "74 °C" in previa[0], m["recipe"]
    assert m["recipe"].index(previa[0]) == 1, "va tras el Mise en place"


def test_acompana_con_pavo_en_agua_tambien():
    def revoltillo():
        return _comida("Revoltillo criollo de tomate con aguacate y atún en agua",
                       ["3 huevos", "1 tomate", "½ aguacate", "165 g de atún en agua"],
                       ["Mise en place: bate 3 huevos.",
                        "El Toque de Fuego: cocina la cebolla y el tomate 3 minutos y cuaja los huevos.",
                        "Montaje: sirve el revoltillo con el aguacate. Acompaña con atún en agua."], "Desayuno")
    plan = _plan(revoltillo)
    ep.limitar_pescado(plan, _FD, db=_DB())
    m = next(mm for d in plan["days"] for mm in d["meals"] if mm.get("_embarazo_pescado_cap"))
    pasos = " ".join(p for p in m["recipe"] if not p.startswith("💡")).lower()
    assert "en agua" not in m["name"].lower() and "en agua" not in pasos, m
    assert any(p.startswith("💡 Cocción previa") for p in m["recipe"]), m["recipe"]


def test_el_pescado_fresco_sigue_como_estaba():
    def tilapia():
        return _comida("Tilapia a la plancha con arroz", ["150 g de filete de tilapia", "½ taza de arroz"],
                       ["El Toque de Fuego: cocina la tilapia a la plancha 4 min por lado."])
    plan = _plan(tilapia)
    ep.limitar_pescado(plan, _FD, db=_DB())
    m = next(mm for d in plan["days"] for mm in d["meals"] if mm.get("_embarazo_pescado_cap"))
    assert not any(p.startswith("💡 Cocción previa") for p in m["recipe"]), m["recipe"]


def test_que_es_de_lata():
    assert ep._es_de_lata("150 g de atún en agua") and ep._es_de_lata("2 sardinas en lata")
    assert ep._es_de_lata("120 g de atún claro") and not ep._es_de_lata("150 g de filete de atún fresco")
    assert not ep._es_de_lata("150 g de filete de tilapia")


def test_el_mise_en_place_no_escurre_ni_desmenuza_el_crudo():
    plan = _plan(_ensalada_atun)
    ep.limitar_pescado(plan, _FD, db=_DB())
    m = plan["days"][2]["meals"][1]
    mise = next(p for p in m["recipe"] if p.lower().startswith("mise en place"))
    assert "pechuga" not in mise.lower() and "pica el tomate" in mise, m["recipe"]
