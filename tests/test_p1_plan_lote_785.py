# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-785 · 2026-09-28] El sustituto del pescado lleva SU punto: 74 °C el ave, 71 °C la carne.

Corpus de la cola 744 (embarazo): el tope de pescado cambió el pescado de unas tortitas por pechuga de pavo y el paso
quedó «hornéalas a 200 °C hasta que estén firmes, el huevo completamente cocido y pechuga de pavo alcance 63 °C».
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


_FD = {"medicalConditions": ["Embarazo"], "dietType": "balanced", "allergies": ["Ninguna"], "dislikes": ["Ninguno"]}


def _plan(comida):
    dias = []
    for d in range(1, 4):
        dias.append({"day": d, "meals": [
            {"meal": "Desayuno", "name": "Revoltillo con casabe", "ingredients": ["2 huevos"], "recipe": ["Montaje: sirve."]},
            comida(),
            {"meal": "Cena", "name": "Res guisada", "ingredients": ["150 g de carne de res"],
             "recipe": ["El Toque de Fuego: guisa la res 30 min."]}]})
    return {"days": dias}


def _tortitas():
    return {"meal": "Almuerzo", "name": "Tortitas horneadas de pescado con arroz",
            "ingredients": ["150 g de filete de pescado", "3 huevos", "⅓ taza de arroz"],
            "ingredients_raw": ["150 g de filete de pescado", "3 huevos", "⅓ taza de arroz"],
            "recipe": ["Mise en place: pica el pescado; bate 3 huevos.",
                       "El Toque de Fuego: mezcla el pescado con el huevo, forma tortitas y hornéalas a 200 °C hasta que "
                       "estén firmes, el huevo completamente cocido y el pescado alcance 63 °C.",
                       "🤰 Seguridad alimentaria (embarazo/lactancia): cocina el pescado POR COMPLETO (63 °C)."]}


def test_el_ave_no_se_queda_en_63():
    plan = _plan(_tortitas)
    assert ep.limitar_pescado(plan, _FD, db=_DB()) == 1
    m = next(mm for d in plan["days"] for mm in d["meals"] if mm.get("_embarazo_pescado_cap"))
    paso = m["recipe"][1]
    assert "pechuga" in paso and "63 °C" not in paso and "74 °C" in paso, paso
    assert "200 °C" in paso, "la cifra del horno no se toca"
    assert m["recipe"][2].startswith("🤰"), "las notas no se tocan"


def test_se_desmenuce_pasa_a_sin_partes_rosadas():
    m = {"name": "x", "recipe": ["El Toque de Fuego: cocina la pechuga de pollo al vapor 8 min, hasta que se desmenuce "
                                 "fácilmente."]}
    assert ep._punto_del_sustituto(m, "Pechuga de pollo") == 1
    assert m["recipe"][0].endswith("hasta que no quede rosada por dentro."), m["recipe"][0]


def test_la_carne_71():
    m = {"name": "x", "recipe": ["El Toque de Fuego: sella la carne de res hasta que alcance 63 °C."]}
    ep._punto_del_sustituto(m, "Carne de res")
    assert "71 °C" in m["recipe"][0]
