# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-239 · 2026-09-25] El cambio de arroz de noche respeta alergias, rechazos, dieta y tiempo de cocina.

Auditoría del 25-sep: la rotación batata/yuca/casabe/ñame/auyama no miraba el formulario y además renombra el plato, así
que la última palabra del escudo no puede quitarla (ya es la identidad); en la versión final corre tras el escaneo.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def _cena(nombre="Pollo guisado con arroz blanco"):
    return {"meal": "Cena", "name": nombre, "ingredients": ["150 g de pechuga de pollo", "100 g de arroz blanco cocido"],
            "ingredients_raw": ["150g de pechuga de pollo", "100g de arroz blanco cocido"],
            "recipe": ["Mise en place: corta el pollo.", "El Toque de Fuego: guisa el pollo 20 min.", "Montaje: sirve con el arroz."],
            "cals": 500, "protein": 45, "carbs": 30, "fats": 10}


def _sustituto(form_data, dia=1):
    days = [{"day": d + 1, "meals": [_cena()]} for d in range(dia + 1)]
    go._night_rice_autofix(days, None, form_data=form_data)
    return " ".join(days[dia]["meals"][0]["ingredients"]).lower(), days[dia]["meals"][0]["name"].lower()


def test_la_rotacion_salta_lo_que_el_usuario_no_come():
    for d in range(5):
        ings, nombre = _sustituto({"allergies": ["Yuca"], "dislikes": ["Batata"]}, dia=d)
        assert "yuca" not in ings and "batata" not in ings, (d, ings)
        assert "yuca" not in nombre and "batata" not in nombre, (d, nombre)


def test_sin_tiempo_solo_casabe():
    for d in range(5):
        ings, _ = _sustituto({"cookingTime": "none"}, dia=d)
        assert "casabe" in ings or "arroz" in ings, ings
        assert not any(t in ings for t in ("batata", "yuca", "ñame", "name ", "auyama")), ings


def test_sin_tiempo_y_sin_casabe_no_se_cambia():
    ings, nombre = _sustituto({"cookingTime": "none", "dislikes": ["Casabe"]})
    assert "arroz" in ings and "casabe" not in ings, ings


def test_sin_restricciones_la_rotacion_de_siempre():
    assert go._night_rice_sub_for(0, {}) == go._NIGHT_RICE_SUB_ROTATION[0]
    assert go._night_rice_sub_for(2, None) == go._NIGHT_RICE_SUB_ROTATION[2]


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 239
