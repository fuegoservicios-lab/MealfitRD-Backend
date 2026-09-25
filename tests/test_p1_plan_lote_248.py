# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-248 · 2026-09-25] La última palabra que retira un alérgeno nombrado en un paso lo dice en una nota.

El relleno fantasma añade «30 g de aguacate» PORQUE un paso dice «sirve con aguacate»; el lote 233 retira la línea a un
alérgico, pero el paso seguía ahí. Escanear los pasos en el revisor dio 5 de 5 falsos positivos en la batería, así que
la defensa va aquí: nota explícita de omisión.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import restricciones_finales as rf  # noqa: E402

PLAN = {"days": [{"day": 1, "meals": [{
    "meal": "Almuerzo", "name": "Bowl de pollo con arroz",
    "ingredients": ["150 g de pechuga de pollo", "½ taza de arroz", "30 g de aguacate"],
    "recipe": ["Mise en place: corta el pollo.", "El Toque de Fuego: saltea el pollo 8 min.",
               "Montaje: sirve con aguacate en láminas."]}]}]}


def test_la_nota_de_omision_aparece():
    p = copy.deepcopy(PLAN)
    rf.retirar_prohibidos(p, {"allergies": ["Aguacate"]})
    m = p["days"][0]["meals"][0]
    assert "30 g de aguacate" not in m["ingredients"]
    notas = [s for s in m["recipe"] if s.startswith("⚠️ Alergia declarada")]
    assert notas and "aguacate" in notas[0], m["recipe"]


def test_sin_mencion_en_los_pasos_no_hay_nota():
    p = copy.deepcopy(PLAN)
    p["days"][0]["meals"][0]["recipe"][-1] = "Montaje: sirve caliente."
    rf.retirar_prohibidos(p, {"allergies": ["Aguacate"]})
    assert not [s for s in p["days"][0]["meals"][0]["recipe"] if s.startswith("⚠️")]


def test_un_rechazo_no_lleva_nota_de_alergia():
    p = copy.deepcopy(PLAN)
    rf.retirar_prohibidos(p, {"dislikes": ["Aguacate"]})
    assert not [s for s in p["days"][0]["meals"][0]["recipe"] if s.startswith("⚠️")]


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 248
