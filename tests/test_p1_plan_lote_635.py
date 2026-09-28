# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-635 · 2026-09-28] El vaso de leche del 587 sigue a la lista cuando el shield la reescala.

Validación del 592 (estudiante, día 2): lista «275 ml de leche descremada»; pasos «mide… 360 ml», «con 150 ml de la
leche», «Sirve los 210 ml de leche restantes» (150 + 210 = 360, la lista de la primera pasada).
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import avena_con_su_leche as acl  # noqa: E402


def _avena(lista_ml):
    return {
        "name": "Avena cremosa con mango, maní y queso cottage",
        "ingredients": ["30 g de avena", f"{lista_ml} ml de leche descremada", "60 g de mango", "10 g de maní picado",
                        "135 g de queso cottage"],
        "recipe": [
            "Mise en place: mide 30 g de avena y 360 ml de leche descremada; pela y corta 60 g de mango en cubos.",
            "El Toque de Fuego: cocina la avena con 150 ml de la leche descremada y la canela a fuego medio durante 7-9 "
            "minutos, removiendo hasta que quede cremosa.",
            "Montaje: sirve la avena en un tazón y corona con el mango. Sirve los 210 ml de leche restantes en un vaso "
            "aparte. Acompaña con queso cottage.",
        ],
        "_avena_leche_vaso": 210,
    }


def test_el_vaso_y_el_mise_siguen_a_la_lista():
    m = _avena(275)
    assert acl.resincronizar(m) == 2
    assert "mide 30 g de avena y 275 ml de leche descremada" in m["recipe"][0]
    assert "con 150 ml de la leche" in m["recipe"][1]
    assert "Sirve los 125 ml de leche restantes en un vaso aparte. Acompaña con queso cottage." in m["recipe"][2]
    assert m["_avena_leche_vaso"] == 125
    assert acl.resincronizar(m) == 0                      # idempotente


def test_si_ya_no_sobra_un_vaso_toda_la_leche_va_a_la_olla():
    m = _avena(220)
    acl.resincronizar(m)
    assert "vaso" not in " ".join(m["recipe"])
    assert "con 220 ml de la leche" in m["recipe"][1]
    assert "220 ml de leche descremada" in m["recipe"][0]
    assert "_avena_leche_vaso" not in m


def test_cuadrado_no_se_toca():
    m = _avena(360)
    antes = copy.deepcopy(m["recipe"])
    assert acl.resincronizar(m) == 0 and m["recipe"] == antes


def test_corre_justo_despues_de_separar_en_el_contrato():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    llamadas = re.findall(r'__import__\("([a-z_]+)"\)\.([a-z_]+)\(meal', src)
    i = llamadas.index(("avena_con_su_leche", "separar"))
    assert llamadas[i + 1] == ("avena_con_su_leche", "resincronizar")
