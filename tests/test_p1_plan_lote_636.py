# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-636 · 2026-09-28] El yogur que el texto nombra es el de la lista.

Validación del 592 (estudiante, merienda del día 2): «20 g de yogurt natural entero» en la lista y «Acompaña con yogurt
natural entero y yogurt griego entero» en el Montaje. Replay: 255 de 5.042 comidas nombran otra clase de yogur.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import yogur_de_la_lista as ydl  # noqa: E402


def test_dos_yogures_en_el_montaje_con_uno_en_la_lista():
    m = {"name": "Casabe crujiente con mantequilla de maní, canela y yogurt natural entero",
         "ingredients": ["1 porción de casabe (30 g)", "15 g de mango", "20 g de yogurt natural entero"],
         "recipe": ["Mise en place: corta 15 g de mango en cubos.",
                    "Montaje: unta el casabe y sirve el mango al lado. Acompaña con yogurt natural entero y yogurt "
                    "griego entero."]}
    assert ydl.alinear(m) == 1
    assert m["recipe"][1] == "Montaje: unta el casabe y sirve el mango al lado. Acompaña con yogurt natural entero."
    assert ydl.alinear(m) == 0                                   # idempotente


def test_lista_generica_tras_la_fusion_sirve_un_solo_yogur():
    m = {"name": "Avena cremosa con lechosa, mantequilla de maní y yogurt natural entero",
         "ingredients": ["½ taza de avena", "95 g de lechosa en cubos", "⅔ taza de yogurt"],
         "recipe": ["El Toque de Fuego: calienta la avena 2-3 min. Sirve Yogurt al lado para acompañar. Sirve yogurt "
                    "griego entero al lado para acompañar.",
                    "Montaje: sirve la avena con la lechosa por encima. Acompaña con yogurt natural entero. Acompaña con "
                    "yogurt griego entero."]}
    ydl.alinear(m)
    assert m["recipe"][0] == "El Toque de Fuego: calienta la avena 2-3 min. Sirve Yogurt al lado para acompañar."
    assert m["recipe"][1] == "Montaje: sirve la avena con la lechosa por encima. Acompaña con yogurt."
    assert m["name"] == "Avena cremosa con lechosa, mantequilla de maní y yogurt"


def test_el_nombre_dice_natural_y_la_lista_compra_griego():
    m = {"name": "Pan integral tostado con queso blanco fresco, lechosa en cubos y yogurt natural entero",
         "ingredients": ["1 rebanada de pan integral", "80 g de yogurt griego entero"],
         "recipe": ["Montaje: coloca el queso sobre el pan. Acompaña con yogurt natural entero."]}
    ydl.alinear(m)
    assert m["name"].endswith("y yogurt griego")
    assert m["recipe"][0] == "Montaje: coloca el queso sobre el pan. Acompaña con yogurt griego."


def test_pasteurizado_se_queda_y_las_notas_no_se_tocan():
    nota = "🤰 Seguridad alimentaria: el yogur natural debe ser pasteurizado."
    m = {"name": "Vasito de yogur natural pasteurizado con guanábana",
         "ingredients": ["½ taza de yogurt griego sin azúcar"],
         "recipe": ["Montaje: sirve el yogur natural pasteurizado con la guanábana.", nota]}
    ydl.alinear(m)
    assert m["name"] == "Vasito de yogurt griego pasteurizado con guanábana"
    assert m["recipe"][1] == nota


def test_sin_contradiccion_no_se_toca():
    casos = [
        # dos yogures en la lista: no se sabe cuál es cuál
        {"name": "Yogurt griego con fresas", "ingredients": ["80 g de yogurt griego", "80 g de yogurt natural"],
         "recipe": ["Montaje: sirve el yogurt griego y el yogurt natural."]},
        # lista genérica y el texto nombra UNA clase
        {"name": "Yogurt natural con uvas", "ingredients": ["80 g de yogurt"], "recipe": ["Montaje: sirve el yogurt natural."]},
        # «entero»/«sin azúcar» no cambian de producto
        {"name": "Yogurt griego entero con mango", "ingredients": ["80 g de yogurt griego sin azúcar"],
         "recipe": ["Montaje: sirve el yogurt griego entero con el mango."]},
    ]
    for m in casos:
        antes = (m["name"], list(m["recipe"]))
        assert ydl.alinear(m) == 0 and (m["name"], m["recipe"]) == antes


def test_corre_tras_el_queso_de_la_lista_en_el_contrato():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    llamadas = re.findall(r'__import__\("([a-z_]+)"\)\.([a-z_]+)\(meal', src)
    i = llamadas.index(("queso_de_la_lista", "alinear"))
    assert llamadas[i + 1] == ("yogur_de_la_lista", "alinear")
