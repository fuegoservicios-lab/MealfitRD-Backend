# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-520 · 2026-09-27] El queso que el texto nombra es el de la lista.

Cadena completa re-encadenada desde la salida guardada de la IA (173 planes): 58 comidas nombran una variedad de queso
que la lista no trae, con UN solo queso distinto en la lista — «Yogurt natural con guineo fresco y queso cottage…
Termina con queso cottage» con «40 g de queso bajo en sodio» (adulto mayor con HTA), «… y queso mozzarella» con «15 g de
queso pasteurizado» (embarazo).
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import queso_de_la_lista as qdl  # noqa: E402
import recipe_contract as rc  # noqa: E402


def test_el_cottage_del_texto_es_el_queso_de_la_lista():
    m = {"name": "Yogurt natural con guineo fresco y queso cottage",
         "ingredients": ["½ taza de yogurt natural sin azúcar", "60 g de guineo", "40 g de queso bajo en sodio"],
         "recipe": ["Montaje: coloca yogurt natural en un tazón y distribuye encima el guineo. Termina con queso cottage."]}
    assert qdl.alinear(m) == 2
    assert m["name"] == "Yogurt natural con guineo fresco y queso bajo en sodio"
    assert m["recipe"][0].endswith("Termina con queso bajo en sodio."), m["recipe"][0]


def test_la_mozzarella_del_nombre_con_queso_pasteurizado():
    m = {"name": "Mango con zanahoria, semillas de calabaza y queso mozzarella",
         "ingredients": ["185 g de mango", "15 g de queso pasteurizado"], "recipe": ["Montaje: sirve el mango."]}
    qdl.alinear(m)
    assert m["name"] == "Mango con zanahoria, semillas de calabaza y queso pasteurizado", m["name"]


def test_con_dos_quesos_o_la_misma_variedad_no_se_toca():
    m = {"name": "Pasta con parmesano y cottage", "ingredients": ["20 g de queso parmesano", "50 g de queso cottage"],
         "recipe": ["Montaje: espolvorea parmesano."]}
    assert qdl.alinear(m) == 0
    m = {"name": "Casabe con queso crema", "ingredients": ["30 g de queso crema"], "recipe": ["Montaje: unta el queso crema."]}
    assert qdl.alinear(m) == 0
    m = {"name": "Bowl con cottage", "ingredients": ["100 g de queso cottage bajo en sodio"],
         "recipe": ["Montaje: sirve el queso cottage."]}
    assert qdl.alinear(m) == 0


def test_las_notas_no_se_tocan():
    m = {"name": "Tostada con queso", "ingredients": ["30 g de queso blanco pasteurizado"],
         "recipe": ["⚠️ Seguridad alimentaria: el queso cottage de tu plan debe ser pasteurizado."]}
    assert qdl.alinear(m) == 0


def test_esta_en_la_cola_del_contrato():
    import inspect
    assert "queso_de_la_lista" in inspect.getsource(rc)
