# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-591 · 2026-09-27] Con sólo claras en la lista, el paso no pocha huevos enteros de yema cremosa.

Batería real (colesterol alto + atorvastatina): «5 claras de huevo» en la lista y «pocha los 2 huevos… hasta que la clara
cuaje y la yema quede cremosa» en el paso.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import claras_pochadas as cp  # noqa: E402

_TOQUE = ("El Toque de Fuego: cocina la berenjena 12-15 minutos. Aparte, pocha los 2 huevos en agua apenas hirviendo con "
          "un chorrito de vinagre de 3 a 4 minutos, hasta que la clara cuaje y la yema quede cremosa. Calienta la tortilla.")
_MONTAJE = "Montaje: cubre con la berenjena y corona con las claras pochados."


def _meal(lista):
    return {"ingredients": lista, "recipe": ["Mise en place: corta la berenjena.", _TOQUE, _MONTAJE]}


def test_la_bateria_cuaja_las_claras_de_la_lista():
    m = _meal(["1 berenjena mediana", "5 claras de huevo"])
    assert cp.alinear(m) == 2
    assert ("cuaja las claras de huevo en agua apenas hirviendo con un chorrito de vinagre de 3 a 4 minutos, hasta que "
            "estén firmes y opacas.") in m["recipe"][1], m["recipe"][1]
    assert "yema" not in m["recipe"][1] and m["recipe"][2].endswith("las claras pochadas.")


def test_con_huevos_enteros_en_la_lista_no_se_toca():
    m = _meal(["1 berenjena mediana", "2 huevos"])
    assert cp.alinear(m) == 0 and m["recipe"][1] == _TOQUE


def test_ancla_en_la_cola():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("claras_pochadas").alinear(meal)  # [P1-PLAN-LOTE-591]' in src
