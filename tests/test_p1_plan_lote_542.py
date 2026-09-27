# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-542 · 2026-09-27] «30 g de quinoa» (sin «seca») usada cocida trae su cocción previa.

Batería real del 27-sep (DM2 con insulina, día 1, almuerzo): la lista compra «30 g de quinoa», el lote 343 escribe «mide
85 g de quinoa cocida» (la fila del catálogo está en seco) y ningún paso la cuece; el 393 sólo miraba la palabra «seca».
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cantidades as pc  # noqa: E402


class _Db:
    def lookup(self, nombre):
        n = str(nombre).lower()
        if "quinoa" in n:
            return SimpleNamespace(name="Quinoa", kcal=368.0)
        if "arroz" in n:
            return SimpleNamespace(name="Arroz blanco cocido", kcal=130.0)
        return None


def _bowl(linea):
    return {"name": "Bowl caribeño fresco de pollo y quinoa al limón",
            "ingredients": ["½ pechuga de pollo (≈120 g)", linea, "½ taza de lechuga romana"],
            "recipe": ["Mise en place: corta la pechuga en tiras; mide 85 g de quinoa cocida.",
                       "El Toque de Fuego: cocina el pollo 5-7 min por lado hasta 74 °C.",
                       "Montaje: coloca la quinoa cocida sobre la lechuga romana e incorpora el pollo."]}


def test_quinoa_sin_seca_con_la_fila_en_seco():
    m = _bowl("30 g de quinoa")
    assert pc.seco_usado_cocido(m, _Db()) == 1
    assert m["recipe"][1] == ("💡 Cocción previa: enjuaga la quinoa cruda y cuécela en agua 12-15 min hasta que esté "
                              "tierna, y escúrrela."), m["recipe"]
    assert pc.seco_usado_cocido(m, _Db()) == 0                     # idempotente


def test_sin_catalogo_o_con_la_fila_en_cocido_no_se_toca():
    m = _bowl("30 g de quinoa")
    assert pc.seco_usado_cocido(m) == 0                            # sin db: la conducta de antes
    m = _bowl("85 g de quinoa cocida")
    assert pc.seco_usado_cocido(m, _Db()) == 0


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").seco_usado_cocido(meal, db)' in src
