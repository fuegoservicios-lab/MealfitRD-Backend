"""[P1-PLAN-LOTE-1098 · 2026-11-18] La cantidad de un renglón SIN gramos («1 taza», «2 unidad») se corrige en la ficha.

Replay del 18-nov en el servidor (768 renglones de 30 días): 163 sin gramos, ~100 con número y unidad que el catálogo
resuelve. El número solo se toca si de verdad es la cantidad: al duplicarlo, la energía se duplica.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import gramos_editables as ge  # noqa: E402

# palabra → kcal por UNIDAD de su medida casera (taza, unidad, vaso…)
_POR_UNIDAD = {"arroz": 205.0, "huevo": 72.0, "leche": 122.0, "aguacate": 57.0}


class Catalogo:
    def macros_from_ingredient_string(self, s):
        plano = s.lower()
        if "%" in plano:           # «leche 2 %»: el número no es la cantidad; la fila es fija
            return {"name": "Leche", "grams": 244, "kcal": 122.0, "protein": 8.0, "carbs": 12.0, "fats": 5.0}
        m = re.search(r"(\d+(?:\.\d+)?)", plano)
        if not m:
            return None
        n = float(m.group(1))
        for clave, k in _POR_UNIDAD.items():
            if clave in plano:
                return {"name": clave, "grams": n * 100, "kcal": round(k * n, 1), "protein": round(2 * n, 2),
                        "carbs": round(10 * n, 2), "fats": round(1 * n, 2)}
        return None


CENA = {"calories": 600, "protein": 30, "carbs": 70, "healthy_fats": 20, "ingredients": [
    "1 taza de arroz blanco", "2 unidad de huevo hervido", "Leche (≈1 vaso)", "Leche 2 % en el café",
    "Pechuga de pollo (≈120 g)", "Sal al gusto"]}


def test_que_renglones_se_corrigen_por_su_cantidad():
    q = ge.cantidades(CENA["ingredients"], Catalogo())
    assert q == {0: {"valor": 1.0, "unidad": "taza"}, 1: {"valor": 2.0, "unidad": "unidad"}, 2: {"valor": 1.0, "unidad": "vaso"}}, \
        "la leche 2 % (el número no es la cantidad), el pollo (tiene gramos) y la sal (sin número) no"


def test_el_numero_que_cambia_es_el_de_la_cantidad():
    assert ge.con_cantidad("1 taza de arroz blanco", 1.5) == "1.5 taza de arroz blanco"
    assert ge.con_cantidad("Leche (≈1 vaso)", 2) == "Leche (≈2 vaso)"
    assert ge.con_cantidad("2 unidad de huevo hervido", 3) == "3 unidad de huevo hervido"


def test_subir_el_arroz_mueve_los_totales_lo_que_cambia_ese_renglon():
    n = ge.ajustar_cantidad(CENA, 0, 1.5, Catalogo())
    assert n["ingredients"][0] == "1.5 taza de arroz blanco"
    assert n["calories"] == round(600 + 205 * 0.5)
    assert n["carbs"] == round(70 + 10 * 0.5)
    assert n["ingredients"][1:] == CENA["ingredients"][1:]


def test_de_cuarto_en_cuarto_y_dentro_del_rango():
    assert ge.ajustar_cantidad(CENA, 1, 2.6, Catalogo())["cantidad"] == 2.5
    assert ge.ajustar_cantidad(CENA, 1, 0, Catalogo())["cantidad"] == ge.CANTIDAD_MIN
    assert ge.ajustar_cantidad(CENA, 3, 2, Catalogo()) is None       # leche 2 %
    assert ge.ajustar_cantidad(CENA, 4, 2, Catalogo()) is None       # tiene gramos: va por `ajustar`


def test_la_ficha_ofrece_la_cantidad_donde_no_hay_gramos():
    from ficha_comida import desglose_de_ingredientes
    d = desglose_de_ingredientes(CENA["ingredients"], 600, Catalogo())
    assert d["lineas"][0]["cantidad"] == {"valor": 1.0, "unidad": "taza"}
    assert "cantidad" not in d["lineas"][3]
    assert "cantidad" not in d["lineas"][4], "el pollo tiene gramos: no va por cantidad"


def test_el_endpoint_acepta_cantidad_o_gramos():
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert "cantidad: Optional[float] = Field(default=None, ge=0.25, le=30)" in src
    assert "ajustar_cantidad({**row, \"ingredients\": ings}, payload.indice, payload.cantidad, _ndb)" in src
