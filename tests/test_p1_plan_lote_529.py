# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-529 · 2026-09-27] «Maní tostado, 20 g» → «20 g de maní tostado» antes del motor de macros.

Batería real del dueño (compra única, día 28): la IA escribió el día entero con el alimento delante y la cantidad detrás;
el motor antepuso la suya y salieron «15 g de maní tostado, 20 g», «¼ taza de yogurt griego entero, 200 g», «2 limones,
0.5 unidad», «Apio, 1 taza picado» y el paso «Añade arroz blanco cocido, listo para calentar, 1½ tazas al lado».
"""
from __future__ import annotations

import pathlib
import sys

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import linea_invertida as li  # noqa: E402


def test_las_lineas_reales_del_dia_28():
    casos = {
        "Maní tostado, 20 g": "20 g de maní tostado",
        "Arroz blanco cocido, listo para calentar, 1½ tazas": "1½ tazas de arroz blanco cocido, listo para calentar",
        "Apio, 1 taza picado": "1 taza de apio picado",
        "Cebolla, ½ unidad picada": "½ cebolla picada",
        "Ajo, 1 diente picado": "1 diente de ajo picado",
        "Limones, 0.5 unidad": "½ limón",
        "Aceite de oliva, 1 cdta": "1 cdta de aceite de oliva",
        "Yogurt griego entero, 150 g": "150 g de yogurt griego entero",
        "Orégano dominicano, ½ cdta": "½ cdta de orégano dominicano",
        "Queso blanco fresco, 60 g": "60 g de queso blanco fresco",
        # del corpus guardado (pipeline_result): lo que el motor no llegó a prefijar
        "Limón, 1 cucharada de jugo": "1 cucharada de jugo de limón",
        "Hojas de menta, 2 unidades": "2 hojas de menta",
        "Agua fría, ½ taza": "½ taza de agua fría",
        "Orégano dominicano, 0.25 cucharadita al gusto": "¼ cucharadita de orégano dominicano al gusto",
    }
    for entrada, esperado in casos.items():
        assert li.enderezar(entrada) == esperado, (entrada, li.enderezar(entrada))


def test_lo_que_ya_empieza_por_cantidad_no_se_toca():
    for linea in ("150 g de pollo", "Sal al gusto", "Pimienta negra", "1 taza de arroz, cocido", "Tomate, picado",
                  "½ taza de yogurt griego natural, 170 g", "Corn Flakes, sin azúcar"):
        assert li.enderezar(linea) is None, linea


def test_la_lista_y_su_copia_en_los_pasos():
    days = [{"meals": [{
        "ingredients": ["Arroz blanco cocido, listo para calentar, 1½ tazas", "Yogurt griego natural, 170 g",
                        "Sal al gusto"],
        "recipe": ["Mise en place: mide ½ taza de yogurt griego natural, 170 g.",
                   "El Toque de Fuego: añade arroz blanco cocido, listo para calentar, 1½ tazas al lado para acompañar."]}]}]
    assert li.normaliza_dias(days) == 2
    m = days[0]["meals"][0]
    assert m["ingredients"] == ["1½ tazas de arroz blanco cocido, listo para calentar",
                                "170 g de yogurt griego natural", "Sal al gusto"], m["ingredients"]
    assert m["recipe"][0] == "Mise en place: mide ½ taza de yogurt griego natural.", m["recipe"][0]
    assert m["recipe"][1] == ("El Toque de Fuego: añade 1½ tazas de arroz blanco cocido, listo para calentar al lado para "
                              "acompañar."), m["recipe"][1]


def test_ancla_en_assemble():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("linea_invertida").normaliza_dias(result.get("days") or [])  # [P1-PLAN-LOTE-529]' in src
