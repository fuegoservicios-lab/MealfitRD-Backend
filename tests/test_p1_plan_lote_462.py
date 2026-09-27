# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-462 · 2026-09-27] La pieza sin peso pasa a su duradero EN GRAMOS.

«1¼ filetes de pescado» → «1¼ de sardinas en lata»: el número suelto no dice cuánto. En el plan vivo del dueño (día 2) la
receta decía «(170 g)» y la lista leía «1¼» unidades; en el corpus forzado, «1 filete de pescado» terminaba como «1 sardina
en lata» (≈25 g en vez de 150) tras el pulido de líneas. Con el peso de la pieza que ya resuelve el catálogo, la línea sale
«190 g de sardinas en lata». Una MEDIDA («2 tazas», «1 cucharada») se conserva: el volumen del sustituto es el mismo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import graph_orchestrator as go  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "never"},
          "diet": {"type": "balanced", "allergies": []}}
_REQ = {"need_days": 11, "allow_frozen": False, "freezer_mode": "none"}


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


@pytest.fixture(autouse=True)
def _limpio():
    yield
    cu._MEMO.clear()


def test_cantidad_de_con_el_peso_de_la_pieza():
    assert cu.cantidad_de("1¼ filetes de pescado", 187.5) == "190 g de "
    assert cu.cantidad_de("2 tomates", 300.0) == "300 g de "
    assert cu.cantidad_de("1 cucharada de cilantro picado", 4.0) == "1 cucharada de ", "la medida manda"
    assert cu.cantidad_de("2 tazas de lechuga", 72.0) == "2 tazas de "
    assert cu.cantidad_de("¾ pechuga de pollo (≈150 g)", 999.0) == "150 g de ", "el peso declarado manda"
    assert cu.cantidad_de("1¼ filetes de pescado") == "1¼ de ", "sin resolvedor, como antes"


def test_sustituir_linea_pide_el_peso_solo_a_la_pieza():
    pedidas = []

    def gramos(t):
        pedidas.append(t)
        return 187.5
    r = cu.sustituir_linea("1¼ filetes de pescado", 10, _REQ, semilla=1, gramos_de=gramos)
    assert r and r[0] == "190 g de sardinas en lata", r
    r = cu.sustituir_linea("150 g de filete de pescado", 10, _REQ, semilla=1, gramos_de=gramos)
    assert r and r[0] == "150 g de sardinas en lata", r
    assert pedidas == ["1¼ filetes de pescado"], pedidas


def test_el_escudo_escribe_gramos(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    monkeypatch.setattr(go, "_resolve_line_food_grams", lambda line, cheap=False: ("filete de pescado blanco", 187.5))
    days = [{"day": i + 1, "meals": [{"meal": "Cena", "name": "x", "ingredients": ["1 taza de arroz"],
                                      "ingredients_raw": ["1 taza de arroz"]}]} for i in range(9)]
    days.append({"day": 10, "meals": [{"meal": "Almuerzo", "name": "Pescado con arroz",
                                       "ingredients": ["1¼ filetes de pescado", "1 taza de arroz"],
                                       "ingredients_raw": ["1¼ filetes de pescado", "1 taza de arroz"]}]})
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced", contexto={})
    linea = days[-1]["meals"][0]["ingredients"][0]
    assert linea.startswith("190 g de "), linea


def test_la_proyeccion_de_la_lista_tambien(monkeypatch):
    monkeypatch.setattr(cu, "_gramos_de_linea", lambda t: 150.0 if "filete" in t else None)
    p = {"total_days_requested": 30, "_plan_policy": {"effective": SINGLE},
         "days": [{"day": n, "meals": [{"meal": "Almuerzo", "name": "Pescado",
                                        "ingredients": ["1 filete de pescado"],
                                        "ingredients_raw": ["1 filete de pescado"]}]} for n in (1, 2, 3)]}
    dias = cu.dias_de_la_compra(p, p["days"])
    resto = [x for d in dias[7:] for m in d["meals"] for x in m["ingredients_raw"]]
    assert resto and all(x.startswith("150 g de ") for x in resto), resto[:5]
