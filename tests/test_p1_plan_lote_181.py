# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-181 · 2026-09-23] Lo último que se escribe en una línea es lo que el usuario lee.

Métrica sobre los planes REALES de la batería (rd9 + rd10, 817 líneas): 37 con el alimento en mayúscula a media frase,
4 «0.5 pepino», 11 especias contadas sin unidad, 2 «½ pizca», «0.06 g de semillas de chía», «1 pechugas de pollo». El
pulido de frontera corría antes de la identidad y del contrato de receta, que reescriben líneas. Además, un «Majarete»
salió con 10 g de maíz y 5 ml de leche (el nombre no los nombra) y con mozzarella del cerrador (no contaba como dulce)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pulido_lineas as pl  # noqa: E402


@pytest.mark.parametrize("antes,despues", [
    ("⅓ taza de Yogurt natural entero", "⅓ taza de yogurt natural entero"),
    ("25 g de Harina de trigo", "25 g de harina de trigo"),
    ("25 g de Queso blanco bajo en sodio", "25 g de queso blanco bajo en sodio"),
    ("1 cdta (4 g) de Canela en polvo", "1 cdta (4 g) de canela en polvo"),
    ("1 Lechosa mediana", "1 lechosa mediana"),
    ("½ pizca de canela en polvo", "1 pizca de canela en polvo"),
    ("1 orégano dominicano", "Orégano dominicano al gusto"),
    ("1 canela en polvo", "Canela en polvo al gusto"),
    ("1 sal al gusto", "Sal al gusto"),
    ("0.06 g de semillas de chía", "1 pizca de semillas de chía"),
    ("1 pechugas de pollo (≈134 g)", "1 pechuga de pollo (≈134 g)"),
    ("½ tomates", "½ tomate"),
])
def test_las_lineas_reales_de_la_bateria(antes, despues):
    assert pl.pulir_linea(antes) == despues
    assert pl.pulir_linea(despues) == despues, "idempotente"


@pytest.mark.parametrize("igual", [
    "1 taza de Corn Flakes",            # marca: otra mayúscula detrás
    "1½ pechugas de pollo",
    "1 rama de canela", "1 hoja de laurel", "2 pimientos",
    "0.4 g de aguacate",                # migaja que no es semilla ni polvo: no se inventa una «pizca de aguacate»
    "Orégano dominicano al gusto", "3 huevos",
])
def test_lo_que_no_reconoce_queda_igual(igual):
    assert pl.pulir_linea(igual) == igual


def test_el_plan_solo_cambia_el_display_y_conserva_las_longitudes():
    raw = ["0.06 g de semillas de chía", "0.33 taza de Yogurt natural entero", "1 orégano dominicano"]
    comida = {"meal": "Desayuno", "name": "Avena cremosa", "ingredients": list(raw), "ingredients_raw": list(raw),
              "_display": {"en-US": {}}}
    plan = {"days": [{"day": 1, "meals": [comida]}]}
    assert pl.pulir_plan(plan) >= 3
    assert comida["ingredients"] == ["1 pizca de semillas de chía", "⅓ taza de yogurt natural entero",
                                     "Orégano dominicano al gusto"]
    assert comida["ingredients_raw"] == raw, "la compra y los macros leen el raw"
    assert "_display" not in comida
    assert pl.pulir_plan(plan) == 0, "idempotente"


def test_el_pulido_va_despues_del_contrato_y_antes_de_restaurar_los_congelados():
    body = (_BACKEND / "db_plans.py").read_text(encoding="utf-8").split("def _finalize_plan_data_for_insert(")[1].split("\ndef ")[0]
    i_contrato = body.index("_rfc_tail_out = _rfc_tail(")
    i_pulido = body.index("_pl_tail.pulir_plan(_pd)")
    i_fin = body.index("_rsd.finish(_rsd_ctx, _pd)")
    i_frz = body.index("_rpd_frz(_pd, _frozen_token)")
    assert i_contrato < i_pulido < i_fin < i_frz
    assert "P1-PLAN-LOTE-181-PULIDO-COLA" in body


def test_la_base_implicita_del_majarete_cuenta_como_nombrada():
    import identidad_plato as idp
    nombre = "Majarete ligero de coco y canela"
    assert idp.nombrada_en_el_nombre(nombre, "10 g de Maíz dulce en granos")
    assert idp.nombrada_en_el_nombre(nombre, "5 ml de leche descremada")
    assert idp.nombrada_en_el_nombre("Mangú con huevo", "1 plátano verde")
    assert not idp.nombrada_en_el_nombre("Ensalada de coco", "10 g de maíz dulce"), "sólo los platos de la tabla"


def test_los_postres_criollos_son_dulces():
    import graph_orchestrator as go
    from constants import strip_accents
    for nombre in ("Majarete ligero de coco y canela", "Flan de coco", "Arroz con leche"):
        assert go._is_sweet_meal({"name": nombre, "ingredients": []}, strip_accents), nombre


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 181 and m.group(2) >= "2026-09-23"
