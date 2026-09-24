# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-194 · 2026-09-24] Los cortes de una pieza se leen como FRACCIÓN de la unidad, no como piezas enteras.

Traspaso de la sesión del lote 192 (medido el 23-sep con el resolutor real): el diario del dueño guardaba «8 rodajas de
plátano maduro hervido» y lo contaba como 8 plátanos enteros (2.240 g; K 9.815 mg, vit C 455, fibra 49,7). «rodajas» no
era unidad: `_split_qty_unit_name` la pegaba al nombre, el nombre resolvía igual y `to_grams` multiplicaba por el peso de
la pieza. Mismo fallo en `shopping_calculator._parse_quantity` (la Nevera)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.mark.parametrize("linea, qty, nombre", [
    ("8 rodajas de plátano maduro hervido", 1.0, "plátano maduro hervido"),
    ("2 lascas de aguacate", 0.25, "aguacate"),
    ("5 tajadas de plátano maduro frito", 1.0, "plátano maduro frito"),
    ("4 gajos de naranja", 0.4, "naranja"),
    ("½ rodaja de tomate mediano", 0.0625, "tomate mediano"),
])
def test_el_corte_es_una_fraccion_de_la_unidad(linea, qty, nombre):
    from nutrition_db import _split_qty_unit_name
    q, u, n = _split_qty_unit_name(linea)
    assert (round(q, 4), u, n) == (round(qty, 4), "unidad", nombre)


@pytest.mark.parametrize("linea, esperado", [
    ("2 pechugas de pollo", (2.0, "unidad", "pechugas de pollo")),
    ("3 dientes de ajo", (3.0, "dientes", "ajo")),
    ("1 rebanada de pan integral", (1.0, "rebanada", "pan integral")),
    ("2 trozos de yuca", (2.0, "unidad", "trozos de yuca")),
])
def test_lo_demas_no_cambia(linea, esperado):
    from nutrition_db import _split_qty_unit_name
    assert _split_qty_unit_name(linea) == esperado


class _Info:
    name, kcal, protein, carbs, fats = "Plátano maduro", 122.0, 1.3, 32.0, 0.4
    density_g_per_unit, density_g_per_cup, container_weight_g = 280.0, None, None


def test_ocho_rodajas_pesan_un_platano_no_ocho():
    import nutrition_db
    db = nutrition_db.IngredientNutritionDB(rows=[])
    db.lookup = lambda name: _Info()
    assert db.grams_from_ingredient_string("8 rodajas de plátano maduro hervido") == pytest.approx(280.0)
    assert db.grams_from_ingredient_string("2 plátanos maduros") == pytest.approx(560.0), "la pieza entera sigue igual"


def test_la_nevera_tambien():
    from shopping_calculator import _parse_quantity
    q, u, n = _parse_quantity("8 rodajas de plátano maduro", apply_yield_multiplier=False)
    assert (round(q, 4), u) == (1.0, "unidad") and "rodaja" not in n.lower() and "platano" in n.lower().replace("á", "a")
    q2, u2, _ = _parse_quantity("2 lascas de aguacate", apply_yield_multiplier=False)
    assert (round(q2, 4), u2) == (0.25, "unidad")


def test_knob_apaga(monkeypatch):
    monkeypatch.setenv("MEALFIT_PIECE_FRACTIONS", "false")
    from nutrition_db import _split_qty_unit_name
    assert _split_qty_unit_name("8 rodajas de plátano")[0] == 8.0


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 194 and m.group(2) >= "2026-09-24"
