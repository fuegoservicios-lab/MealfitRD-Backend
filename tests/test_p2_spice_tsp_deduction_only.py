"""[P2-SPICE-TSP-DEDUCTION-ONLY · 2026-09-02] El fallback «unidad de especia = cucharadita» es
OPT-IN y solo lo pide la deducción de la Nevera; el guard de coherencia (que usa el MISMO
convert_amount vía `from db_inventory import convert_amount as _conv`, P1-COHERENCE-GRAM-NORM)
sigue estricto.

Medido en prod: con el fallback global, los dos primeros planes tras el deploy fueron
RECHAZADOS por «COHERENCIA RECETAS LISTA: Orégano dominicano» (únicos bloqueos críticos por
alimento en 7 días) y forzaron un reintento completo cada uno. «1 cucharadita de orégano»
en gramos contra el sobre de 45 g de la lista es una divergencia inventada por el guard.

Tooltip-anchor: P2-SPICE-TSP-DEDUCTION-ONLY | spice_tsp=True solo en la deducción
"""
import logging
import re
from pathlib import Path

import pytest

import db_inventory

BACKEND = Path(__file__).resolve().parents[1]


def test_convert_amount_is_strict_by_default_for_spices(caplog, monkeypatch):
    monkeypatch.setenv("MEALFIT_CROSS_UNIT_CONVERSION_STRICT", "true")
    item = {"name": "Orégano dominicano", "density_g_per_cup": 24}
    with caplog.at_level(logging.WARNING, logger="db_inventory"):
        assert db_inventory.convert_amount(1.0, "unidad", "g", item) is None
    assert any("convert_amount" in r.getMessage() for r in caplog.records)


def test_convert_amount_opt_in_converts_spices():
    item = {"name": "Orégano dominicano", "density_g_per_cup": 24}
    assert db_inventory.convert_amount(1.0, "unidad", "g", item, spice_tsp=True) == pytest.approx(0.5)


def test_only_deduction_callers_opt_in():
    src = (BACKEND / "db_inventory.py").read_text(encoding="utf-8")
    optin = re.findall(r"convert_amount\(quantity, unit, current_unit, master_item, spice_tsp=True\)", src)
    assert len(optin) == 1, "solo add_or_update_inventory_item (la deducción real) pide el fallback; las 2 reservas de chunks conservan skip-on-incompatible"
    j = src.find("def add_or_update_inventory_item(")
    assert src.find("spice_tsp=True)", j) != -1 and src.find("spice_tsp=True)", j) < j + 6000
    guard = (BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "from db_inventory import convert_amount as _conv" in guard
    assert "spice_tsp" not in guard, "el guard de coherencia NO debe pedir el fallback de especias"
