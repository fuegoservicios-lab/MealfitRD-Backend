"""[P3-PKG-DAIRY-VEG · 2026-06-22] Pluralización de las unidades nuevas de envase
(tarro/barrita) introducidas por el batch de lácteos/grasas (mantequilla).

El resto del batch fue data en Neon (market_packages de mantequilla/queso cottage/parmesano/
espinacas). Este test ancla el único cambio de CÓDIGO: get_plural_unit ahora pluraliza
'tarro'→'tarros' y 'barrita'→'barritas' (antes mostraba "3 barrita"/"2 tarro").
"""
from __future__ import annotations

import pytest

from shopping_calculator import get_plural_unit, MARKET_MINIMUMS


@pytest.mark.parametrize("num,unit,expected", [
    (2, "tarro", "tarros"),
    (3, "barrita", "barritas"),
    (1, "tarro", "tarro"),      # singular sin cambio
    (1, "barrita", "barrita"),
    (2, "Tarro", "Tarros"),     # preserva capitalización
])
def test_tarro_barrita_pluralize(num, unit, expected):
    assert get_plural_unit(num, unit) == expected


def test_no_regression_existing_plurals():
    assert get_plural_unit(2, "paquete") == "paquetes"
    assert get_plural_unit(2, "pote") == "potes"
    assert get_plural_unit(2, "cartón (30 uds.)") == "cartones (30 uds.)"


def test_new_units_have_market_minimum():
    assert MARKET_MINIMUMS.get("tarro") == 1
    assert MARKET_MINIMUMS.get("barrita") == 1
