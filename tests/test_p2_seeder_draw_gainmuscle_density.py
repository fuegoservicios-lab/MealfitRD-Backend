"""[P2-SEEDER-DRAW-GAINMUSCLE-DENSITY · 2026-09-03] Ganancia muscular: ni leguminosa ni queso como MAIN.

Medido (plan 8f364c87, 135 g de proteína, Nevera vacía): el sorteo ponderado del seeder dio
«Queso parmesano / Habas / Queso de hoja» y luego «Costilla de cerdo / Guisantes secos / Frijoles
pintos» como proteína PRINCIPAL del día; el revisor rechazó 2 intentos por déficit (día 3: 91 g).
`_LOW_DENSITY_AS_MAIN` era exact-match sobre nombres DO y los quesos duros sólo contaban para
bariátrica: el catálogo de países (F2) trajo leguminosas con otros nombres. Ahora, para
`gain_muscle`, la misma regla que reemplaza mains de baja densidad cubre leguminosas por token y
cualquier queso — en el sorteo Y en el filtro de la nevera (un solo criterio, dos call sites).
"""
from __future__ import annotations

from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
SRC = (BACKEND / "ai_helpers.py").read_text(encoding="utf-8")


def _ah():
    return pytest.importorskip("ai_helpers")


@pytest.mark.parametrize("name", ["Habas", "Guisantes secos", "Frijoles pintos", "Alubias blancas",
                                  "Queso de hoja", "Queso parmesano", "Queso gouda", "Lentejas"])
def test_gain_muscle_excludes_legumes_and_cheese_as_main(name, monkeypatch):
    ah = _ah()
    monkeypatch.delenv("MEALFIT_GAINMUSCLE_MAIN_DENSITY_STRICT", raising=False)
    assert ah._is_low_density_main(name, False, gain_muscle=True) is True


@pytest.mark.parametrize("name", ["Pechuga de pollo", "Filete de pescado blanco", "Huevo", "Carne de res",
                                  "Costilla de cerdo", "Atún en agua", "Sardinas en lata", "Camarones"])
def test_gain_muscle_keeps_dense_mains(name):
    ah = _ah()
    assert ah._is_low_density_main(name, False, gain_muscle=True) is False


def test_other_goals_are_untouched_and_knob_turns_it_off(monkeypatch):
    ah = _ah()
    # sin gain_muscle: la regla previa (exact-match DO) sigue igual
    assert ah._is_low_density_main("Habas", False) is False
    assert ah._is_low_density_main("Lentejas", False) is True          # ya estaba en _LOW_DENSITY_AS_MAIN
    assert ah._is_low_density_main("Queso de hoja", False) is False     # sólo bariátrica (y ahora gain_muscle)
    assert ah._is_low_density_main("Queso de hoja", True) is True
    monkeypatch.setenv("MEALFIT_GAINMUSCLE_MAIN_DENSITY_STRICT", "false")
    assert ah._is_low_density_main("Habas", False, gain_muscle=True) is False


def test_both_call_sites_pass_the_goal():
    """Parser: el sorteo (closure) y el filtro de la nevera usan el MISMO criterio con el goal."""
    assert 'return _is_low_density_main(_p, _is_bariatric, gain_muscle=(_main_goal == "gain_muscle"))' in SRC
    assert "_is_low_density_main(x, is_bariatric, gain_muscle=gain_muscle)" in SRC
    assert 'gain_muscle=(_main_goal == "gain_muscle"))' in SRC
    assert SRC.count("_GAINMUSCLE_LEGUME_TOKENS") >= 2


def test_marker_present():
    assert "P2-SEEDER-DRAW-GAINMUSCLE-DENSITY" in SRC
