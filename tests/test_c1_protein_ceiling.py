"""[C1-PROTEIN-CEILING · 2026-06-13] Techo clínico de proteína por kg de peso corporal.

El split por % de calorías (30% para gain_muscle) daba 2.8+ g/kg en personas livianas con
TDEE alto (owner: 54kg, 2050kcal → 154g = 2.8g/kg, inalcanzable). El techo (default 2.2 g/kg,
ISSN) lo capea y redistribuye las calorías liberadas a carbohidratos.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nutrition_calculator import calculate_macros, _protein_ceiling_g_per_kg


def test_caps_protein_for_light_person():
    # Owner: 54kg, 2050 kcal, gain_muscle (30% prot = 154g = 2.8g/kg) → capeado a 2.2g/kg.
    m = calculate_macros(2050, "gain_muscle", weight_kg=54)
    assert m["protein_g"] == round(2.2 * 54)  # ≈ 119g
    assert m["protein_g"] / 54 <= 2.25  # ya no 2.8 g/kg
    # las calorías liberadas fueron a carbos (subieron vs sin techo).
    no_cap = calculate_macros(2050, "gain_muscle")
    assert m["carbs_g"] > no_cap["carbs_g"]
    assert m["fats_g"] == no_cap["fats_g"]  # grasa intacta
    # total calórico se preserva (±2%).
    total = m["protein_g"] * 4 + m["carbs_g"] * 4 + m["fats_g"] * 9
    assert abs(total - 2050) / 2050 < 0.02


def test_no_cap_when_below_ceiling():
    # Persona pesada (100kg): 30% de 2050 = 154g < 2.2×100=220 → sin cambio.
    m = calculate_macros(2050, "gain_muscle", weight_kg=100)
    assert m["protein_g"] == calculate_macros(2050, "gain_muscle")["protein_g"]


def test_backward_compat_no_weight():
    # Sin weight_kg → comportamiento legacy (% de calorías, sin techo).
    m = calculate_macros(2050, "gain_muscle")
    assert m["protein_g"] == round(2050 * 0.30 / 4)  # 154


def test_ceiling_knob_clamps():
    assert 1.6 <= _protein_ceiling_g_per_kg() <= 3.0


def test_str_fields_match_numeric():
    m = calculate_macros(2050, "gain_muscle", weight_kg=54)
    assert m["protein_str"] == f"{m['protein_g']}g"
    assert m["carbs_str"] == f"{m['carbs_g']}g"
