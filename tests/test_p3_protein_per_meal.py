"""[P3-PROTEIN-PER-MEAL · 2026-06-13] Ancla la regla de composición: el prompt del
day_generator (vía preferences.py) DEBE exigir proteína real en las 4 comidas.

Razón: test E2E live (2026-06-13) reveló que el déficit sistémico de proteína es de
COMPOSICIÓN — el prompt trataba las opciones de proteína livianas (huevos, yogurt,
queso) como diversificación OPCIONAL para desayuno/merienda, así el modelo producía
desayunos de solo mangú (2g proteína) y meriendas de solo fruta (0g), dejando el día
corto del objetivo (Día1: 47g/154 = 31%). El solver (cerebro dividido) hace los macros
honestos+consistentes pero no puede crear proteína que no está en el plato → el lever
es la composición upstream.
"""
import os

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFS = os.path.join(BACKEND, "prompts", "preferences.py")


@pytest.fixture(scope="module")
def prefs_src():
    with open(PREFS, encoding="utf-8") as fh:
        return fh.read()


def test_protein_per_meal_rule_present(prefs_src):
    # La regla explícita de proteína en cada comida debe existir.
    assert "PROTEÍNA EN CADA COMIDA" in prefs_src
    assert "las 4 comidas" in prefs_src.replace("CUATRO comidas", "las 4 comidas") \
        or "CUATRO comidas" in prefs_src


def test_prohibits_protein_free_breakfast_and_snack(prefs_src):
    # Debe PROHIBIR explícitamente desayuno/merienda sin proteína (el modo de fallo real).
    low = prefs_src.lower()
    assert "prohibido" in low
    assert "mangú" in low or "mangu" in low  # ejemplo de desayuno de solo almidón
    assert "merienda de solo fruta" in low or "solo fruta" in low


def test_light_options_are_mandatory_not_optional(prefs_src):
    # Las opciones livianas dejan de ser "para diversificar" (opcional) y pasan a OBLIGATORIAS.
    assert "OBLIGATORIA" in prefs_src
    # Las fuentes de proteína livianas siguen listadas.
    for src_protein in ["Huevos", "Yogurt griego", "Queso fresco"]:
        assert src_protein in prefs_src
