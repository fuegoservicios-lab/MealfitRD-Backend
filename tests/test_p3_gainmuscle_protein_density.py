"""[P3-GAINMUSCLE-PROTEIN-DENSITY · 2026-06-23] Para gain_muscle, el ANTI MODE-COLLAPSE garantiza
proteínas PRINCIPALES de alta densidad (animal) — reemplaza leguminosas/ricotta/cottage/crema
elegidas como main por alta densidad.

Bug en vivo (corr=f36bd39f): el selector eligió Queso Ricotta + Habichuelas + Gandules (3 de baja
densidad) → días bajo el piso de proteína (124g) → el LLM rellenó con huevo → chocó con el cap de
huevo Y el piso → 3 rechazos → entrega DEGRADADA. La garantía corta esa cascada.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent

# Leguminosas + lácteos blandos que NO deben ser proteína PRINCIPAL para gain_muscle.
_LOW_DENSITY = {
    "habichuelas rojas", "habichuelas negras", "habichuelas blancas", "gandules",
    "lentejas", "garbanzos", "queso ricotta", "queso cottage", "queso crema",
    "'yogurt'",  # regular (match exacto entre comillas para no matchear 'yogurt griego')
}


def test_marker_and_knob_present():
    src = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert "P3-GAINMUSCLE-PROTEIN-DENSITY" in src
    assert "MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN" in src
    assert "_LOW_DENSITY_AS_MAIN" in src


@pytest.mark.parametrize("trial", range(15))
def test_gainmuscle_no_low_density_main_protein(trial, caplog):
    """En 15 corridas, gain_muscle NUNCA debe elegir una leguminosa/ricotta como proteína main."""
    from ai_helpers import get_deterministic_variety_prompt
    with caplog.at_level(logging.INFO):
        get_deterministic_variety_prompt("", {"mainGoal": "gain_muscle"}, user_id=None)
    # Extraer la línea "Proteínas elegidas para 3 días: [...]"
    chosen = None
    for rec in caplog.records:
        m = re.search(r"Prote[ií]nas elegidas.*?:\s*(\[.*?\])", rec.message)
        if m:
            chosen = m.group(1).lower()
    assert chosen is not None, "no se encontró el log de proteínas elegidas"
    for low in _LOW_DENSITY:
        assert low not in chosen, f"gain_muscle eligió '{low}' como proteína principal (baja densidad): {chosen}"
