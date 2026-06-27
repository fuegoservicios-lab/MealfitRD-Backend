"""[P1-VARIETY-RENEWAL-NO-CYCLE-LOCK · 2026-06-27] El GROCERY CYCLE LOCK reutilizaba los ingredientes base del
ciclo de compras (quincenal/mensual) en cada renovación → el usuario veía SIEMPRE los mismos alimentos (solo
variaban los platos). Causa raíz de "renovar repite lo mismo". El owner priorizó variedad sobre reuso. Ahora
el lock está detrás de un knob (default OFF = variety-first).
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = (Path(__file__).resolve().parent.parent / "ai_helpers.py").read_text(encoding="utf-8")


def test_knob_defined_default_off():
    """El knob existe y default es False (variety-first)."""
    m = re.search(r'GROCERY_CYCLE_LOCK_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_GROCERY_CYCLE_LOCK"\s*,\s*(\w+)\s*\)', SRC)
    assert m, "falta el knob GROCERY_CYCLE_LOCK_ENABLED = _env_bool('MEALFIT_GROCERY_CYCLE_LOCK', ...)"
    assert m.group(1) == "False", f"el default debe ser False (variety-first), es {m.group(1)}"


def test_cycle_lock_gated_by_knob():
    """El bloqueo (cycle_locked=True + reuso de base_*) sólo ocurre si el knob está ON."""
    # la condición del lock incluye GROCERY_CYCLE_LOCK_ENABLED
    assert re.search(r"if\s+2\s*<=\s*days_elapsed\s*<\s*grocery_days\s+and\s+GROCERY_CYCLE_LOCK_ENABLED", SRC), \
        "la condición del cycle lock debe estar gateada por GROCERY_CYCLE_LOCK_ENABLED"
    # el marker está presente
    assert "P1-VARIETY-RENEWAL-NO-CYCLE-LOCK" in SRC


def test_variety_first_branch_exists():
    """Con lock OFF en Día 2..N debe haber rama variety-first (new_cycle_started)."""
    assert "Variety-first (lock OFF)" in SRC
