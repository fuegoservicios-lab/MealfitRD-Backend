"""[P2-F3-FAMILY-INJECT-EMPTY-PANTRY · 2026-09-03] Con la Nevera vacía, la familia programada se cubre.

Medido (plan 8f364c87 del dueño, ganancia muscular 135 g de proteína, Nevera 0): el sorteo ponderado
del seeder dio «Queso parmesano / Habas / Queso de hoja» y luego «Costilla / Guisantes / Frijoles»;
el revisor rechazó los intentos 1 y 2 por déficit de proteína (día 3: 91 g de 135). La rebanada
pedía Pollo / Pescado / Huevo, pero el pool no tenía ningún miembro y el fallback conservaba la
elección del seeder. Ahora, SOLO cuando la Nevera está por debajo de `PANTRY_GUARD_MIN_ITEMS` (el
guard de despensa no aplica), la familia ausente se inyecta con su representante canónico.
"""
from __future__ import annotations

from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
h = pytest.importorskip("horizon")

SL = {"days": [{"protein": "Pollo"}, {"protein": "Pescado"}, {"protein": "Huevo"}], "recurrence": {"global_mode": "balanced"}}
WEAK = ["Queso parmesano", "Habas", "Queso de hoja"]


def test_inject_uses_catalog_representatives_when_pool_lacks_the_family():
    assert h.apply_slice_to_seeder_pools(SL, WEAK, WEAK, days=3, inject_missing=True) == [
        "Pechuga de pollo", "Filete de pescado blanco", "Huevo"]
    assert h.family_representative("Pescado") == "Filete de pescado blanco"
    assert h.family_representative("Atún") == "Atún en agua" and h.family_representative("nada") is None


def test_without_inject_the_seeder_choice_is_kept_and_pool_members_win():
    assert h.apply_slice_to_seeder_pools(SL, WEAK, WEAK, days=3, inject_missing=False) == WEAK
    pool = ["Habas", "Sardinas en lata", "Pechuga de pollo", "Huevos"]
    out = h.apply_slice_to_seeder_pools(SL, pool, pool, days=3, inject_missing=True)
    assert out == ["Pechuga de pollo", "Sardinas en lata", "Huevos"]   # miembros reales del pool antes que inyectar


def test_seeder_gates_injection_on_the_pantry_guard_threshold():
    src = (BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert "from constants import PANTRY_GUARD_MIN_ITEMS as _pgmi_f3" in src
    assert '_bp_inject = len(form_data.get("current_pantry_ingredients") or []) < int(_pgmi_f3 or 0)' in src
    assert "inject_missing=_bp_inject" in src


def test_marker_present():
    assert "P2-F3-FAMILY-INJECT-EMPTY-PANTRY" in (BACKEND / "horizon.py").read_text(encoding="utf-8")
