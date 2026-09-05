# -*- coding: utf-8 -*-
"""[P1-SLOT-PROTEIN-FLOOR-ALL · 2026-09-05] El cierre por franja del final del pipeline cubre TAMBIÉN almuerzo y cena:
plan vivo a2b40e4e entregado degradado con «Croquetas de papa y queso» de 19 g sobre un reparto de 47 y 104 g de 135 en
el día. Los casos funcionales viven en `test_p1_light_slot_protein_floor.py` (misma función); aquí se ancla el contrato."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def test_knob_on_by_default_and_wired():
    assert go.SLOT_PROTEIN_FLOOR_ALL_SLOTS is True
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "if not (_meal_slot_is_light(_m, _sa_lf) or SLOT_PROTEIN_FLOOR_ALL_SLOTS):" in src
    assert "P1-SLOT-PROTEIN-FLOOR-ALL" in src
