"""[P1-FALLBACK-BAND-AWARE · 2026-06-28] El fallback de emergencia matemático leía target con `.get('target_calories',
2000)` — que devuelve None si la key existe con None (TypeError downstream) o pasa valores patológicos (0, 50000). Fix
DEFENSIVO: sanear a default 2000 si ausente/None/0/fuera de [1200,3500]. NO se escala hacia banda a propósito — un plan de
contingencia ligeramente bajo banda es el lado SEGURO post-bariátrico (review clínica adversaria rechazó forzarlo arriba).
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_SRC = (Path(g.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _fb_total(target):
    nutr = {"target_calories": target, "macros": {"protein_g": 90, "carbs_g": 180, "fats_g": 60}}
    fb = g._get_extreme_fallback_plan(nutr, "maintenance", num_days=1)
    return (fb.get("days") or [{}])[0].get("total_calories")


def test_pathological_targets_default_safely():
    for bad in (None, 0, 50000, -100):
        t = _fb_total(bad)
        assert t == 2000, f"target {bad} debe defaultear a 2000, dio {t}"


def test_valid_target_preserved():
    assert _fb_total(1800) == 1800
    assert _fb_total(2100) == 2100


def test_floor_protects_hypocaloric():
    # un target absurdamente bajo (800) cae fuera de [1200,3500] → default 2000 (no hipocalórico peligroso)
    assert _fb_total(800) == 2000


def test_anchor_present():
    assert "P1-FALLBACK-BAND-AWARE" in _SRC
    # no escala hacia banda (rechazado clínicamente): no debe haber un escalado-up del fallback
    assert "1200 <= float(target_cal) <= 3500" in _SRC
