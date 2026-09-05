# -*- coding: utf-8 -*-
"""[P1-ROUTINE-MONOTONY-CEILING · 2026-09-05] «Rutina» silenciaba TODAS las señales de repetición (P2-CRITIQUE-RESPECTS-
ROUTINE, decisión correcta: repetir es lo pedido). Pero el plan vivo a2b40e4e (rutina) llevaba avena en 8 de 12 comidas,
incluidas dos cenas, y se entregó degradado. Rutina tolera la repetición hasta un TECHO; por encima, la señal pasa."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import horizon  # noqa: E402

ROUTINE = {"recurrence": {"global_mode": "routine"}}
EXPLORE = {"recurrence": {"global_mode": "explore"}}
BALANCED = {"recurrence": {"global_mode": "balanced"}}


def _f(counts, eff=ROUTINE, enforced=True):
    return horizon.filter_repetition_counts_for_policy(counts, eff, enforced=enforced)


def test_routine_still_silences_normal_repetition():
    assert _f({"avena": 0.5, "yogur": 0.42}) == {}          # fracciones de comidas bajo el techo (0,6)
    assert _f({"avena": 3, "arroz": 2}) == {}                # días bajo el techo (4)


def test_routine_lets_the_monotony_through():
    assert _f({"avena": 0.67}) == {"avena": 0.67}            # 8 de 12 comidas: el caso vivo
    assert _f({"avena": 6}) == {"avena": 6}                  # 6 días
    mixed = _f({"avena": 0.67, "yogur": 0.4})
    assert mixed == {"avena": 0.67}, mixed


def test_other_modes_and_no_policy_are_untouched():
    assert _f({"avena": 0.5}, EXPLORE) == {"avena": 0.5}
    assert _f({"avena": 0.5}, ROUTINE, enforced=False) == {"avena": 0.5}
    assert _f({"avena": 0.5}, None) == {"avena": 0.5}
    assert _f({"avena": 0.5}, BALANCED) == {"avena": 0.5}    # equilibrada solo retira las anclas


def test_ceiling_helper():
    assert horizon._above_monotony_ceiling(0.67) and not horizon._above_monotony_ceiling(0.6)
    assert horizon._above_monotony_ceiling(5) and not horizon._above_monotony_ceiling(4)
    assert not horizon._above_monotony_ceiling("x") and not horizon._above_monotony_ceiling(None)
