# -*- coding: utf-8 -*-
"""[P1-SINGLE-TRIP-BADGES · 2026-09-05] Captura del dueño (plan a2b40e4e, ciclo 15 días, «solo la compra grande», sin
congelador): la lista decía «alcanza ~5 de 7 días — recompra» y «no recompres cada semana» bajo un encabezado que ya
avisaba «UNA SOLA COMPRA». Dos defectos: el ciclo real no llegaba a la nota (contaba sobre 7) y el copy invitaba a
recomprar a quien va UNA vez."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import shopping_calculator as sc  # noqa: E402


def teardown_function(_):
    sc.set_single_trip_notes(False)


def test_single_trip_copy_never_invites_to_repurchase():
    sc.set_single_trip_notes(True)
    assert sc._single_trip_notes_on() is True
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert '"consúmelo en esos días" if _single_trip_notes_on() else "no recompres cada semana"' in src
    assert '_tail = ("consúmelo en esos primeros días" if _single_trip_notes_on() else "recompra")' in src


def test_cycle_days_reaches_the_main_aggregate_pass():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    i = src.index("res = aggregate_and_deduct_shopping_list(all_ingredients, items_to_deduct")
    assert "cycle_days=_cycle_days_eff" in src[i:i + 700], "la pasada principal pasa el ciclo real"


def test_context_flag_defaults_off_and_toggles():
    assert sc._single_trip_notes_on() is False
    sc.set_single_trip_notes(True)
    assert sc._single_trip_notes_on() is True
    sc.set_single_trip_notes(False)
    assert sc._single_trip_notes_on() is False


def test_policy_derivation_shape():
    """El bloque lee la política del plan: ciclo del shopping y ausencia de reposición."""
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert '_shop_pol.get("main_cycle_days")' in src and 'not _shop_pol.get("fresh_topup_days")' in src
    assert "set_single_trip_notes(_single_trip_eff)" in src
