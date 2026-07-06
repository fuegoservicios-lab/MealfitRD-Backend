"""[P2-NUTBUTTER-MICRO-BUMP · 2026-07-06] Review #14: "¼ cdta de mantequilla de maní" (~1g) es
inservible como untar/topping. Las cremas de fruto seco están en la exención de condimentos del
shrink-floor ("mantequilla"), así que sobrevivían. Micro-bump análogo al del aceite: nut butter
< 1 cdta → mínimo servible 1 cdta (5g), ANTES de la exención.
"""
import pytest

import graph_orchestrator as go


@pytest.fixture(autouse=True)
def _stub_truthup(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda m, db: None)


def _days(pb_line):
    return [{"day": 1, "meals": [{
        "name": "Panqueques de Avena",
        "ingredients": ["30 g de avena", pb_line, "1 huevo"],
        "ingredients_raw": ["30 g de avena", pb_line, "1 huevo"],
        "recipe": ["Montaje: sirve."],
    }]}]


def test_quarter_cdta_pb_bumped_to_one():
    days = _days("¼ cdta de mantequilla de maní (1g)")
    go._floor_subservible_portions(days, db=object())
    ings = days[0]["meals"][0]["ingredients"]
    pb = next(s for s in ings if "man" in s.lower() and "avena" not in s.lower())
    assert pb.startswith("1 cdta"), f"bump al mínimo servible: {pb}"
    assert "(5g)" in pb, f"hint re-escalado a 5g: {pb}"
    # lockstep raw
    assert any(s.startswith("1 cdta") and "man" in s.lower()
               for s in days[0]["meals"][0]["ingredients_raw"])


def test_half_cdta_pb_bumped():
    days = _days("½ cdta de crema de maní")
    go._floor_subservible_portions(days, db=object())
    assert any(s.startswith("1 cdta") and "man" in s.lower()
               for s in days[0]["meals"][0]["ingredients"])


def test_servible_pb_unchanged():
    days = _days("1 cda de mantequilla de maní (15g)")
    go._floor_subservible_portions(days, db=object())
    assert any("1 cda de mantequilla de maní" in s for s in days[0]["meals"][0]["ingredients"]), (
        "1 cda ya es servible → intacta"
    )


def test_almond_butter_also_bumped():
    days = _days("¼ cdta de mantequilla de almendra")
    go._floor_subservible_portions(days, db=object())
    assert any(s.startswith("1 cdta") and "almendra" in s.lower()
               for s in days[0]["meals"][0]["ingredients"])


def test_marker_anchored():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P2-NUTBUTTER-MICRO-BUMP" in src
