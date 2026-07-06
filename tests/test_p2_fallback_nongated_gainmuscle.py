"""[P2-FALLBACK-NONGATED-GAINMUSCLE · 2026-07-06] Review de logs en vivo (plan del owner 4339544f,
perfil ganancia muscular, degradado): el fallback no-gated de P1-PROTEIN-REPEAT-FALLBACK-NONGATED
(legumbre/queso cuando la escalera de carnes se agota) disparó "Habichuela" en un perfil gain_muscle
con DÉFICIT DE PROTEÍNA. Swapear proteína animal → legumbre BAJA la densidad proteica, justo lo
contrario de lo que exige ganancia muscular. Fix: para gain_muscle NO se añade el fallback — la
escalera de CARNES sigue; si se agota, tgt None → el retry regenera con una proteína ANIMAL
distinta (mejor outcome muscular que un legumbre).
"""
import pytest

import graph_orchestrator as go


class _StubDB:
    def macros_from_ingredient_string(self, s):
        return None


@pytest.fixture()
def _go(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda m, db: None)
    return go


def _pollo_x2_exhausted():
    # pollo×2 + tilapia (pescado tomado); pavo/cerdo/res disliked → escalera de carnes agotada.
    return [{"day": 1, "meals": [
        {"name": "Tilapia Salteada", "ingredients": ["150 g de filete de tilapia"],
         "ingredients_raw": ["150 g de filete de tilapia"], "recipe": ["x"]},
        {"name": "Pechuga de pollo al Vapor", "ingredients": ["150 g de pechuga de pollo"],
         "ingredients_raw": ["150 g de pechuga de pollo"], "recipe": ["x"]},
        {"name": "Pollo en Salsa", "ingredients": ["150 g de pechuga de pollo"],
         "ingredients_raw": ["150 g de pechuga de pollo"], "recipe": ["x"]},
    ]}]


def test_gain_muscle_skips_legume_fallback(_go):
    fd = {"dislikes": ["pavo", "cerdo", "res"], "mainGoal": "gain_muscle"}
    n = _go._protein_repeat_autofix(_pollo_x2_exhausted(), fd, db=_StubDB())
    assert n == 0, ("gain_muscle + escalera agotada → NO swap a legumbre (baja proteína); "
                    "tgt None → el retry regenera con proteína animal distinta")


def test_non_gain_muscle_still_uses_fallback(_go):
    fd = {"dislikes": ["pavo", "cerdo", "res"], "mainGoal": "lose_weight"}
    days = _pollo_x2_exhausted()
    n = _go._protein_repeat_autofix(days, fd, db=_StubDB())
    assert n == 1, "en pérdida de peso el fallback no-gated (legumbre/queso) SÍ aplica"
    _swapped = days[0]["meals"][2]
    assert any(t in _swapped["name"].lower() for t in ("habichuela", "lenteja", "queso"))


def test_gain_muscle_still_swaps_when_heavy_available(_go):
    # gain_muscle NO bloquea el swap a otra CARNE (solo bloquea el fallback legumbre).
    days = [{"day": 1, "meals": [
        {"name": "Pechuga de pollo", "ingredients": ["150 g de pechuga de pollo"],
         "ingredients_raw": ["150 g de pechuga de pollo"], "recipe": ["x"]},
        {"name": "Pollo Guisado", "ingredients": ["150 g de pechuga de pollo"],
         "ingredients_raw": ["150 g de pechuga de pollo"], "recipe": ["x"]},
    ]}]
    n = _go._protein_repeat_autofix(days, {"mainGoal": "gain_muscle"}, db=_StubDB())
    assert n == 1 and days[0]["meals"][1]["_protein_autofix_applied"] == "pollo->pavo", (
        "pavo disponible → swap a carne (gain_muscle solo excluye el fallback legumbre)"
    )


def test_marker_anchored():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P2-FALLBACK-NONGATED-GAINMUSCLE" in src
    assert "P2-CROSSDAY-DIVERSIFY-ADJ" in src
