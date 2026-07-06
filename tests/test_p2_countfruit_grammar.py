"""[P2-COUNTFRUIT-GRAMMAR + P2-MANI-GENDER-STEP · 2026-07-06] Cosméticos de gramática del review
#14 (plan 17c3fa8f):

1. "1 ciruelas (50 g)" — el singularizador de "1 <plural>" solo cubría UNIDADES (cdas/tazas),
   no frutas contables. Añadidas ciruela/uva/durazno/mandarina/chinola/guineo/melocotón/higo
   con concordancia bidireccional.
2. "Tuesta ligeramente maní fileteadas" — maní es masc mass-noun; el display de ingredientes ya
   corregía el lead, pero la PROSA de los pasos no. Fix de género en el boundary.
"""
import pytest

from humanize_ingredients import _prettify_quantity_display as p
import graph_orchestrator as go


# ───────────── 1. concordancia frutas contables ─────────────

@pytest.mark.parametrize("inp,expected", [
    ("1 ciruelas (50 g)", "1 ciruela (50 g)"),
    ("1 duraznos", "1 durazno"),
    ("1 uvas", "1 uva"),
    ("1 mandarinas", "1 mandarina"),
    ("1 higos", "1 higo"),
])
def test_countfruit_singular_after_one(inp, expected):
    assert p(inp) == expected


@pytest.mark.parametrize("inp,expected", [
    ("2 ciruela", "2 ciruelas"),
    ("3 mandarina", "3 mandarinas"),
    ("2 durazno", "2 duraznos"),
])
def test_countfruit_plural_after_many(inp, expected):
    assert p(inp) == expected


def test_countfruit_one_singular_unchanged():
    assert p("1 ciruela") == "1 ciruela"
    assert p("2 ciruelas") == "2 ciruelas"


# ───────────── 2. maní género en pasos ─────────────

def _mani_fix(step):
    return go._MANI_GENDER_STEP_RX.sub(lambda m: f"{m.group(1)} {m.group(2)}o", step)


def test_mani_fileteadas_in_step():
    assert _mani_fix("Tuesta ligeramente maní fileteadas por 1 minuto") == \
        "Tuesta ligeramente maní fileteado por 1 minuto"


def test_mani_other_cooking_adjectives():
    assert _mani_fix("Decora con maní tostadas") == "Decora con maní tostado"
    assert _mani_fix("añade maní picada") == "añade maní picado"


def test_mani_already_correct_unchanged():
    s = "Espolvorea el maní fileteado por encima."
    assert _mani_fix(s) == s


def test_non_mani_food_untouched():
    # el fix es específico a maní; no toca otros alimentos con adjetivo fem legítimo.
    s = "Corta la cebolla picada finamente."
    assert _mani_fix(s) == s
