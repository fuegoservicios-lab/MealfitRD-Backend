"""[P1-CLOSER-LEGUME-WORDING · 2026-07-08] Wording coherente del paso 💪 del closer para legumbres.

Review de recetas en vivo (plan 9b6a43f6): "💪 Cocina guisantes secos a la plancha o hervido y sírvelo
como proteína del plato". Una legumbre seca (chícharo/guisante/lenteja/garbanzo/habichuela) se HIERVE —
no se hace a la plancha — y en un plato dominicano es acompañante, no "la proteína del plato". El default
del SSOT `_closer_protein_step_text` (pescado/pollo → "a la plancha") no aplica a legumbres.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_and_knob_present():
    assert "P1-CLOSER-LEGUME-WORDING" in _GO
    assert 'PROTEIN_STEP_LEGUME_WORDING = _env_bool("MEALFIT_PROTEIN_STEP_LEGUME_WORDING", True)' in _GO


@pytest.mark.parametrize("nm", ["guisantes secos", "chícharos", "lentejas", "garbanzos", "habichuelas rojas"])
def test_legume_uses_boiled_wording(nm):
    import graph_orchestrator as g
    txt = g._closer_protein_step_text(nm, no_cook=False)
    assert "en agua" in txt and "a la plancha" not in txt, f"legumbre debe hervirse, no a la plancha: {txt!r}"
    assert "sírvelo como proteína del plato" not in txt, f"legumbre no es 'la proteína del plato': {txt!r}"


def test_the_exact_live_defect_fixed():
    """El caso literal del review: 'guisantes secos' ya no cae en el default 'a la plancha'."""
    import graph_orchestrator as g
    txt = g._closer_protein_step_text("guisantes secos", no_cook=False)
    assert txt == "Cocina guisantes secos en agua hasta que ablanden e incorpóralos al plato."


@pytest.mark.parametrize("nm,expect_snippet", [
    ("filete de pescado", "a la plancha o hervido"),   # pescado → default intacto (coherente)
    ("pechuga de pollo", "a la plancha o hervido"),     # pollo → default intacto
])
def test_meat_fish_default_unchanged(nm, expect_snippet):
    import graph_orchestrator as g
    txt = g._closer_protein_step_text(nm, no_cook=False)
    assert expect_snippet in txt, f"carne/pescado debe mantener el wording de plancha: {txt!r}"


def test_canned_and_softdairy_unchanged():
    import graph_orchestrator as g
    assert "ya viene cocido" in g._closer_protein_step_text("sardinas en lata", no_cook=False)
    assert "mézclalo" in g._closer_protein_step_text("queso cottage", no_cook=False)
    # legumbre ENLATADA gana la rama precocido (escurrir), no la de hervir
    assert "ya viene cocido" in g._closer_protein_step_text("garbanzos en lata", no_cook=False)


def test_cocido_legume_uses_incorporate():
    """Una legumbre que ya trae 'cocido' en el nombre usa 'Incorpora... mézclalo' (rama previa), no 'cocina'."""
    import graph_orchestrator as g
    txt = g._closer_protein_step_text("guisantes cocidos", no_cook=False)
    assert "Incorpora" in txt and "a la plancha" not in txt


def test_knob_off_reverts_to_default():
    import graph_orchestrator as g
    _prev = g.PROTEIN_STEP_LEGUME_WORDING
    try:
        g.PROTEIN_STEP_LEGUME_WORDING = False
        txt = g._closer_protein_step_text("guisantes secos", no_cook=False)
        assert "a la plancha" in txt, "knob OFF → comportamiento previo"
    finally:
        g.PROTEIN_STEP_LEGUME_WORDING = _prev
