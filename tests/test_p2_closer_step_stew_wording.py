"""[P2-CLOSER-STEP-STEW-WORDING · 2026-07-24] La proteína añadida a un GUISO se incorpora al
guiso; no se cocina aparte "como proteína del plato".

Revisión del owner sobre el plan b7d07aeb: el único bolt-on que quedó era `55 g de pavo molido`
en unos *Frijoles Pintos Guisados con Arroz Integral*. **El alimento estaba bien** —arroz,
habichuelas y carne es la bandera dominicana, y el guard de coherencia lo permitió a propósito—
pero el paso lo contaba así:

    "Cocina pavo molido a la plancha o hervido y sírvelo como proteína del plato."

En un guiso que lleva 45 minutos en la olla, eso lee como un añadido de última hora en vez de
como una receta. El defecto no es QUÉ lleva el plato, es CÓMO se cuenta.

`_closer_protein_step_text` ya tenía ramas por naturaleza del alimento (licuado, lácteo blando,
enlatado, legumbre, plural). Faltaba la del CONTEXTO: si el plato es de olla —guiso, estofado,
caldo, sofrito— la proteína entra ahí.
"""
from __future__ import annotations

import graph_orchestrator as g
from constants import strip_accents as _sa


def _meal(name, steps=()):
    return {"name": name, "ingredients": [], "recipe": list(steps)}


# ---------------------------------------------------------------------------
# 1. El detector de plato de olla
# ---------------------------------------------------------------------------
def test_detecta_platos_de_olla():
    casos = [
        _meal("Frijoles Pintos Guisados con Arroz Integral"),
        _meal("Res Estofada con Víveres"),
        _meal("Pollo en Salsa Criolla"),
        _meal("Bowl", ["El Toque de Fuego: cocina tapado a fuego bajo hasta que el caldo espese."]),
    ]
    for m in casos:
        assert g._meal_is_stewy(m, _sa) is True, m["name"]


def test_no_confunde_platos_secos():
    casos = [
        _meal("Tortilla de Yuca con Palmito"),
        _meal("Batata Asada con Queso Blanco"),
        _meal("Ensalada de Berro con Ajonjolí"),
        _meal("Panqueques de Avena"),
    ]
    for m in casos:
        assert g._meal_is_stewy(m, _sa) is False, m["name"]


# ---------------------------------------------------------------------------
# 2. El wording
# ---------------------------------------------------------------------------
def test_guiso_incorpora_en_vez_de_cocinar_aparte():
    txt = g._closer_protein_step_text("pavo molido", no_cook=False, stewy=True)
    assert "guiso" in txt.lower()
    assert "como proteína del plato" not in txt
    assert "a la plancha" not in txt


def test_enlatado_en_guiso_no_dice_a_la_preparacion():
    """'Escurre e incorpora atún a la preparación' en un guiso queda igual de robótico."""
    txt = g._closer_protein_step_text("atún en agua", no_cook=False, stewy=True)
    assert "ya viene cocido" in txt          # sigue sin mandar a cocinar lo cocido
    assert "guiso" in txt.lower()


def test_plato_seco_conserva_el_wording_de_siempre():
    txt = g._closer_protein_step_text("pechuga de pollo", no_cook=False, stewy=False)
    assert "a la plancha" in txt


def test_las_ramas_previas_ganan_sobre_la_de_guiso():
    """Un batido o un lácteo blando no se 'incorporan al guiso' aunque el nombre despiste."""
    assert "licuadora" in g._closer_protein_step_text("queso cottage", no_cook=False,
                                                      blended=True, stewy=True)
    txt_dairy = g._closer_protein_step_text("yogurt griego", no_cook=True, stewy=True)
    assert "guiso" not in txt_dairy.lower()


def test_knob_permite_rollback(monkeypatch):
    monkeypatch.setattr(g, "CLOSER_STEP_STEW_WORDING", False)
    txt = g._closer_protein_step_text("pavo molido", no_cook=False, stewy=True)
    assert "a la plancha" in txt, "con el knob OFF vuelve al wording anterior"


def test_marker_presente():
    import pathlib
    src = pathlib.Path(g.__file__).with_suffix(".py").read_text(encoding="utf-8", errors="replace")
    assert "[P2-CLOSER-STEP-STEW-WORDING · 2026-07-24]" in src
