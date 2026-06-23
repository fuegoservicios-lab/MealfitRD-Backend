"""[P3-PROTEIN-POWDER-REMOVE · 2026-06-22] La proteína en polvo se eliminó del catálogo
(La Sirena no vende suplementos + precio muy volátil). El prompt de plan ya NO la ofrece como
fuente de proteína liviana ni la sugiere para batidos (queda yogur griego como proteína de
batido segura). El opt-in de suplementos (feature aparte, se compra fuera) NO se toca.
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_PREFS = (_BACKEND / "prompts" / "preferences.py").read_text(encoding="utf-8")


def test_protein_powder_not_offered_as_light_protein():
    # La lista de proteínas livianas (desayuno/merienda) ya no incluye proteína en polvo.
    assert "Proteína en polvo (en batidos)" not in _PREFS


def test_shake_protein_suggests_yogurt_not_powder():
    # La regla de seguridad de batidos sugiere yogur griego, no proteína en polvo.
    assert "usa proteína en polvo o yogur griego" not in _PREFS
    assert "usa yogur griego" in _PREFS


def test_light_protein_options_still_present():
    # Las otras fuentes de proteína liviana siguen (el requisito de proteína por comida no se rompe).
    for opt in ("Huevos enteros", "ricotta", "Yogurt griego", "Mantequilla de maní"):
        assert opt in _PREFS, f"falta opción de proteína liviana: {opt}"


def test_food_safety_blended_note_no_protein_powder():
    """[P3-PROTEIN-POWDER-REMOVE] El food-safety note de batidos (huevo crudo) ya no sugiere
    proteína en polvo — solo yogur griego (alternativa segura disponible). Cazado por workflow."""
    import re
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    m = re.search(r"_FOOD_SAFETY_NOTE_BLENDED\s*=\s*\(\n(.*?)\n\)", src, re.DOTALL)
    assert m, "no se encontró _FOOD_SAFETY_NOTE_BLENDED"
    note = m.group(1).lower()
    assert "polvo" not in note, "el food-safety note de batidos aún sugiere proteína en polvo"
    assert "yogur" in note, "el food-safety note debe ofrecer yogur griego como alternativa"
