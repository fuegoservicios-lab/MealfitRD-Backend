"""[P1-CLOSER-FRESH-COCIDO · 2026-07-07] pescado FRESCO con "(ya viene cocido)".

Review visual plan 30d (plato "Catibías... y Pescado blanco"):
  "💪 Escurre e incorpora filete de pescado (ya viene cocido) a la preparación
   antes de servir."
El ingrediente es FRESCO ("½ filete de pescado") → decirle al usuario que el pescado
fresco "ya viene cocido" es incoherente + riesgo food-safety (servir pescado crudo).

Causa: la pasada `_align_closer_note_food_names` (P2-NOTE-LINE-NAME-ALIGN) adopta el
nombre del alimento de la LÍNEA de compras en el paso 💪, pero un simple replace del
nombre CONSERVABA el sufijo "(ya viene cocido)" del wording de enlatado. Al alinear un
protein enlatado→fresco, quedaba el sufijo mentiroso.

Fix: si el paso trae "(ya viene cocido)" pero el alimento de la línea NO matchea
_PRECOOKED_PROTEIN_HINT (es fresco), re-renderizar el paso completo con el SSOT
`_closer_protein_step_text` → "Cocina filete de pescado a la plancha o hervido".
tooltip-anchor: P1-CLOSER-FRESH-COCIDO
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_SRC = (Path(g.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_marker_present():
    assert "P1-CLOSER-FRESH-COCIDO" in _SRC


def test_fresh_fish_loses_cocido_suffix():
    """El caso exacto del PDF: pescado fresco NO debe quedar '(ya viene cocido)'."""
    meal = {
        "name": "Catibías Rellenas con Pescado blanco",
        "ingredients": ["½ filete de pescado", "55 g de queso blanco"],
        "recipe": [
            "MISE EN PLACE: mezcla la harina de maíz con agua.",
            "EL TOQUE DE FUEGO: hornea las catibías a 200°C por 15 minutos hasta dorar.",
            "💪 Escurre e incorpora filete de pescado enlatado (ya viene cocido) a la preparación antes de servir.",
            "MONTAJE: sirve las catibías.",
        ],
    }
    g._align_closer_note_food_names(meal)
    joined = " ".join(meal["recipe"])
    assert "ya viene cocido" not in joined, f"pescado fresco no debe declararse ya-cocido: {meal['recipe']}"
    assert "Cocina filete de pescado" in joined, f"debe indicar cocinar el pescado fresco: {meal['recipe']}"


def test_canned_protein_keeps_cocido():
    """CONTROL: un enlatado REAL (atún en agua) SÍ conserva '(ya viene cocido)'."""
    meal = {
        "name": "Ensalada de Atún",
        "ingredients": ["1 lata de atún en agua", "2 tazas de lechuga"],
        "recipe": [
            "MISE EN PLACE: lava la lechuga.",
            "💪 Escurre e incorpora atún en agua (ya viene cocido) a la preparación antes de servir.",
            "MONTAJE: sirve.",
        ],
    }
    g._align_closer_note_food_names(meal)
    joined = " ".join(meal["recipe"])
    assert "ya viene cocido" in joined, f"el atún enlatado SÍ viene cocido — no tocar: {meal['recipe']}"


def test_no_cocido_step_untouched():
    """Un paso normal sin '(ya viene cocido)' no se ve afectado por el fix."""
    meal = {
        "name": "Pollo a la Plancha",
        "ingredients": ["150 g de pechuga de pollo"],
        "recipe": [
            "💪 Cocina pechuga de pollo a la plancha o hervido y sírvelo como proteína del plato.",
            "MONTAJE: sirve.",
        ],
    }
    before = list(meal["recipe"])
    g._align_closer_note_food_names(meal)
    assert meal["recipe"] == before, "un paso sin sufijo cocido no debe cambiar"
