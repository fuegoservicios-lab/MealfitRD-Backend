"""[P1-CLOSER-SWEET-GUARD · 2026-06-27] Re-test en vivo (corr=26311f6f) tras P1-CLOSER-COHERENCE: el plan bariátrico
fue RECHAZADO CRÍTICO porque FASE A (el re-cierre de proteína post-cap) inyectó +81g de CAMARONES en CADA comida con
déficit — incluyendo platos DULCES (avena+guineo, yogurt+lechosa, tortilla+fresas). FASE A excluye lácteos (para no
re-recortarlos) → elige la proteína animal no-láctea más densa (camarón) → la mete en yogures/avenas dulces =
organolépticamente aberrante (el revisor lo marcó crítico en 8 comidas).

Fix: `_close_protein_gap_for_meal` detecta plato DULCE (`_is_sweet_meal`: yogurt/avena/batido/fruta dulce en el
nombre) y filtra del pool las proteínas SALADAS (`_MEAT_PROTEIN_HINT`: camarón/pescado/carne). Si el pool queda
vacío → return 0 (no fuerza el combo; el piso de proteína se cubre en las comidas saladas). Gateado por
CLOSER_DISH_COHERENCE_ENABLED. tooltip-anchor: P1-CLOSER-SWEET-GUARD
"""
from __future__ import annotations

import graph_orchestrator as g
from constants import strip_accents as _sa


class _Info:
    def __init__(self, name, protein, carbs=2.0, fats=1.0, kcal=95.0):
        self.name, self.protein, self.carbs, self.fats, self.kcal = name, protein, carbs, fats, kcal


_CANDS = [(0.20, "Camarones", _Info("Camarones", 24)),
          (0.25, "Pechuga de pollo", _Info("Pechuga de pollo", 31))]


def _meal(name):
    return {"name": name, "protein": 5, "carbs": 20, "fats": 3, "cals": 150,
            "ingredients": ["base", "fruta"]}


def test_is_sweet_meal_detects_sweet_contexts():
    for nm in ("Avena Cremosa con Guineo", "Yogurt Griego con Lechosa y Linaza",
               "Tortilla de Huevos con Fresas y Ricotta", "Batido de Mango", "Panqueques con Miel"):
        assert g._is_sweet_meal({"name": nm}, _sa) is True, nm


def test_is_sweet_meal_savory_is_false():
    for nm in ("Bacalao Guisado con Arroz Integral", "Queso de Hoja al Vapor con Coliflor y Batata",
               "Pollo a la Plancha con Ensalada"):
        assert g._is_sweet_meal({"name": nm}, _sa) is False, nm


def test_no_savory_protein_into_sweet_meal():
    """El closer NO debe meter camarón/pescado/carne en un plato dulce (solo candidatos salados → skip)."""
    sweet = _meal("Yogurt Griego con Lechosa y Linaza")
    added = g._close_protein_gap_for_meal(sweet, 25, None, _CANDS)
    assert added == 0, "no debe inyectar proteína salada en un yogurt+fruta"
    assert not any("camaron" in str(i).lower() or "pollo" in str(i).lower() for i in sweet["ingredients"])


def test_savory_protein_into_savory_meal_ok():
    savory = _meal("Queso de Hoja al Vapor con Coliflor y Batata")
    added = g._close_protein_gap_for_meal(savory, 25, None, _CANDS)
    assert added > 0, "en plato salado SÍ debe cerrar el piso de proteína"


def test_anchor():
    src = (g.__file__)
    import pathlib
    txt = pathlib.Path(src).read_text(encoding="utf-8")
    assert "P1-CLOSER-SWEET-GUARD" in txt or "_is_sweet_meal" in txt
    assert "_SWEET_MEAL_MARKERS" in txt and "def _is_sweet_meal" in txt
