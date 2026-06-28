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


def test_seafood_is_savory():
    """[corr=713b8e84] el cerrador metió 'Piña con Ricotta y Cangrejo' — cangrejo/mariscos deben contar como
    proteína SALADA para que el sweet-guard los bloquee en platos dulces."""
    for sea in ("cangrejo", "langosta", "pulpo", "almeja", "mejillon", "jaiba"):
        assert sea in g._MEAT_PROTEIN_HINT, sea


def test_savory_override_ceviche_and_verde():
    """Una preparación salada (ceviche, fruta VERDE/unripe, guiso) NO es postre aunque traiga un token dulce —
    ahí SÍ va proteína de mar/carne."""
    assert g._is_sweet_meal({"name": "Ceviche de Lechosa Verde con Limón y Cangrejo"}, _sa) is False
    assert g._is_sweet_meal({"name": "Guineítos Verdes Salteados con Pollo"}, _sa) is False
    # pero piña fresca (madura, sin override) SÍ es dulce → bloquea mariscos
    assert g._is_sweet_meal({"name": "Piña Fresca con Queso Ricotta"}, _sa) is True


def test_crab_blocked_from_sweet_pineapple():
    sweet = _meal("Piña Fresca con Queso Ricotta")
    cands = [(0.20, "Cangrejo", _Info("Cangrejo", 19)), (0.25, "Pechuga de pollo", _Info("Pechuga de pollo", 31))]
    assert g._close_protein_gap_for_meal(sweet, 25, None, cands) == 0, "no debe meter cangrejo en un plato de piña dulce"


def test_goat_and_meats_are_savory():
    """[corr=6b859fc4] el cerrador metió 'Yogurt con Lechosa y Chivo' — chivo/cordero/conejo/etc. deben contar como
    carne para que el sweet-guard los bloquee en platos dulces."""
    for meat in ("chivo", "cabra", "cordero", "conejo", "pato", "ternera", "chorizo", "jamon", "pernil"):
        assert meat in g._MEAT_PROTEIN_HINT, meat


def test_ground_seeds_not_flagged_as_meat():
    """'molida' NO debe estar en la lista de carnes (falso positivo: linaza/almendra/canela molida)."""
    assert "molida" not in g._MEAT_PROTEIN_HINT


def test_goat_blocked_from_sweet_yogurt():
    sweet = _meal("Yogurt Griego con Lechosa y Linaza Molida")
    cands = [(0.20, "Chivo", _Info("Chivo", 27)), (0.25, "Pechuga de pollo", _Info("Pechuga de pollo", 31))]
    assert g._close_protein_gap_for_meal(sweet, 25, None, cands) == 0, "no debe meter chivo en un yogurt+fruta dulce"


def test_anchor():
    src = (g.__file__)
    import pathlib
    txt = pathlib.Path(src).read_text(encoding="utf-8")
    assert "P1-CLOSER-SWEET-GUARD" in txt or "_is_sweet_meal" in txt
    assert "_SWEET_MEAL_MARKERS" in txt and "def _is_sweet_meal" in txt
