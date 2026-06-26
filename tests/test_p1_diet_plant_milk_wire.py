"""[P1-DIET-PLANT-MILK-WIRE · 2026-06-26] Las leches/yogures VEGETALES nuevos del catálogo NO deben marcarse
como lácteo (animal) en planes veganos.

El backstop de dieta `_scan_diet_violations` (P1-DIET-HARD-GUARD) excusa análogos plant-based adyacentes al
término prohibido ("leche de coco" → no viola). Pero la lista de análogos no incluía "avena" → "Leche de
avena" (oat milk, recién añadida al catálogo) se marcaba como violación láctea en vegano, excluyéndola de
los planes veganos para los que se compró. Fix: +avena/arroz/nuez/nueces/avellana al regex `_plant_adj`.

Este test ancla que las 5 bebidas vegetales del lote 1 pasen en vegano, y que el guard SIGA vetando lácteo/
carne reales (no se debilitó).
"""
from __future__ import annotations

import graph_orchestrator as go


def _viol(ingredient: str, diet="vegano"):
    plan = {"days": [{"meals": [{"name": "t", "ingredients": [ingredient]}]}]}
    return go._scan_diet_violations(plan, diet)


def test_plant_milks_not_flagged_vegan():
    """Las 5 bebidas/yogures vegetales del catálogo deben pasar en vegano (cero violaciones)."""
    for ing in (
        "1 taza de leche de avena (240g)",     # el bug que cerramos
        "1 taza de leche de almendras (240g)",
        "1 taza de leche de coco (240g)",
        "1 taza de leche de soya (240g)",
        "1 pote de yogur de coco (150g)",
    ):
        assert _viol(ing) == [], f"falso positivo vegano: {ing} -> {_viol(ing)}"


def test_real_dairy_and_meat_still_flagged_vegan():
    """Defensa-en-profundidad: el guard NO se debilitó — lácteo/carne/huevo reales siguen vetados."""
    assert _viol("1 taza de leche de vaca (240g)"), "leche de vaca debe violar en vegano"
    assert _viol("1 taza de leche entera (240g)"), "leche entera debe violar en vegano"
    assert _viol("100g de pechuga de pollo"), "pollo debe violar en vegano"
    assert _viol("1 yogur griego (150g)"), "yogur (animal) debe violar en vegano"
    assert _viol("2 huevos"), "huevo debe violar en vegano"


def test_plant_milks_ok_in_vegetarian_and_balanced():
    """En vegetariano/omnívoro las bebidas vegetales tampoco violan (no aplica restricción láctea)."""
    assert _viol("1 taza de leche de avena (240g)", "vegetariano") == []
    assert _viol("1 taza de leche de avena (240g)", "balanced") == []


def test_anchor_present():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-DIET-PLANT-MILK-WIRE" in src
    idx = src.index("_plant_adj = _re.compile(")
    block = src[idx: idx + 400]  # el bloque del regex compile
    assert "avena" in block, "avena debe estar en el regex _plant_adj"
