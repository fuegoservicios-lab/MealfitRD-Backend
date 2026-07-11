"""[P1-RECALC-FLOOR-READONLY · 2026-07-11] Surfaces derivativas (recalc de lista) NO mutan
porciones: el floor sub-servible/protagonista solo corre donde el meal SE PERSISTE.

Caso vivo (16:53Z, corr=bce1f0bf): /recalculate-shopping-list ejecuta
`finalize_single_meal_recipe_coherence` sobre una COPIA descartable de plan_data (persiste
solo aggregated_*). El piso PROTAGONISTA (P1-RECIPE-VISIBLE-DEFECTS, day_kcal_target=None
→ sin límite de headroom) bombeó 25g→75g de pavo en 'Locrio de Pavo' EN LA COPIA → la
lista de compras se computó para 75g mientras plan_data persistido decía "25 g de pavo
molido" (verificado SQL). Exactamente la clase de incoherencia receta↔lista que el
coherence guard defiende.

Contrato: kwarg `portion_floors` (default True — swap/chat-modify/expand persisten el meal
y conservan el floor); el recalc pasa False.

tooltip-anchor: P1-RECALC-FLOOR-READONLY
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))


def test_signature_has_portion_floors_default_true():
    import graph_orchestrator as go
    sig = inspect.signature(go.finalize_single_meal_recipe_coherence)
    p = sig.parameters.get("portion_floors")
    assert p is not None, "kwarg portion_floors desapareció del finalizador single-meal"
    assert p.default is True, "el default debe ser True (swap/expand persisten el meal)"


def test_floor_gated_by_kwarg_in_source():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "if PORTION_SHRINK_FLOOR_ENABLED and portion_floors:" in src, (
        "el floor dentro del finalizador single-meal debe respetar portion_floors"
    )


def test_recalc_callsite_passes_false():
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i = src.find("def api_recalculate_shopping_list")
    assert i != -1
    body = src[i:i + 20000]
    assert "portion_floors=False" in body, (
        "/recalculate-shopping-list computa sobre copia descartable — sin "
        "portion_floors=False la lista diverge del plan persistido (pavo 25g→75g vivo)"
    )


def test_functional_floor_respects_kwarg():
    from graph_orchestrator import finalize_single_meal_recipe_coherence

    def _meal():
        return {"meal": "Almuerzo", "name": "Locrio de Pavo con Papa Majada",
                "ingredients": ["25 g de pavo molido", "80 g de arroz blanco"],
                "ingredients_raw": ["25 g de pavo molido", "80 g de arroz blanco"],
                "recipe": ["Cocina el pavo molido con el sofrito.", "Agrega el arroz."],
                "protein": 12, "carbs": 30, "fats": 6, "cals": 220}

    def _pavo_grams(meal):
        for ln in meal["ingredients"]:
            m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\b", str(ln).lower())
            if m and "pavo" in str(ln).lower():
                return float(m.group(1).replace(",", "."))
        return None

    m_ro = _meal()
    finalize_single_meal_recipe_coherence(m_ro, portion_floors=False)
    assert _pavo_grams(m_ro) is not None and _pavo_grams(m_ro) < 75, (
        "con portion_floors=False la línea protagonista NO debe bombearse (surface derivativa)"
    )

    m_rw = _meal()
    finalize_single_meal_recipe_coherence(m_rw)  # default True
    assert _pavo_grams(m_rw) is not None and _pavo_grams(m_rw) >= 75, (
        "con el default True el piso PROTAGONISTA sigue activo (surfaces que persisten)"
    )


def test_marker_anchored_in_source():
    go_src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    pl_src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert go_src.count("P1-RECALC-FLOOR-READONLY") >= 1
    assert pl_src.count("P1-RECALC-FLOOR-READONLY") >= 1
