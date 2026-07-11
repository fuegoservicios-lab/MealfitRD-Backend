"""[P1-EGG-INTRINSIC-DEDUP · 2026-07-11] Pase (3) del dedup same-day de huevo.

Caso vivo corr=9cc4317e (PRIMER plan modo-Nevera del owner, 11:47): Día 1 con
"Revoltillo Dominicano..." (desayuno, 1 huevo) + "Tortilla de Queso Blanco con
Espinacas y Tomate" (cena, 5 claras) — ambas INTRÍNSECAS (cabeza revoltillo/tortilla)
→ pase (1) exige huevo fuera del nombre, pase (2) excluye intrínsecas → NADIE corrigió
→ el gate same-day rechazó los 3 intentos → entrega degradada con banda 0.92.

Contrato del pase (3):
1. Con ≥2 comidas-huevo intrínsecas el mismo día: conserva UNA (prioriza desayuno) y
   reasigna las demás con la escalera slot-aware.
2. TRANSPLANTE DE CABEZA: el plato sin huevo no puede llamarse tortilla/revoltillo →
   "Salteado de ..." (y si la proteína nueva no queda nombrada, se inserta).
3. `prefer_label`: si el nombre ya menciona queso/pollo/yogurt, esa proteína va primero
   — el reemplazo coincide con el nombre (name-honesty por construcción; la tortilla
   del caso vivo decía "de Queso Blanco" con queso AUSENTE de ingredientes).
4. Knob MEALFIT_EGG_INTRINSIC_DEDUP (default ON) — revertible sin redeploy.

tooltip-anchor: P1-EGG-INTRINSIC-DEDUP
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))


def _meal(name, slot, ingredients, recipe=None):
    return {"name": name, "meal": slot, "ingredients": list(ingredients),
            "ingredients_raw": list(ingredients),
            "recipe": list(recipe or ["Prepara los ingredientes."]),
            "protein": 20, "carbs": 15, "fats": 8}


def _live_day():
    """Día 1 del plan 0a491a0b (recorte a los 2 platos-huevo + 1 neutro)."""
    return {"day": 1, "meals": [
        _meal("Revoltillo Dominicano con Casabe y Aguacate", "Desayuno",
              ["1 huevo", "1½ tomates", "½ cebolla", "1½ casabe albahaca"],
              ["Bate el huevo y revuélvelo con el tomate y la cebolla."]),
        _meal("Res a la Parrilla con Plátano Maduro", "Almuerzo",
              ["120g de carne de res (lomo magro)", "½ plátano maduro", "¼ taza de arroz blanco"],
              ["Asa la res y sirve con el plátano."]),
        _meal("Tortilla de Queso Blanco con Espinacas y Tomate", "Cena",
              ["5 claras de huevo", "2 tazas de espinacas frescas", "1½ tomates"],
              ["Bate las claras y cuaja la tortilla con las espinacas."]),
    ]}


def _egg_meals(day):
    import re
    out = []
    for m in day["meals"]:
        blob = (str(m.get("name", "")) + " " + " ".join(str(i) for i in m["ingredients"])).lower()
        if re.search(r"\bhuevos?\b|\bclaras?\b|\byemas?\b", blob):
            out.append(m)
    return out


class _FDB:
    def macros_from_ingredient_string(self, s):
        return None


def test_live_case_dedups_to_single_egg_meal():
    from graph_orchestrator import _protein_repeat_autofix
    d = _live_day()
    fixed = _protein_repeat_autofix([d], {"mainGoal": "lose_fat"}, db=_FDB())
    assert fixed >= 1, "revoltillo+tortilla same-day debe corregirse (era el residual sin dueño)"
    eggs = _egg_meals(d)
    assert len(eggs) == 1, f"debe quedar UNA comida-huevo, quedaron {len(eggs)}"


def test_keeper_is_breakfast():
    from graph_orchestrator import _protein_repeat_autofix
    d = _live_day()
    _protein_repeat_autofix([d], {"mainGoal": "lose_fat"}, db=_FDB())
    desayuno = d["meals"][0]
    assert "huevo" in " ".join(desayuno["ingredients"]).lower(), (
        "el desayuno (hogar cultural del huevo) es el keeper — el revoltillo queda intacto"
    )
    assert desayuno["name"].startswith("Revoltillo"), "el keeper no se renombra"


def test_loser_head_transplant_and_name_honesty():
    from graph_orchestrator import _protein_repeat_autofix
    d = _live_day()
    _protein_repeat_autofix([d], {"mainGoal": "lose_fat"}, db=_FDB())
    cena = d["meals"][2]
    name_l = cena["name"].lower()
    ings_l = " ".join(cena["ingredients"]).lower()
    assert "clara" not in ings_l and "huevo" not in ings_l, "las claras deben reasignarse"
    assert not name_l.startswith("tortilla"), (
        f"un plato sin huevo no puede llamarse tortilla: {cena['name']!r}"
    )
    # prefer_label: el nombre decía 'Queso Blanco' → el reemplazo debe SER queso
    # (name-honesty por construcción, cierra también el flag P2-NAME-HONESTY del caso vivo).
    assert "queso" in ings_l, f"el nombre menciona queso → ingredientes deben tener queso: {ings_l!r}"
    assert "queso" in name_l, f"la proteína nueva queda nombrada: {cena['name']!r}"
    assert cena.get("_protein_autofix_applied", "").startswith("huevo->"), "marca de fidelity"


def test_single_egg_day_untouched():
    from graph_orchestrator import _protein_repeat_autofix
    d = {"day": 2, "meals": [
        _meal("Revoltillo Criollo", "Desayuno", ["2 huevos", "½ cebolla"]),
        _meal("Pollo Guisado", "Almuerzo", ["150g de pechuga de pollo", "arroz"]),
    ]}
    _protein_repeat_autofix([d], {"mainGoal": "lose_fat"}, db=_FDB())
    assert d["meals"][0]["name"] == "Revoltillo Criollo"
    assert "huevos" in " ".join(d["meals"][0]["ingredients"]).lower()


def test_knob_off_restores_previous_behavior(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "EGG_INTRINSIC_DEDUP_ENABLED", False)
    d = _live_day()
    go._protein_repeat_autofix([d], {"mainGoal": "lose_fat"}, db=_FDB())
    assert len(_egg_meals(d)) == 2, (
        "con el knob OFF, las intrínsecas vuelven a quedar para el gate (rollback limpio)"
    )


def test_marker_anchored_in_source():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("P1-EGG-INTRINSIC-DEDUP") >= 3
