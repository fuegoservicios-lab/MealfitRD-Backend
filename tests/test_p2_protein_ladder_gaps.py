"""[P2-PROTEIN-LADDER-GAPS · 2026-07-11] atún y mariscos tienen escalera de swap en el
autofix de proteína repetida.

Caso vivo (corr=c0a950c6, generación modo-Nevera del owner, 08:23): day-gen emitió atún
en 2 comidas del Día 2 → el autofix lo DETECTÓ pero `no_ladder_for_label` (la exclusión
original temía "lata de pollo en agua" y asumía que el gate degradaba a advisory — ya no:
rechaza HIGH en intentos no-finales) → rechazo de plan COMPLETO evitable. La solución
correcta: escalera + COMPUESTOS largos-primero ("atún en agua" se reescribe ENTERA →
"pechuga de pollo", jamás "lata de pollo en agua").

tooltip-anchor: P2-PROTEIN-LADDER-GAPS
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))


class _FDB:
    def macros_from_ingredient_string(self, s):
        return None


def _day(meals):
    return {"day": 2, "meals": meals}


def _meal(name, slot, ingredients, recipe):
    return {"name": name, "meal": slot, "ingredients": list(ingredients),
            "ingredients_raw": list(ingredients), "recipe": list(recipe),
            "protein": 25, "carbs": 10, "fats": 4}


# ---------------------------------------------------------------------------
# 1. Todos los labels del gate tienen escalera (paridad gate↔autofix)
# ---------------------------------------------------------------------------

def test_every_gate_label_has_ladder_or_special_handling():
    from graph_orchestrator import _SAME_DAY_PROTEIN_GATE_LABELS, _PROTEIN_REPEAT_SWAP_LADDER
    # huevo tiene maquinaria propia (egg-cap/protagonist); el resto DEBE tener escalera.
    missing = [l for l in _SAME_DAY_PROTEIN_GATE_LABELS
               if l != "huevo" and l not in _PROTEIN_REPEAT_SWAP_LADDER]
    assert not missing, (
        f"labels del gate SIN escalera de swap: {missing} — el autofix queda impotente "
        "(no_ladder_for_label) y el reviewer rechaza el plan completo (corr=c0a950c6)"
    )


def test_ladder_targets_have_forms():
    from graph_orchestrator import _PROTEIN_REPEAT_SWAP_LADDER, _PROTEIN_TARGET_FORMS
    for src, targets in _PROTEIN_REPEAT_SWAP_LADDER.items():
        for t in targets:
            assert t in _PROTEIN_TARGET_FORMS, (
                f"escalera {src}→{t}: target sin _PROTEIN_TARGET_FORMS (el rewrite fallaría)"
            )


# ---------------------------------------------------------------------------
# 2. El caso vivo: atún ×2 same-day se corrige SIN 'lata de pollo en agua'
# ---------------------------------------------------------------------------

def test_atun_repeat_fixed_with_clean_compound_rewrite():
    from graph_orchestrator import _protein_repeat_autofix
    d = _day([
        _meal("Ensalada de Atún en Agua", "Almuerzo",
              ["120 g de atún en agua", "50 g de lechuga"],
              ["Mezcla el atún en agua con la lechuga."]),
        _meal("Wrap de Atún", "Cena",
              ["100 g de atún en agua", "1 tortilla"],
              ["Rellena la tortilla con el atún."]),
    ])
    fixed = _protein_repeat_autofix([d], {"mainGoal": "lose_fat"})
    assert fixed >= 1, "atún ×2 same-day debe corregirse (era no_ladder_for_label)"
    blob = " ".join(str(x) for m in d["meals"] for x in (m["ingredients"] + m["recipe"] + [m["name"]]))
    assert "lata de pollo" not in blob.lower() and "pollo en agua" not in blob.lower(), (
        f"el compound debe reescribir la frase enlatada ENTERA: {blob!r}"
    )
    # exactamente UNA comida conserva atún (identidad del keeper)
    _atun_meals = sum(1 for m in d["meals"] if "atun" in (m["name"] + " " + " ".join(m["ingredients"])).lower()
                      or "atún" in (m["name"] + " " + " ".join(m["ingredients"])).lower())
    assert _atun_meals == 1


def test_camarones_repeat_fixed():
    from graph_orchestrator import _protein_repeat_autofix
    d = _day([
        _meal("Camarones Guisados", "Almuerzo",
              ["150 g de camarones frescos", "arroz"],
              ["Guisa los camarones frescos."]),
        _meal("Salteado de Camarones", "Cena",
              ["120 g de camarones cocidos", "vegetales"],
              ["Saltea los camarones cocidos."]),
    ])
    fixed = _protein_repeat_autofix([d], {"mainGoal": "lose_fat"})
    assert fixed >= 1, "camarones ×2 same-day debe corregirse (label del gate desde b27eb26)"


def test_marker_anchored_in_source():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("P2-PROTEIN-LADDER-GAPS") >= 2
