"""[P1-SODIUM-POSTMOTOR · 2026-07-04] Segunda pasada del autofix de sodio POST-motor.

Caso vivo (plan ed6db673, renovación 2026-07-04): los 3 días NACIERON bajo el techo
(el pase pre-motor no tuvo nada que hacer), pero el solver/rebalance/refine son
SODIO-CIEGOS y re-inflaron líneas saladas al dimensionar macros → Día 3 terminó sobre
2,000mg → `per_day_ceilings.flagged` → banner `micro_worst_day_ceiling` con el pase
pre-motor "verde". Mismo bug-clase que el micro-recheck P2-2 (erosión del motor).

Fix en dos piezas:
  1. Re-fire de `_day_sodium_autofix` en el seam post-motor (después del micro-recheck
     P2-2, ANTES del recompute del panel P2-1 — el soft-reject lee el estado reparado),
     con quantize + qty-sync de los meals tocados.
  2. El SWAP preserva los gramos ya dimensionados (medidos vía
     `db.grams_from_ingredient_string` sobre la línea original) en vez de un "150 g"
     fijo — post-motor un tamaño arbitrario introduciría drift de macros. Fallback
     150 g si la línea no resuelve a gramos.
"""
import os
import unicodedata

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


def _norm(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn")


class _FakeDB:
    """Sodio por substring + gramos medibles (línea de sardinas resuelve a 130 g)."""
    _NA = (("sardina", 900.0), ("atun", 700.0), ("queso", 400.0),
           ("pescado blanco", 60.0))

    def micros_from_ingredient_string(self, s):
        low = _norm(s)
        for tok, na in self._NA:
            if tok in low:
                return {"sodium_mg": na}
        return {"sodium_mg": 20.0}

    def grams_from_ingredient_string(self, s):
        return 130.0 if "sardina" in _norm(s) else 100.0


class _FakeDBNoGrams(_FakeDB):
    """DB que no resuelve gramos → el swap debe caer al fallback 150 g."""

    def grams_from_ingredient_string(self, s):
        raise RuntimeError("sin conversión")


def _mk_two_canned_day():
    # 900 + 700 + 20 (arroz) + 400 (queso) + 20 (casabe) = 2040 > 2000; sin cubito
    # → el strip no aplica y la escalera va directo al swap del más rico (sardinas).
    return [{
        "day": 1,
        "meals": [
            {"meal": "Almuerzo",
             "name": "Sardinas con Arroz",
             "ingredients": ["1 lata de sardinas en lata", "1 lata de atún en agua",
                             "150 g de arroz blanco"],
             "ingredients_raw": ["1 lata de sardinas en lata", "1 lata de atún en agua",
                                 "150 g de arroz blanco"],
             "recipe": ["Desmenuza las sardinas."]},
            {"meal": "Cena",
             "name": "Queso con Casabe",
             "ingredients": ["100 g de queso blanco", "60 g de casabe"],
             "ingredients_raw": ["100 g de queso blanco", "60 g de casabe"],
             "recipe": ["Sirve."]},
        ],
    }]


# ---------------------------------------------------------------------------
# wiring: el seam post-motor existe y corre ANTES del recompute del panel
# ---------------------------------------------------------------------------

def test_postmotor_callsite_between_p2_recheck_and_panel_recompute():
    i_recheck = _GO.index("(P2-2) Micro-recheck post-motor")
    i_sodium = _GO.index("[P1-SODIUM-POSTMOTOR · 2026-07-04] Segunda pasada POST-motor")
    i_panel = _GO.index("(P2-1) Panel de micros FRESCO al final del motor")
    assert i_recheck < i_sodium < i_panel, \
        "el re-fire de sodio debe correr tras el micro-recheck y ANTES del recompute del panel"
    win = _GO[i_sodium:i_sodium + 2200]
    assert "_day_sodium_autofix(days, form_data)" in win
    assert "_sync_recipe_step_quantities" in win, "qty-sync de los meals tocados"
    assert "_apply_portion_quantization" in win, "re-quantize del estado final"


def test_postmotor_gated_by_same_knob():
    i_sodium = _GO.index("[P1-SODIUM-POSTMOTOR · 2026-07-04] Segunda pasada POST-motor")
    win = _GO[i_sodium:i_sodium + 900]
    assert "if SODIUM_DAY_AUTOFIX_ENABLED:" in win, \
        "mismo kill-switch MEALFIT_SODIUM_DAY_AUTOFIX para ambos seams"


# ---------------------------------------------------------------------------
# funcional: swap preserva gramos dimensionados
# ---------------------------------------------------------------------------

def test_swap_preserves_measured_grams(go, monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _mk_two_canned_day()
    n = go._day_sodium_autofix(days, {}, db=_FakeDB())
    assert n == 1
    lunch = days[0]["meals"][0]
    swapped = [s for s in lunch["ingredients"] if "pescado blanco" in _norm(s)]
    assert swapped, "el enlatado más rico debe swapearse a pescado fresco"
    assert swapped[0].startswith("130 g de"), \
        f"el swap debe conservar los 130 g medidos de la línea original, no 150 fijos: {swapped[0]!r}"
    # raw alineado posicionalmente con la misma línea.
    assert swapped[0] in lunch["ingredients_raw"]


def test_swap_falls_back_to_150g_when_grams_unresolvable(go, monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _mk_two_canned_day()
    n = go._day_sodium_autofix(days, {}, db=_FakeDBNoGrams())
    assert n == 1
    lunch = days[0]["meals"][0]
    swapped = [s for s in lunch["ingredients"] if "pescado blanco" in _norm(s)]
    assert swapped and swapped[0].startswith("150 g de")


def test_idempotent_second_pass_noop(go, monkeypatch):
    """Simetría con el diseño de doble seam: si el pre-motor ya dejó el día bajo techo
    y el motor no lo re-infló, la segunda pasada NO debe tocar nada."""
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _mk_two_canned_day()
    assert go._day_sodium_autofix(days, {}, db=_FakeDB()) == 1
    snapshot = [list(m["ingredients"]) for m in days[0]["meals"]]
    assert go._day_sodium_autofix(days, {}, db=_FakeDB()) == 0, \
        "segunda pasada sobre un día ya reparado = no-op"
    assert [list(m["ingredients"]) for m in days[0]["meals"]] == snapshot


def test_marker_anchored_in_source():
    assert "P1-SODIUM-POSTMOTOR" in _GO
