"""[P1-UPDATE-PROTAGONIST-FLOOR · 2026-07-29] El piso protagonista llevaba el día entero inerte en
las surfaces que PERSISTEN el plato.

`P1-PROTAGONIST-CONTEXT-GATE` (esta misma mañana) hizo que el piso DECLINE cuando no hay contexto
verificable — correcto: antes bombeaba ciego al piso completo y el reconciliador de banda lo
deshacía, produciendo un log fantasma. Pero el finalizador de updates llamaba a
`_floor_subservible_portions` con `day_kcal_target=None` **hardcodeado**, así que a partir de ese
gate el piso protagonista (proteína Y carbo) no podía disparar NUNCA en swap-persist, chat-modify ni
agent. El comentario de producción que había justo encima seguía prometiendo lo contrario
("Surfaces que SÍ persisten el meal (swap/chat-modify/expand) conservan el default True").

El único callsite que sí tenía contexto (`_fpc`) corre solo dentro de `_finalize_plan_data_for_insert`,
que por su propio docstring es "previo a CUALQUIER INSERT" — y un swap-persist hace `jsonb_set` sobre
un plan que ya existe, así que jamás lo alcanza. Nada aguas abajo lo reparaba.

Medido con el pool abierto y la nutrition-db real, plato 'Locrio de Pavo':
    sin day-target (producción hoy) → '25 g de pavo molido'   (declina)
    con day-target (tras el fix)    → '75 g de pavo molido'   (dispara)

Un gate que declina por falta de contexto solo es correcto si ALGUIEN puede suministrar el contexto.
Si ningún callsite puede, no es un gate: es un apagado.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parents[1]
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PL = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TL = (_BACKEND / "tools.py").read_text(encoding="utf-8")


class _FakeDB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g", str(s))
        g = float(m.group(1).replace(",", ".")) if m else 0.0
        return {"kcal": g * 1.65, "protein": g * 0.2, "carbs": 0.0, "fats": g * 0.05}


def _meal():
    return {"meal": "Almuerzo", "name": "Locrio de Pavo con Papa Majada",
            "ingredients": ["25 g de pavo molido", "80 g de arroz blanco"],
            "ingredients_raw": ["25 g de pavo molido", "80 g de arroz blanco"],
            "recipe": ["Cocina el pavo molido con el sofrito.", "Agrega el arroz."],
            "protein": 12, "carbs": 30, "fats": 6, "cals": 220}


def _pavo_g(meal) -> float | None:
    for ln in meal["ingredients"]:
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\b", str(ln).lower())
        if m and "pavo" in str(ln).lower():
            return float(m.group(1).replace(",", "."))
    return None


# ---------------------------------------------------------------- el arreglo, funcionalmente

def test_finalizer_accepts_day_kcal_target_and_the_floor_fires():
    """Lo que el bug impedía: que una surface con día pudiera despertar el piso."""
    m = _meal()
    go.finalize_single_meal_recipe_coherence(m, db=_FakeDB(), day_kcal_target=2000.0)
    assert _pavo_g(m) >= 75, f"con contexto el piso protagonista debe disparar, quedó {_pavo_g(m)}"


def test_without_context_it_still_declines():
    """El gate NO se revierte: sin contexto sigue declinando (eso era lo correcto de
    P1-PROTAGONIST-CONTEXT-GATE). Lo que se arregla es poder suministrarlo."""
    m = _meal()
    go.finalize_single_meal_recipe_coherence(m, db=_FakeDB())      # sin day_kcal_target
    assert _pavo_g(m) == 25, "sin day-target debe declinar, no bombear ciego"


def test_headroom_still_caps_the_bump():
    """El bump sigue acotado por affordability: con el día casi lleno no salta al piso entero.
    Sin esto, 'restaurar el piso' sería reintroducir el bombeo ciego por otra puerta."""
    m = _meal()
    go.finalize_single_meal_recipe_coherence(m, db=_FakeDB(), day_kcal_target=260.0)
    g = _pavo_g(m)
    assert 25 < g < 75, f"headroom apretado ⇒ bump PARCIAL, quedó {g}"


# ---------------------------------------------------------------- el helper 4-4-9

@pytest.mark.parametrize("macros", [
    {"protein_g": 150, "carbs_g": 200, "fats_g": 70},     # forma del pipeline (target_macros)
    {"protein": 150, "carbs": 200, "fats": 70},           # forma persistida (plan_data['macros'])
])
def test_helper_accepts_both_shapes(macros):
    """Las dos formas conviven en el repo y las surfaces de update tienen la SEGUNDA. Un helper que
    solo entendiera `*_g` habría devuelto None justo donde hace falta, y el fix sería un no-op
    silencioso indistinguible del bug."""
    assert go._day_kcal_from_target_macros(macros) == pytest.approx(2030.0)


@pytest.mark.parametrize("bad", [None, {}, {"protein": None}, "no soy un dict", 42])
def test_helper_is_none_safe(bad):
    assert go._day_kcal_from_target_macros(bad) is None


# ---------------------------------------------------------------- el cableado en cada surface

def test_swap_persist_supplies_the_day_target():
    """swap-persist es el ÚNICO round-trip que persiste el swap editado por el cliente, y ya tenía
    `plan_data` a mano. Si alguien quita este kwarg, el piso vuelve a morir en silencio."""
    i = _PL.index("P2-SWAP-PERSIST-FINALIZE")
    seg = _PL[i:i + 3000]
    assert "_fin_sp(" in seg
    assert "day_kcal_target=_dkt_sp(plan_data.get(\"macros\"))" in seg, (
        "swap-persist debe pasar el day-target derivado del plan")
    assert "_day_kcal_from_target_macros as _dkt_sp" in seg


def test_chat_modify_supplies_it_too():
    i = _TL.index("_fin_rc_m(")
    seg = _TL[max(0, i - 1200):i + 400]
    assert "day_kcal_target=_dkt_m(" in seg, "chat-modify debe pasar el day-target"


def test_no_hardcoded_none_left_in_the_update_finalizer():
    """La regresión concreta: `day_kcal_target=None` escrito a mano en el finalizador de updates.
    Anclado al literal para que reaparecer cueste un rojo."""
    i = _GO.index("def finalize_single_meal_recipe_coherence")
    j = _GO.index("\ndef ", i + 10)
    body = _GO[i:j]
    assert "_floor_subservible_portions(_wrap, day_kcal_target=day_kcal_target" in body
    assert "_floor_subservible_portions(_wrap, day_kcal_target=None" not in body, (
        "vuelve a estar hardcodeado a None ⇒ piso protagonista inerte en las surfaces que persisten")


def test_production_comment_matches_reality_per_surface():
    """El comentario que había prometía que swap/chat-modify/expand 'conservan' el piso, cuando
    ninguno podía dispararlo. Una doc que promete lo contrario de lo que hace es peor que ninguna:
    el siguiente que lea creerá que la surface está cubierta. Ahora enumera el estado REAL de cada
    una, incluidas las dos que siguen declinando por invocar sin `db`."""
    i = _GO.index("Estado real del piso PROTAGONISTA por surface")
    seg = _GO[i:i + 700]
    for surface in ("swap-persist", "chat-modify", "agent swap"):
        assert surface in seg, f"falta el estado declarado de {surface}"
    assert "VIVO" in seg and "declina" in seg
