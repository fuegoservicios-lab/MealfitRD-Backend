"""[P1-PROTAGONIST-CONTEXT-GATE · 2026-07-29] El bump protagonista disparaba CIEGO sin
contexto kcal y el reconciliador de banda lo deshacía — oscilación intra-llamada + log
fantasma, confirmado en DB dos planes seguidos:

  · Burrito Dominicano (corr=6acd0c94): log "proteína PROTAGONISTA 40g→75g" en el shield
    pre-INSERT; la fila persistió **40 g** (a33807a0).
  · Revoltillo (corr=8469264b): DECLINADO ×2 con contexto ("alcanzable 30g/20g") en los
    surfaces de assemble, y "20g→75g" en el shield SIN contexto 8 s después del insert.

Causa: los surfaces de assemble pasan `day_kcal_target` + `db` (headroom real → declinan
honesto), pero el shield pre-INSERT llamaba `_fpc` sin `db` y el call interno del floor
iba con `day_kcal_target=None` → `_headroom is None` y `_kcal_per_g=0` → el cap de
affordability se saltaba → bump al piso completo (75 g) que la banda revertía después.

Cura en 3 capas:
1. GATE: sin contexto verificable (`_headroom is None` o `_kcal_per_g<=0`), los bumps
   protagonistas (proteína Y carbo) DECLINAN con razón visible — jamás disparan ciego.
2. `finalize_plan_data_coherence` deriva `day_kcal_target` 4-4-9 desde su kwarg
   `target_macros` y lo pasa a su floor interno.
3. El shield pre-INSERT (db_plans) pasa `db=IngredientNutritionDB()` a `_fpc` → los
   planes que SALTAN assemble (partial/SSE-fallback, la razón de ser del shield)
   conservan el piso protagonista, ahora con headroom real.

tooltip-anchor: P1-PROTAGONIST-CONTEXT-GATE
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parents[1]


class _FakeDB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g", str(s))
        g = float(m.group(1).replace(",", ".")) if m else 0.0
        return {"kcal": g * 1.65, "protein": g * 0.2, "carbs": 0.0, "fats": g * 0.05}


def _burrito(cals=700):
    return [{"day": 1, "meals": [{
        "name": "Burrito Dominicano de Cerdo con Cebada y Queso Parmesano",
        "ingredients": ["40 g de lomo de cerdo (en tiras)", "89 g de cebada perlada"],
        "ingredients_raw": ["40 g de lomo de cerdo (en tiras)", "89 g de cebada perlada"],
        "recipe": ["Cocina el cerdo."],
        "protein": 20, "carbs": 60, "fats": 15, "cals": cals}]}]


def test_protein_blind_fire_gated():
    """El caso vivo del burrito: sin day-target ni db, el bump NO dispara (antes: 40→75
    ciego que la banda revertía — el log mentía y la DB quedaba en 40)."""
    days = _burrito()
    go._floor_subservible_portions(days, day_kcal_target=None, db=None)
    assert days[0]["meals"][0]["ingredients"][0] == "40 g de lomo de cerdo (en tiras)"


def test_protein_fires_with_generous_headroom():
    days = _burrito(cals=700)
    n = go._floor_subservible_portions(days, day_kcal_target=2500, db=_FakeDB())
    assert n >= 1
    assert days[0]["meals"][0]["ingredients"][0].startswith("75 g de lomo de cerdo")


def test_protein_declines_with_tight_headroom():
    days = _burrito(cals=2490)
    go._floor_subservible_portions(days, day_kcal_target=2500, db=_FakeDB())
    assert days[0]["meals"][0]["ingredients"][0] == "40 g de lomo de cerdo (en tiras)"


def test_carb_blind_fire_gated():
    """Espejo carbo (la yautía de P1-RECIPE-POLISH-5): sin contexto, declina."""
    days = [{"day": 1, "meals": [{
        "name": "Pechuga Guisada con Cama de Yautía Asada",
        "ingredients": ["120 g de pechuga de pollo", "15 g de yautía"],
        "ingredients_raw": ["120 g de pechuga de pollo", "15 g de yautía"],
        "recipe": ["Hornea la yautía."],
        "protein": 30, "carbs": 10, "fats": 8, "cals": 280}]}]
    go._floor_subservible_portions(days, day_kcal_target=None, db=None)
    assert any(str(i) == "15 g de yautía" for i in days[0]["meals"][0]["ingredients"])


def test_carb_fires_with_context():
    days = [{"day": 1, "meals": [{
        "name": "Pechuga Guisada con Cama de Yautía Asada",
        "ingredients": ["120 g de pechuga de pollo", "15 g de yautía"],
        "ingredients_raw": ["120 g de pechuga de pollo", "15 g de yautía"],
        "recipe": ["Hornea la yautía."],
        "protein": 30, "carbs": 10, "fats": 8, "cals": 280}]}]
    n = go._floor_subservible_portions(days, day_kcal_target=2000, db=_FakeDB())
    assert n >= 1
    y = next(i for i in days[0]["meals"][0]["ingredients"] if "yaut" in str(i).lower())
    assert y.startswith("60 g")


def test_fpc_derives_day_kcal_and_shield_passes_db():
    """Anclas parser: (2) fpc deriva 4-4-9 desde target_macros para su floor interno;
    (3) el shield pre-INSERT pasa un IngredientNutritionDB real a _fpc."""
    src_go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src_go.index("def finalize_plan_data_coherence(")
    body = src_go[i:i + 40000]
    assert "P1-PROTAGONIST-CONTEXT-GATE" in body, "ancla del derive 4-4-9 en fpc"
    assert "_floor_subservible_portions(days, day_kcal_target=_sfl_day_kcal_fpc, db=db)" in body

    src_db = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    j = src_db.index("from graph_orchestrator import finalize_plan_data_coherence as _fpc")
    shield = src_db[j:j + 3000]
    assert "IngredientNutritionDB" in shield, (
        "el shield pre-INSERT debe pasar db= a _fpc (sin db, el floor protagonista "
        "declina y los planes que saltan assemble pierden el piso)"
    )


def test_gate_reason_is_visible_in_source():
    """El DECLINADO sin contexto debe loguearse con razón propia (diagnóstico ≠ headroom)."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("sin contexto kcal") >= 2, "gate visible en rama proteína Y carbo"
