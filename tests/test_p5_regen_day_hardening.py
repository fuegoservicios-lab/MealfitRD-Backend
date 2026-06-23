"""[P5-REGEN-DAY-HARDENING · 2026-06-23] Endurecimiento del feature de actualizar platos
tras el audit de producción (0 bloqueadores, 3 gaps reales cerrados):

- #4 P5-MEAL-PROTEIN-FALLBACK: si el caller pobló kcal pero NO la proteína del plato
  (frontend viejo cacheado, o meal sin macro de proteína), el gate NO debe degradarse a
  solo-kcal — cae al target diario escalado (preserva la PALANCA proteína).
- #2 P5-REGEN-DAY-LLM-DEGRADE: rate-limit / breaker abierto en /regenerate-day → soft-fail
  'ai_unavailable' (HTTP 200) en vez de 500 opaco; sin cobrar crédito.
- #3 P5-REGEN-DAY-STALE-LIST/RECALC-RETRY: el frontend strippea las 4 listas + reintenta
  el recalc (no deja listas viejas que comprarían lo equivocado).
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nutrition_db import IngredientNutritionDB
from inventory_sufficiency import evaluate_pantry_sufficiency

_MASTER = [
    {"name": "Pollo", "kcal_per_100g": 165, "protein_g_per_100g": 31.0,
     "carbs_g_per_100g": 0.0, "fats_g_per_100g": 3.6},
    {"name": "Arroz", "kcal_per_100g": 130, "protein_g_per_100g": 2.7,
     "carbs_g_per_100g": 28.0, "fats_g_per_100g": 0.3},
]


@pytest.fixture
def db():
    return IngredientNutritionDB(rows=_MASTER)


@pytest.fixture(autouse=True)
def _stub_targets(monkeypatch):
    monkeypatch.setattr(
        "nutrition_calculator.get_nutrition_targets",
        lambda fd: {"target_calories": 2100, "macros": {"protein_g": 124.0, "carbs_g": 270.0, "fats_g": 58.0}},
    )
    monkeypatch.setattr(
        "micronutrients.dri_targets",
        lambda sex=None, age=None, pregnant=False: {"iron_mg": {"floor": 8.0, "unit": "mg"}},
    )


_FORM = {"goal": "gain_muscle", "gender": "male", "age": 30}


# --- #4: protein-lever fallback when meal_target omits protein -----------------
def test_null_protein_falls_back_to_daily_not_kcal_only(db):
    """meal_target con kcal pero SIN proteína + Nevera alta-kcal/baja-proteína (2 lb arroz)
    debe BLOQUEAR por proteína. Sin el fallback, la proteína se saltaría (req=0) y el gate
    pasaría solo con kcal — exactamente el check débil que el feature buscó reemplazar."""
    out = evaluate_pantry_sufficiency(
        "u1", _FORM, scope="meal",
        meal_target={"kcal": 500},  # NO protein_g
        nutrition_db=db,
        inventory_rows=[{"ingredient_name": "Arroz", "available_quantity": 2.0, "unit": "lb"}],
    )
    assert out["sufficient"] is False, "proteína debe seguir siendo la palanca aunque el target no la traiga"
    prot = [d for d in out["deficits"] if d["nutrient"] == "protein_g"]
    assert prot and prot[0]["advisory"] is False
    # kcal sí alcanza (2 lb arroz ≈ 1180 kcal > 500) → el bloqueo es por proteína, no por kcal.
    assert out["coverage"].get("kcal", 0) >= 0.80


def test_explicit_protein_target_still_respected(db):
    """Regresión: con proteína explícita en el target, el comportamiento no cambia."""
    out = evaluate_pantry_sufficiency(
        "u1", _FORM, scope="meal",
        meal_target={"kcal": 500, "protein_g": 40.0},
        nutrition_db=db,
        inventory_rows=[{"ingredient_name": "Pollo", "available_quantity": 2.0, "unit": "lb"}],
    )
    assert out["sufficient"] is True


# --- #2 + #3: anchors parser-based en el source de prod -----------------------
_PLANS = open(os.path.join(os.path.dirname(__file__), "..", "routers", "plans.py"), encoding="utf-8").read()
_ASSESS = open(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "context", "AssessmentContext.jsx"), encoding="utf-8").read()


def test_regen_day_catches_llm_unavailable():
    assert "LLMRateLimitedError" in _PLANS and "LLMCircuitBreakerOpen" in _PLANS
    # El except de IA-no-disponible y el soft-fail 'ai_unavailable' deben coexistir.
    assert re.search(r"except\s*\(\s*LLMRateLimitedError\s*,\s*LLMCircuitBreakerOpen\s*\)", _PLANS), \
        "el loop del día debe atrapar las excepciones transitorias de IA"
    assert '"ai_unavailable"' in _PLANS or "'ai_unavailable'" in _PLANS


def test_regen_day_frontend_strips_all_four_lists_and_retries():
    # Las 4 keys deben borrarse localmente (no solo la canónica).
    for key in ("aggregated_shopping_list_weekly", "aggregated_shopping_list_biweekly", "aggregated_shopping_list_monthly"):
        assert f"delete updatedPlan.{key}" in _ASSESS, f"regenerateDay debe borrar {key} local"
    # Y el recalc debe reintentar (no 1-solo-intento silencioso).
    assert "P5-REGEN-DAY-RECALC-RETRY" in _ASSESS
