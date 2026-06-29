"""[P2-REGEN-DAY-BIOMETRICS-PROPAGATE · 2026-06-29] (cierre follow-up testing en vivo)

El `meal_form` que regenerate-day pasa a `swap_meal` es un dict de keys EXPLÍCITAS (no hace spread de
`data`). `_enrich_clinical_from_profile` hidrata los biométricos en `data` (block P2-UPDATE-HYDRATE-
BIOMETRICS), pero el meal_form NO los propagaba → swap_meal recibía form SIN weight/height/age →
`get_nutrition_targets` caía a defaults (154lb/25/moderate) → WARN P2-MINOR-GATE/P0-FORM-4 + lógica
biométrica del swap (micro-steer, caps) con datos falsos.

(El síntoma band-0.0 ya lo cerró P2-REGEN-DAY-SLOT-OVERRIDE-SKIP; esto es defensa-en-profundidad.)

Fix: el meal_form propaga weight/height/age/gender/weightUnit/activityLevel/bodyFat desde `data`.
Test parser-based (Neon-free): ancla la propagación dentro del bloque meal_form de regen-day.
"""
from pathlib import Path

_PLANS = (Path(__file__).resolve().parent.parent / "routers" / "plans.py").read_text(encoding="utf-8")


def test_marker_present():
    assert "P2-REGEN-DAY-BIOMETRICS-PROPAGATE" in _PLANS


def test_meal_form_propagates_biometrics():
    # El meal_form de regen-day debe pasar cada biométrico desde `data`.
    for line in (
        '"weight": data.get("weight")',
        '"height": data.get("height")',
        '"age": data.get("age")',
        '"gender": data.get("gender")',
        '"weightUnit": data.get("weightUnit")',
        '"bodyFat": data.get("bodyFat")',
    ):
        assert line in _PLANS, f"meal_form de regen-day no propaga: {line}"
    assert '"activityLevel": data.get("activityLevel") or data.get("activity_level")' in _PLANS


def test_biometrics_live_in_meal_form_block():
    """La propagación debe vivir en el meal_form (junto a los target_*/goal), no suelta."""
    idx_bio = _PLANS.find('"weight": data.get("weight")')
    idx_goal = _PLANS.find('"goal": data.get("goal") or data.get("mainGoal")')
    idx_skip = _PLANS.find('"_skip_slot_target_override": True')
    assert idx_bio != -1 and idx_goal != -1 and idx_skip != -1
    # debe estar cerca del bloque del meal_form (goal + skip-flag del mismo dict).
    assert abs(idx_bio - idx_goal) < 1200 and abs(idx_bio - idx_skip) < 1500


def test_enrich_helper_still_hydrates_data():
    """Depende de que _enrich_clinical_from_profile siga hidratando `data` (la fuente)."""
    assert "P2-UPDATE-HYDRATE-BIOMETRICS" in _PLANS
    assert "MEALFIT_UPDATE_HYDRATE_BIOMETRICS" in _PLANS
