"""[P1-BARIATRIC-KCAL-CEILING · 2026-06-28] El target calórico = TDEE completo (ej. 2203/95kg) ignoraba la capacidad del
pouch (~150-200mL) → el plan entregaba ~1700 kcal → band kcal 0.333 penalizaba un DENOMINADOR irreal. El techo lo alinea
a la realidad fisiológica. Subir el band aquí es CLÍNICAMENTE correcto: el target irreal era el bug, no el plan.

Calibrado por review clínica adversaria (ASMBS): default 2000 (NO 1800, para no inducir déficit/sarcopenia en el
bariátrico estable en mantenimiento legítimo) + VETO de metas de superávit (gain_muscle/performance) + veto embarazo/menor
+ nunca bajo el min_kcal + nunca sube kcal.
"""
from __future__ import annotations

from pathlib import Path

import nutrition_calculator as nc

_SRC = (Path(nc.__file__).resolve().parent / "nutrition_calculator.py").read_text(encoding="utf-8")


def _baria(extra=None):
    f = {"weight": 95, "height": 160, "age": 40, "gender": "female", "weightUnit": "kg",
         "activityLevel": "light", "mainGoal": "maintenance", "medicalConditions": ["Cirugía Bariátrica"]}
    if extra:
        f.update(extra)
    return f


def test_anchor_present():
    assert "P1-BARIATRIC-KCAL-CEILING" in _SRC
    assert "MEALFIT_BARIATRIC_KCAL_CEILING_KCAL" in _SRC
    assert "MEALFIT_BARIATRIC_KCAL_CEILING_ENABLED" in _SRC


def test_knob_registered():
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_BARIATRIC_KCAL_CEILING_KCAL" in snap


def test_ceiling_applied_high_tdee_bariatric():
    r = nc.get_nutrition_targets(_baria())
    assert r.get("bariatric_kcal_ceiling_applied", {}).get("applied") is True
    assert r["target_calories"] <= 2000
    assert r["total_daily_calories"] <= 2000          # original_target_calories (Dashboard) TAMBIÉN capeado
    assert r["tdee"] >= 2000                            # TDEE educativo INTACTO
    # OBLIGATORIO clínico: la proteína absoluta no cae bajo el piso post-bariátrico al recomputar sobre menos kcal
    assert r["macros"]["protein_g"] >= 60


def test_not_applied_non_bariatric():
    r = nc.get_nutrition_targets(_baria({"medicalConditions": ["Ninguna"]}))
    assert r.get("bariatric_kcal_ceiling_applied") is None  # cero regresión no-bariátrico


def test_surplus_goal_vetoed():
    # gain_muscle (superávit) NO debe capearse → no convertir un superávit pedido en déficit (iatrogénico)
    r = nc.get_nutrition_targets(_baria({"mainGoal": "gain_muscle"}))
    assert r.get("bariatric_kcal_ceiling_applied") is None
    assert r["target_calories"] > 2000


def test_lose_fat_not_re_capped():
    # ya en déficit (target < techo) → el techo NUNCA sube kcal ni re-capea
    r = nc.get_nutrition_targets(_baria({"mainGoal": "lose_fat"}))
    assert r.get("bariatric_kcal_ceiling_applied") is None
    assert r["target_calories"] < 2000


def test_stable_thin_bariatric_not_over_restricted():
    # caso clínico (c): bariátrico delgado/estable en mantenimiento legítimo (TDEE bajo) → NO capear (evita sarcopenia)
    r = nc.get_nutrition_targets(_baria({"weight": 60, "height": 165, "activityLevel": "sedentary"}))
    assert r.get("bariatric_kcal_ceiling_applied") is None
    assert r["target_calories"] >= 1200  # piso clínico mujer respetado


def test_default_ceiling_is_2000():
    # el default clínico-conservador es 2000 (NO 1800)
    assert nc.BARIATRIC_KCAL_CEILING_KCAL == 2000
