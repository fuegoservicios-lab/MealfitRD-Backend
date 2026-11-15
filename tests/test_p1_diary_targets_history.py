"""[P1-DIARY-DAY-TARGETS] Metas históricas por día en el diario de comidas.

Cubre:
1. Reconstrucción de timeline con cambios de plan y objetivo entre fechas
   (ej. jueves 3350 kcal -> viernes 2050 kcal) sin reescribir días pasados.
2. Precedencia de snapshots en app_kv_store sobre meal_plans en la misma fecha.
3. No fabricación de meta para fechas anteriores al primer evento registrado.
4. Contratos de limpieza GDPR/borrado de cuenta (_USER_SCOPED_KV_PREFIXES).
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import db_profiles  # noqa: E402
from diary_targets import (  # noqa: E402
    KV_PREFIX,
    extract_targets_from_plan_data,
    resolve_targets_timeline,
)


def test_timeline_preserva_meta_historica_al_cambiar_de_plan():
    # Caso reportado: jueves tenía plan de 3350 kcal (ganar músculo), viernes cambió a 2050 kcal (perder grasa).
    # Jueves consumió 3082 kcal -> 92% de 3350 (dentro de banda verde), no 150% de 2050.
    plans = [
        {
            "dia": "2026-11-09",
            "calories": "3350",
            "macros": {"protein": "180g", "carbs": "380g", "fats": "100g"},
            "main_goal": "gain_muscle",
        },
        {
            "dia": "2026-11-13",
            "calories": "2050",
            "macros": {"protein": "134g", "carbs": "251g", "fats": "57g"},
            "main_goal": "lose_fat",
        },
    ]

    res = resolve_targets_timeline(
        ["2026-11-08", "2026-11-09", "2026-11-12", "2026-11-13"],
        plan_rows=plans,
    )

    # Fecha previa al primer plan no inventa una meta futura
    assert "2026-11-08" not in res

    assert res["2026-11-09"]["calories"] == 3350
    assert res["2026-11-09"]["main_goal"] == "gain_muscle"
    assert res["2026-11-12"]["calories"] == 3350
    assert res["2026-11-12"]["macros"] == {"protein": 180, "carbs": 380, "fats": 100}
    assert res["2026-11-12"]["main_goal"] == "gain_muscle"

    assert res["2026-11-13"]["calories"] == 2050
    assert res["2026-11-13"]["macros"] == {"protein": 134, "carbs": 251, "fats": 57}
    assert res["2026-11-13"]["main_goal"] == "lose_fat"


def test_kv_snapshot_sobreescribe_misma_fecha_sin_tocar_dias_anteriores():
    plans = [
        {"dia": "2026-11-10", "calories": 3200, "main_goal": "gain_muscle"},
    ]
    kv_hist = {
        "days": {
            "2026-11-10": {"calories": 3350, "macros": {"protein": 160, "carbs": 400, "fats": 95}, "main_goal": "gain_muscle"},
            "2026-11-13": {"calories": 2050, "macros": {"protein": 134, "carbs": 251, "fats": 57}, "main_goal": "lose_fat"},
        }
    }

    res = resolve_targets_timeline(["2026-11-10", "2026-11-12", "2026-11-13"], plan_rows=plans, kv_hist=kv_hist)

    assert res["2026-11-10"]["calories"] == 3350
    assert res["2026-11-12"]["calories"] == 3350
    assert res["2026-11-13"]["calories"] == 2050


def test_extract_targets_from_plan_data_soporta_nested_assessment_y_strings():
    pd = {
        "calories": "2,050 kcal",
        "macros": {"protein": "134g", "carbs": "251g", "fats": "57g"},
        "assessment": {"mainGoal": "Lose_Fat"},
    }
    snap = extract_targets_from_plan_data(pd)
    assert snap == {
        "calories": 2050,
        "macros": {"protein": 134, "carbs": 251, "fats": 57},
        "main_goal": "lose_fat",
    }


def test_kv_prefix_incluido_en_borrado_de_cuenta():
    assert KV_PREFIX in db_profiles._USER_SCOPED_KV_PREFIXES
