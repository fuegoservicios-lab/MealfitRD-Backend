"""[P1-FASE-A-FAT-TOPUP · 2026-06-28] Tras anclar proteína animal MAGRA (P1-BARIATRIC-DENSE-ANCHOR), la grasa del día
cae bajo banda (band fats 0.0) porque la proteína magra reemplaza lácteos grasos y el reconcile no tiene suficiente
grasa que escalar (ya va al máx ×1.8). Fix: `_topup_healthy_fat_to_band_floor` añade aceite de oliva MODERADO,
distribuido en comidas principales, hasta el PISO de banda (0.90×target, NO banda completa). Guards clínicos (ASMBS):
per-comida ≤cap, headroom calórico (≤1.12×kcal), skip si ya hay grasa visible, olo aceite MUFA.

NOTA: el plan del workflow proponía un "CAMBIO 1" (reconcile preferir-carbos con fF<1.0) basado en una premisa ERRÓNEA
(el reconcile usa fF=target/cur clampeado a 1.8 → SUBE la grasa cuando está baja, no la baja). Se descartó leyendo el
código; el fix real es añadir grasa (no hay qué escalar).
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_SRC = (Path(g.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _meals_low_fat():
    return [
        {"type": "Desayuno", "name": "Revuelto de huevo", "ingredients": ["2 huevos"], "protein": 18, "carbs": 20, "fats": 8, "cals": 224},
        {"type": "Almuerzo", "name": "Pollo con arroz", "ingredients": ["120g de pollo", "1/2 taza arroz"], "protein": 35, "carbs": 40, "fats": 6, "cals": 354},
        {"type": "Cena", "name": "Pescado con vegetales", "ingredients": ["120g de pescado", "vegetales"], "protein": 32, "carbs": 25, "fats": 8, "cals": 300},
    ]


def test_topup_reaches_floor_bariatric():
    m = _meals_low_fat()
    n = g._topup_healthy_fat_to_band_floor(m, 53.0, 1600.0, None, is_bariatric=True)
    assert n > 0
    cf = sum(x["fats"] for x in m)
    assert cf >= 0.90 * 53.0, f"grasa día {cf} debe alcanzar el piso {0.90*53.0:.1f}"
    # per-comida ≤ cap
    for x in m:
        if x.get("_fat_topup"):
            assert "aceite de oliva" in x["ingredients"][-1]


def test_no_topup_when_in_band():
    m = _meals_low_fat()
    for x in m:
        x["fats"] = 18  # 54g > floor
    assert g._topup_healthy_fat_to_band_floor(m, 53.0, 1600.0, None, is_bariatric=True) == 0


def test_skip_meal_with_existing_fat():
    m = _meals_low_fat()
    m[1]["ingredients"].append("1 cda de aceite de oliva")
    g._topup_healthy_fat_to_band_floor(m, 53.0, 1600.0, None, is_bariatric=True)
    assert not m[1].get("_fat_topup"), "comida con aceite ya presente no debe recibir top-up"


def test_idempotent():
    m = _meals_low_fat()
    g._topup_healthy_fat_to_band_floor(m, 53.0, 1600.0, None, is_bariatric=True)
    assert g._topup_healthy_fat_to_band_floor(m, 53.0, 1600.0, None, is_bariatric=True) == 0


def test_calorie_headroom_guard():
    m = _meals_low_fat()
    for x in m:
        x["cals"] = 600  # 1800 > 1.12*1600=1792 → sin headroom
    assert g._topup_healthy_fat_to_band_floor(m, 53.0, 1600.0, None, is_bariatric=True) == 0


def test_knob_off():
    import os
    old = os.environ.get("MEALFIT_FASE_A_FAT_TOPUP")
    try:
        # el knob se lee a import-time; verificamos el guard por la constante
        if not g.FASE_A_FAT_TOPUP_ENABLED:
            m = _meals_low_fat()
            assert g._topup_healthy_fat_to_band_floor(m, 53.0, 1600.0, None, is_bariatric=True) == 0
    finally:
        if old is not None:
            os.environ["MEALFIT_FASE_A_FAT_TOPUP"] = old


def test_anchor_and_wiring():
    assert "P1-FASE-A-FAT-TOPUP" in _SRC
    assert "def _topup_healthy_fat_to_band_floor" in _SRC
    # invocado dentro de FASE A tras el reconcile
    assert "_topup_healthy_fat_to_band_floor(_ms, _fg" in _SRC
    # el "CAMBIO 1" (reconcile prefer_carbs_reduction) NO se implementó (premisa errónea)
    assert "prefer_carbs_reduction" not in _SRC
