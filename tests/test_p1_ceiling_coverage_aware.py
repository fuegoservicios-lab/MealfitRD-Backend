"""[P1-CEILING-COVERAGE-AWARE · 2026-06-15] Techo de micros consciente de cobertura POR-NUTRIENTE (gap-audit G5).

Bug original (gap-audit 2026-06-15, G5):
    En `build_micronutrient_report`, el path de TECHO hacía `status = "alto" if val > ceil else "ok"`,
    sin coverage-awareness. Las filas MANUAL de embutidos DD (Salami/Longaniza/Queso de hoja) tenían
    `saturated_fat_g` NULL → resuelven (tienen macros) pero aportan 0.0 al satfat → el total se subestima
    → un plan rico en embutidos reportaba satfat 'ok' a un DISLIPIDÉMICO (falso-negativo silencioso, sobre
    exactamente la condición que la regla existe para cazar). El path de PISO ya era coverage-aware
    ('estimado_bajo'); el de techo no — y la dirección peligrosa de un techo es la SUB-estimación.

Cierre:
    Cobertura POR-NUTRIENTE (fracción de ingredientes resueltos con dato NO-NULL de ese micro). Si un techo
    en apariencia 'ok' tiene cobertura por-nutriente < 0.6 → status 'estimado_alto' (incierto + caveat),
    NUNCA 'ok' liso. Simétrico al 'estimado_bajo' de los pisos. (El dato de los embutidos también se pobló
    en populate_nutrition_db.py — MANUAL_MICROS — pero el código coverage-aware es el backstop robusto
    independiente de que el dato esté completo.)

Test funcional rápido (stub db, sin LLM/Neon).
"""
from __future__ import annotations

import pytest

import micronutrients as mn


_MICRO_KEYS = ("fiber", "sodium_mg", "vit_d_mcg", "calcium_mg", "iron_mg", "b12_mcg",
               "potassium_mg", "magnesium_mg", "saturated_fat_g", "sugars_g")


def _micros(**overrides):
    base = {k: 0.0 for k in _MICRO_KEYS}
    base.update(overrides)
    return base


class _StubDB:
    """satfat NULL para 'salami' (resuelve pero sin dato → el bug); presente para el resto."""
    def __init__(self, salami_satfat=None, other_satfat=1.0):
        self.salami_satfat = salami_satfat
        self.other_satfat = other_satfat

    def micros_from_ingredient_string(self, s):
        s = str(s).lower()
        if "salami" in s:
            return _micros(sodium_mg=500.0, iron_mg=1.5, b12_mcg=1.6,
                           saturated_fat_g=self.salami_satfat)  # None = NULL en catálogo
        return _micros(fiber=2.0, sodium_mg=50.0, saturated_fat_g=self.other_satfat)


def _plan(*ingredients):
    return {"days": [{"meals": [{"ingredients": list(ingredients)}]}]}


def test_satfat_estimado_alto_when_partial_nutrient_coverage():
    """Salami con satfat NULL + lechuga con satfat → cobertura satfat 0.5 < 0.6 y val ≤ techo →
    debe ser 'estimado_alto', NO 'ok' (cierre del falso-negativo silencioso)."""
    db = _StubDB(salami_satfat=None, other_satfat=1.0)
    rep = mn.build_micronutrient_report(_plan("100g salami", "100g lechuga"), db,
                                        sex="M", conditions=["dislipidemia"], daily_kcal=2000)
    satfat = next(e for e in rep["panel"] if e["key"] == "saturated_fat_g")
    assert satfat["status"] == "estimado_alto", (
        f"satfat debió ser 'estimado_alto' (cobertura parcial), no {satfat['status']!r} — "
        "el techo subestimado NO debe reportar 'ok' a un dislipidémico."
    )
    assert satfat.get("cobertura_nutriente") == 0.5
    assert satfat.get("nota"), "estimado_alto debe llevar un caveat honesto."
    assert any(g["key"] == "saturated_fat_g" for g in rep["gaps"]), "debe entrar a gaps (surfaced)."


def test_satfat_ok_when_full_coverage_and_under_ceiling():
    db = _StubDB(salami_satfat=2.0, other_satfat=1.0)  # ambos con dato → cobertura 1.0
    rep = mn.build_micronutrient_report(_plan("100g salami", "100g lechuga"), db,
                                        sex="M", conditions=["dislipidemia"], daily_kcal=2000)
    satfat = next(e for e in rep["panel"] if e["key"] == "saturated_fat_g")
    assert satfat["status"] == "ok", "con cobertura completa y bajo el techo debe ser 'ok'."
    assert satfat.get("cobertura_nutriente") == 1.0


def test_satfat_alto_when_over_ceiling_regardless_of_coverage():
    db = _StubDB(salami_satfat=40.0, other_satfat=40.0)  # total muy alto
    rep = mn.build_micronutrient_report(_plan("100g salami", "100g lechuga"), db,
                                        sex="M", conditions=["dislipidemia"], daily_kcal=2000)
    satfat = next(e for e in rep["panel"] if e["key"] == "saturated_fat_g")
    assert satfat["status"] == "alto", "sobre el techo siempre es 'alto' (no se enmascara como estimado)."


def test_nutrient_coverage_distinguishes_null_from_zero():
    """compute_plan_micronutrient_totals debe contar presencia NO-NULL por micro (no confundir
    NULL con 0 real)."""
    db = _StubDB(salami_satfat=None, other_satfat=1.0)
    totals = mn.compute_plan_micronutrient_totals(_plan("100g salami", "100g lechuga"), db)
    ncov = totals["nutrient_coverage"]
    assert ncov["saturated_fat_g"] == 0.5, "1 de 2 resueltos trae satfat → cobertura 0.5."
    assert ncov["sodium_mg"] == 1.0, "ambos traen sodio → cobertura 1.0."
