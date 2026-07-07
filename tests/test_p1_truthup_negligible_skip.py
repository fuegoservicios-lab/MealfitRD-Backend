"""[P1-TRUTHUP-NEGLIGIBLE-SKIP · 2026-07-07] Bug PROFUNDO del solver (forense plan vivo 766893f4):

El "Bowl de Soya Ceviche" reportaba 8g grasa / 576 kcal pero sus ingredient-strings suman 41g / 935 kcal
— 34g de grasa del aguacate (1½ aguacates = 221g) SIN CONTAR. El usuario cree comer 576 kcal y come 935.

Causa raíz: `_truth_up_meal_macros_from_strings` tenía un gate conservador que ABORTA el meal completo si
UN ingrediente resuelve por nombre pero su cantidad no convierte a gramos. El "0.25 taza de cilantro"
(hierba sin density_g_per_cup → gramos=None) disparaba el abort → los macros nunca se refrescaban desde
los strings finales → la grasa del aguacate quedaba invisible para relevel/banda/gate (que leen macros STORED).

Fix (2 partes):
- P1-TRUTHUP-NEGLIGIBLE-SKIP: el gate salta ingredientes no-convertibles de aporte macro DESPRECIABLE
  (hierba/hoja: kcal≤40 y grasa≤1.5 por-100g) en vez de abortar. Los significativos siguen abortando.
- P1-TRUTHUP-BEFORE-RELEVEL: `_relevel_fats_universal` refresca macros desde strings ANTES de medir el día
  (relevel usa `sum(m.get("fats"))` stored) → deja de estar ciego al aguacate under-contado.
"""
import os
import re

import graph_orchestrator as g

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


class _Info:
    def __init__(self, kcal, fats):
        self.kcal = kcal
        self.fats = fats


class _DB:
    """cilantro/lata = cantidad no-convertible (gramos None). aguacate con hint = convertible."""
    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "cilantro" in low or "lata" in low:  # cantidad no-convertible
            return None
        mg = re.search(r"\((\d+(?:\.\d+)?)\s*g", low) or re.match(r"^\s*(\d+(?:\.\d+)?)\s*g\b", low)
        if not mg:
            return None
        gr = float(mg.group(1))
        if "aguacate" in low:
            return {"protein": gr * 0.02, "carbs": gr * 0.09, "fats": gr * 0.15, "kcal": gr * 1.6}
        return {"protein": gr * 0.10, "carbs": gr * 0.10, "fats": gr * 0.01, "kcal": gr * 1.0}

    def lookup(self, s):
        low = str(s).lower()
        if "cilantro" in low or "perejil" in low or "lechuga" in low:
            return _Info(23, 0.5)          # hierba/hoja → DESPRECIABLE
        if "aguacate" in low:
            return _Info(160, 15)          # significativo
        if "atun" in low or "atún" in low or "lata" in low:
            return _Info(130, 5)           # proteína enlatada → significativo
        return None


def _bowl():
    # 65g soya (hint), 221g aguacate (hint), cilantro en taza (NO convertible)
    return {"name": "Bowl de Soya Ceviche", "meal": "Almuerzo",
            "ingredients": ["65 g de soya texturizada", "1 aguacate (221 g)",
                            "0.25 taza de cilantro fresco picado"],
            "ingredients_raw": ["65 g de soya texturizada", "1 aguacate (221 g)",
                                "0.25 taza de cilantro fresco picado"],
            "protein": 41, "carbs": 83, "fats": 8, "cals": 576}  # STALE: aguacate sin contar


# ───────────────────────── gate: herbs no abortan ─────────────────────────

def test_herb_no_longer_aborts_truthup_counts_avocado():
    """El cilantro en taza (hierba negligible) ya NO aborta → el aguacate (34g grasa) se cuenta."""
    meal = _bowl()
    changed = g._truth_up_meal_macros_from_strings(meal, _DB())
    assert changed is True, "truth-up debe reescribir (antes abortaba por el cilantro)"
    # 221g aguacate × 0.15 = ~33g grasa + soya ~0.6 → ~33-34g (NO los 8g stale)
    assert meal["fats"] >= 30, f"grasa honesta debe contar el aguacate, dio {meal['fats']}"
    assert meal["fats"] != 8


def test_significant_unconvertible_still_aborts():
    """Una proteína enlatada (atún en lata, kcal 130/100g) sin peso convertible SIGUE abortando
    (preserva la intención conservadora: no under-contar masa real significativa)."""
    meal = {"name": "Ensalada de Atún", "meal": "Almuerzo",
            "ingredients": ["1 lata de atún", "1 aguacate (150 g)"],
            "ingredients_raw": ["1 lata de atún", "1 aguacate (150 g)"],
            "protein": 30, "carbs": 5, "fats": 20, "cals": 300}
    before = (meal["protein"], meal["carbs"], meal["fats"])
    changed = g._truth_up_meal_macros_from_strings(meal, _DB())
    assert changed is False, "atún-en-lata significativo no-convertible → abort conservador"
    assert (meal["protein"], meal["carbs"], meal["fats"]) == before  # macros intactos


def test_negligible_thresholds_gate(monkeypatch):
    """Con umbrales en 0 (todo es 'significativo'), el cilantro vuelve a abortar (comportamiento previo)."""
    monkeypatch.setattr(g, "TRUTHUP_NEGLIGIBLE_KCAL_100", 0.0)
    monkeypatch.setattr(g, "TRUTHUP_NEGLIGIBLE_FAT_100", 0.0)
    meal = _bowl()
    assert g._truth_up_meal_macros_from_strings(meal, _DB()) is False
    assert meal["fats"] == 8  # intacto (abortó)


def test_wholly_negligible_meal_still_resolves():
    """Meal solo de resolubles convertibles → truth-up normal (no regresión)."""
    meal = {"name": "X", "ingredients": ["100 g de soya texturizada"],
            "ingredients_raw": ["100 g de soya texturizada"],
            "protein": 1, "carbs": 1, "fats": 1, "cals": 1}
    assert g._truth_up_meal_macros_from_strings(meal, _DB()) is True
    assert meal["protein"] == 10  # 100g × 0.10


# ───────────────────────── relevel refresca macros antes de medir ─────────────────────────

def test_relevel_truthups_before_measuring():
    """Estructural: `_relevel_fats_universal` corre truth-up de todos los meals ANTES del trim loop."""
    i_def = _GO.find("def _relevel_fats_universal(")
    assert i_def != -1
    seg = _GO[i_def:i_def + 3000]
    assert "TRUTHUP_BEFORE_RELEVEL_ENABLED" in seg
    i_truthup = seg.find("_truth_up_meal_macros_from_strings(_m, db)")
    i_trim = seg.find("_trim_day_fats_to_target(_meals")
    assert i_truthup != -1 and i_trim != -1 and i_truthup < i_trim, \
        "el truth-up debe correr ANTES del trim (si no, relevel mide macros stale)"


def test_relevel_truthup_knob_default_on():
    assert g.TRUTHUP_BEFORE_RELEVEL_ENABLED is True
    assert 'TRUTHUP_BEFORE_RELEVEL_ENABLED = _env_bool("MEALFIT_TRUTHUP_BEFORE_RELEVEL", True)' in _GO


def test_markers_anchored():
    assert "P1-TRUTHUP-NEGLIGIBLE-SKIP" in _GO
    assert "P1-TRUTHUP-BEFORE-RELEVEL" in _GO
    assert 'TRUTHUP_NEGLIGIBLE_KCAL_100 = _env_float("MEALFIT_TRUTHUP_NEGLIGIBLE_KCAL_100"' in _GO
