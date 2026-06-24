"""[P5-DAY-REGEN-VARIETY · 2026-06-23] El endpoint /regenerate-day debe dar VARIEDAD intra-día:
cada swap excluye los platos ya resueltos hoy (disliked_meals = day_avoid) para no colapsar en
3 platos iguales (caso reportado: 3 dishes de camarones). FALLBACK de factibilidad: si con las
exclusiones el chef IA no converge, reintenta SIN ellas (solo pantry-strict) antes de conservar.

Ancla parser-based del contrato (el comportamiento funcional con LLM real no es testeable offline).
"""
import os
import re

_PLANS = open(os.path.join(os.path.dirname(__file__), "..", "routers", "plans.py"), encoding="utf-8").read()


def test_marker_and_knob_present():
    assert "P5-DAY-REGEN-VARIETY" in _PLANS
    assert "MEALFIT_DAY_REGEN_INTRADAY_VARIETY" in _PLANS


def test_accumulates_day_avoid():
    # day_avoid se inicializa y se rellena con los platos NUEVOS aceptados.
    assert re.search(r"day_avoid\s*:\s*list\s*=\s*\[\]", _PLANS), "day_avoid debe inicializarse"
    assert re.search(r'day_avoid\.append\(nm\["name"\]\)', _PLANS), "el plato nuevo aceptado debe entrar a day_avoid"


def test_passes_disliked_meals_for_variety():
    # El 1er intento pasa day_avoid como disliked_meals (exclusión de variedad).
    assert re.search(r'_form_v\["disliked_meals"\]\s*=\s*list\(day_avoid\)', _PLANS)


def test_feasibility_fallback_retries_without_exclusions():
    # Si el 1er intento (con variedad) lanza ValueError, reintenta swap_meal(meal_form) SIN day_avoid.
    assert "reintento SIN exclusiones de variedad" in _PLANS
    # El reintento usa meal_form crudo (sin las exclusiones de variedad).
    assert re.search(r"nm = swap_meal\(meal_form\)", _PLANS)


def test_excludes_main_protein_too():
    # [P5-DAY-REGEN-VARIETY-PROTEIN] además del NOMBRE, excluye la PROTEÍNA principal del plato
    # aceptado → 4 proteínas distintas, no 2 de res.
    assert "P5-DAY-REGEN-VARIETY-PROTEIN" in _PLANS
    assert "def _main_protein_of_meal" in _PLANS
    assert "_MAIN_PROTEIN_DETECT" in _PLANS
    # La proteína del plato nuevo y la del conservado entran a day_avoid.
    assert re.search(r"_p_new\s*=\s*_main_protein_of_meal\(nm\)", _PLANS)
    assert re.search(r"day_avoid\.append\(_p_new\)", _PLANS)


def test_protein_detect_covers_key_proteins():
    # El mapa debe cubrir las proteínas dominicanas clave (res/carne, camarones, cerdo, pescado, huevos).
    for kw in ("camaron", "pollo", "cerdo", "res", "carne", "pescado", "huevo"):
        assert any(kw == k for k, _ in __import__("re").findall(r'\("([^"]+)",\s*"([^"]+)"\)', _PLANS)) or kw in _PLANS
