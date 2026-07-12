"""[P1-SLOT-FATS-FLOOR · 2026-07-12] El retarget del regen-día no asigna slots de grasa imposibles.

Vivo ×2 (planes df263d1b, regens 05:5x y 06:34Z): el retarget proporcional asignó a un
almuerzo de 736 kcal un target de grasa de 6g. Comida criolla no cabe ahí: el candidato
honesto trae 20-22g (truth-up + fats-trim solo bajan hasta la grasa intrínseca de los
alimentos), el validador rechaza los 3 intentos (drift 2.667) y el slot se CONSERVA —
retries LLM quemados sin salida posible ("gate sin corrector = rechazo incorregible").

Fix: piso `SLOT_FATS_FLOOR_G` (default 10g, knob) para slots con target ≥`MIN_KCAL`
(default 450). El día no se desbanda: P2-REGEN-DAY-FATS-RELEVEL (shrink-only) recorta el
exceso agregado post-rebalance. Contrato parser sobre routers/plans.py.
tooltip-anchor: P1-SLOT-FATS-FLOOR
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8") as f:
    _PL = f.read()


def test_knobs_registered_with_safe_defaults():
    assert re.search(
        r'SLOT_FATS_FLOOR_G = _env_int\(\s*"MEALFIT_SLOT_FATS_FLOOR_G", 10', _PL), \
        "knob del piso (default 10g) via _env_int (auto-registro en _KNOBS_REGISTRY)"
    assert re.search(
        r'"MEALFIT_SLOT_FATS_FLOOR_MIN_KCAL", 450', _PL), \
        "umbral de comida fuerte (default 450 kcal)"
    assert "0 <= v <= 25" in _PL, "clamp del piso (0 desactiva, techo 25g)"


def test_floor_applied_before_meal_form():
    # OJO: el marker aparece 2 veces (bloque knob a nivel módulo + bloque del loop).
    # Anclamos en el CÓDIGO del loop (no en el comment) — lección parser-decoy.
    i = _PL.find('_sff_cals = round(float(meal.get("cals")')
    assert i != -1, "bloque del piso desapareció del regen-día"
    win = _PL[max(0, i - 1000):i + 1600]
    assert "_sff_fats = float(SLOT_FATS_FLOOR_G)" in win, "el clamp asigna el piso"
    assert "0 < float(_sff_fats or 0) < float(SLOT_FATS_FLOOR_G)" in win, \
        "solo eleva targets POSITIVOS bajo el piso (target 0/None no se inventa)"
    assert "float(SLOT_FATS_FLOOR_MIN_KCAL)" in win, "gate por kcal del slot (comida fuerte)"
    # el meal_form consume las variables floored (no la fórmula cruda):
    j = _PL.find('"target_fats": _sff_fats', i)
    assert j != -1 and j - i < 4000, "target_fats del meal_form usa el valor con piso"
    assert '"target_calories": _sff_cals' in _PL[i:j + 200]


def test_functional_clamp_semantics():
    """Réplica del clamp (misma expresión del source) sobre los casos vivo + bordes."""
    FLOOR, MIN_KCAL = 10, 450

    def clamp(cals, fats):
        if FLOOR > 0 and float(cals or 0) >= float(MIN_KCAL) and 0 < float(fats or 0) < float(FLOOR):
            return float(FLOOR)
        return fats

    assert clamp(736, 6) == 10.0, "caso vivo: almuerzo 736 kcal / 6g → piso"
    assert clamp(736, 15) == 15, "sobre el piso → intacto"
    assert clamp(200, 3) == 3, "merienda ligera → sin piso (batidos/snacks pueden ser magros)"
    assert clamp(736, 0) == 0, "target 0/None no se inventa (slot sin dato)"
    assert clamp(None, 6) == 6, "sin kcal → conservador, no aplica"
