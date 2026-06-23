"""[P5-CASABE-UNIT-WEIGHT · 2026-06-23] El pantry guard valoraba "1 unidad de Casabe"
como 400g (peso de una yuca entera, heredado vía el sinónimo casabe→yuca) → over_limit
falso → retry innecesario en los swaps. Una hoja de casabe ≈ 20g. El override de peso-por-unidad
físico corrige la matemática del guard SIN romper el sinónimo nutricional casabe→yuca.

Observado en prod: swap cravings corr=ae089712 (Yogurt Griego → Merienda), 19.73s/1 retry,
'over_limit=1' = "[1 unidad de Casabe (aprox. 30g)] (Pediste 1.0 unidad, convertido
dinámicamente excede tu inventario de 0.62 lbs)".
"""
import pytest
from constants import (
    _get_converted_quantity,
    _PHYSICAL_UNIT_WEIGHT_OVERRIDES,
    validate_ingredients_against_pantry,
    normalize_ingredient_for_tracking,
)


def test_synonym_casabe_to_yuca_preserved():
    """El fix NO debe tocar el sinónimo nutricional: casabe sigue normalizando a yuca."""
    assert normalize_ingredient_for_tracking("casabe") == "yuca"
    assert normalize_ingredient_for_tracking("1 unidad de Casabe") == "yuca"


def test_override_registered():
    assert _PHYSICAL_UNIT_WEIGHT_OVERRIDES.get("casabe") == 20.0


def test_converted_quantity_uses_yuca_weight_without_original_name():
    """Sin original_name (legacy), 1 unidad bajo base 'yuca' = 400g (raíz entera)."""
    assert _get_converted_quantity(1.0, "unidad", "g", "yuca") == pytest.approx(400.0)


def test_converted_quantity_override_for_casabe():
    """Con original_name='Casabe...', el peso físico (20g) gana sobre yuca (400g)."""
    got = _get_converted_quantity(1.0, "unidad", "g", "yuca", original_name="Casabe (aprox. 30g, una hoja pequeña)")
    assert got == pytest.approx(20.0), f"esperaba 20g (hoja de casabe), obtuvo {got}"


def test_guard_no_false_over_limit_for_casabe_unit():
    """E2E: 1 unidad de casabe vs un paquete de casabe (~281g) NO debe rechazarse por over_limit."""
    generated = ["1 unidad de Casabe"]
    pantry = ["281 g de Casabe"]  # ≈ 0.62 lb, el paquete de 10 oz
    result = validate_ingredients_against_pantry(generated, pantry, strict_quantities=True)
    # True = aprobado; un str de error contendría "CANTIDADES"/"over"
    assert result is True, f"casabe 1 unidad no debe exceder el paquete; guard devolvió: {result!r}"


def test_guard_still_rejects_genuinely_excessive_casabe():
    """El override NO debe volver el guard permisivo: pedir mucho más que el paquete sí falla."""
    generated = ["40 unidades de Casabe"]  # 40 × 20g = 800g >> 281g × 1.30
    pantry = ["281 g de Casabe"]
    result = validate_ingredients_against_pantry(generated, pantry, strict_quantities=True)
    assert isinstance(result, str), "pedir 40 hojas (800g) sobre un paquete de 281g DEBE rechazarse"
