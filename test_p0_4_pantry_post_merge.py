"""
Tests P0-4: Hard validation post-merge contra la nevera.

Cubre:
1. _validate_merged_days_against_pantry — el helper directo:
   a. Si pantry está vacío, no valida (legacy behavior).
   b. Si todos los días pasan, retorna (True, []).
   c. Si algún día tiene un ingrediente core faltante, retorna (False, [violation,...]).
   d. Si pasamos new_chunk_day_range, días fuera del rango se ignoran.
   e. Condimentos básicos (sal, ajo, aceite) no fallan aunque no estén en pantry.

2. _PantryViolationPostMerge — la excepción:
   a. Atributos preservados (violations, pantry_size).
"""
from cron_tasks import (
    _validate_merged_days_against_pantry,
    _PantryViolationPostMerge,
)


def _day(day_num, ingredients_per_meal):
    """Helper para construir un día con N comidas, cada una con sus ingredientes."""
    return {
        "day": day_num,
        "meals": [
            {"name": f"meal_{i}", "ingredients": ings}
            for i, ings in enumerate(ingredients_per_meal)
        ],
    }


def test_validate_returns_true_when_pantry_empty():
    """Sin pantry conocido, no podemos validar — comportamiento legacy."""
    days = [_day(1, [["1000g pollo", "200g arroz"]])]
    ok, violations = _validate_merged_days_against_pantry(days, pantry_ingredients=[])
    assert ok is True
    assert violations == []


def test_validate_passes_when_all_ingredients_in_pantry():
    """Cuando los ingredientes del día están cubiertos por la nevera, ok=True."""
    pantry = ["2000g pollo", "1000g arroz", "500g cebolla"]
    days = [
        _day(1, [["100g pollo", "100g arroz"]]),
        _day(2, [["100g pollo", "50g cebolla"]]),
    ]
    ok, violations = _validate_merged_days_against_pantry(days, pantry)
    assert ok is True, f"Esperaba ok=True, pero hubo violaciones: {violations}"


def test_validate_fails_when_core_ingredient_missing():
    """Si un día requiere un ingrediente core que no está en la nevera, ok=False."""
    pantry = ["1000g arroz", "500g cebolla"]  # NO hay camarones
    days = [
        _day(1, [["100g camarones", "100g arroz"]]),  # camarones no está
    ]
    ok, violations = _validate_merged_days_against_pantry(days, pantry)
    assert ok is False
    assert len(violations) == 1
    assert violations[0]["day"] == 1
    assert "camarones" in violations[0]["error"].lower() or "INEXISTENTES" in violations[0]["error"]


def test_validate_fails_when_quantity_exceeds_pantry():
    """Si un día pide más cantidad de la disponible (sin tolerancia), ok=False."""
    pantry = ["100g pollo"]  # solo 100g
    days = [
        _day(1, [["500g pollo asado"]]),  # pide 500g
    ]
    ok, violations = _validate_merged_days_against_pantry(days, pantry)
    assert ok is False
    assert len(violations) == 1


def test_validate_ignores_basic_condiments():
    """Los condimentos básicos (sal, ajo, aceite, etc.) no fallan aunque no estén
    en la nevera — están en allowed_condiments dentro de validate_ingredients_against_pantry."""
    pantry = ["1000g pollo", "500g arroz"]  # sin sal, sin aceite, sin ajo
    days = [
        _day(1, [["100g pollo", "100g arroz", "1 pizca sal", "1 cucharada aceite", "2 dientes ajo"]]),
    ]
    ok, violations = _validate_merged_days_against_pantry(days, pantry)
    assert ok is True, f"Condimentos básicos no deben fallar. Violaciones: {violations}"


def test_validate_only_checks_days_in_range():
    """Si pasamos new_chunk_day_range, días fuera del rango se ignoran aunque
    contengan ingredientes problemáticos."""
    pantry = ["1000g pollo"]  # NO hay arroz
    days = [
        _day(1, [["200g arroz"]]),  # día 1 violaría, pero está fuera de rango
        _day(2, [["200g arroz"]]),  # día 2 violaría, pero está fuera de rango
        _day(3, [["100g pollo"]]),  # día 3 (en rango) está OK
    ]
    # Solo validamos día 3 (el chunk recién mergeado)
    ok, violations = _validate_merged_days_against_pantry(
        days, pantry, new_chunk_day_range=(3, 3)
    )
    assert ok is True, f"Días fuera del rango deben ignorarse. Violaciones: {violations}"


def test_validate_catches_violation_in_range():
    """Si new_chunk_day_range contiene un día con violación, debe rechazar."""
    pantry = ["1000g pollo"]
    # Usamos "camarones" en lugar de proteínas similares a "pollo" para evitar que
    # el matcher por similitud cosine las empareje (salmón/pescado/etc tienen score
    # alto contra pollo y validate_ingredients_against_pantry los aprueba).
    days = [
        _day(1, [["100g pollo"]]),  # OK
        _day(2, [["100g pollo"]]),  # OK
        _day(3, [["200g camarones"]]),  # VIOLACIÓN: camarones no están
    ]
    ok, violations = _validate_merged_days_against_pantry(
        days, pantry, new_chunk_day_range=(3, 3)
    )
    assert ok is False
    assert len(violations) == 1
    assert violations[0]["day"] == 3


def test_validate_skips_days_with_no_meals():
    """Días sin comidas o sin ingredientes se saltan sin violar."""
    pantry = ["1000g pollo"]
    days = [
        {"day": 1, "meals": []},  # sin comidas
        {"day": 2, "meals": [{"name": "x", "ingredients": []}]},  # sin ingredientes
        _day(3, [["100g pollo"]]),
    ]
    ok, violations = _validate_merged_days_against_pantry(days, pantry)
    assert ok is True


def test_validate_handles_malformed_days_gracefully():
    """Días no-dict o con day=None no deben crashear."""
    pantry = ["1000g pollo"]
    days = [
        None,  # no es dict
        "string",  # no es dict
        {"day": None, "meals": [{"name": "x", "ingredients": ["100g pollo"]}]},  # day None
        _day(1, [["100g pollo"]]),
    ]
    ok, violations = _validate_merged_days_against_pantry(days, pantry)
    # No crashea; los None/string se saltan, day=None usa 0 (queda fuera del default)
    assert ok is True


def test_pantry_violation_exception_preserves_data():
    """La excepción guarda violations y pantry_size para que el outer catch los persista."""
    violations = [
        {"day": 4, "error": "INEXISTENTES en inventario: salmón."},
        {"day": 5, "error": "CANTIDADES: pollo (Pediste 500g, límite: 200g)."},
    ]
    exc = _PantryViolationPostMerge(violations, pantry_size=12)
    assert exc.violations == violations
    assert exc.pantry_size == 12
    # Mensaje legible para logs
    assert "2 día" in str(exc) or "2 dia" in str(exc)


def test_pantry_violation_exception_handles_none_args():
    """Args defensivos: None violations o pantry_size deben no crashear."""
    exc = _PantryViolationPostMerge(None, pantry_size=None)
    assert exc.violations == []
    assert exc.pantry_size == 0
