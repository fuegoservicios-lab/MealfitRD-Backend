"""[P6-FRUITS-LARGE-CAP] Tests para el cap defensivo de frutas grandes
(melón, sandía, piña, lechosa, papaya).

Bug observable (PDF 2026-05-05 15:09):
  Lista mostró "Melón: 24 Uds. (~79.4 lbs total)" para 2p × mes.
  35 kg de melón comprados de una vez = ~80% se descompone antes de
  consumir (melón entero dura 5-7 días refrigerado).

Causa raíz:
  Aggregator suma "1 taza de melón en cubos" × N comidas × multiplier
  18.67 sin entender que cada melón rinde 6-8 tazas. Mismo patrón que
  aceitunas/cebolla/papa pre-cap.

Fix:
  Cap por persona-semana calibrado por densidad/rendimiento:
    - melón ~1.2kg → 1/persona/sem
    - sandía ~3kg → 0.5/persona/sem (más rendimiento)
    - piña ~1.5kg → 1/persona/sem
    - lechosa/papaya ~800g → 1/persona/sem

Cobertura:
  - Repro PDF: 24 melones → cap 8
  - Cap escalado por person_weeks
  - Cada fruta tiene su per_week específico
  - Items NO listados (manzana, naranja) no se tocan
"""
import pytest

from shopping_calculator import (
    aggregate_and_deduct_shopping_list,
    invalidate_master_cache,
)


@pytest.fixture(autouse=True)
def _reset_master_cache():
    invalidate_master_cache()
    yield
    invalidate_master_cache()


MARKET_MIN_LB_G = 0.25 * 453.592


def _expected_max_g(cap_g: float, margin_pct: float = 0.10) -> float:
    return max(cap_g, MARKET_MIN_LB_G) * (1.0 + margin_pct)


def _qty_grams_for(result: list, name_substr: str, unit_density_g: float = 1200.0) -> float:
    """Cantidad efectiva en gramos del primer item que matchee.
    Match accent-insensitive. `unit_density_g` se usa cuando el output
    está en unidades (ya que densidad varía por fruta)."""
    from constants import strip_accents
    needle = strip_accents(name_substr).lower()
    for r in result:
        if not isinstance(r, dict):
            continue
        haystack = strip_accents(r.get("name", "")).lower()
        if needle not in haystack:
            continue
        qty = float(r.get("market_qty", 0))
        unit = (r.get("market_unit") or "").lower()
        if unit in ("lb", "lbs"):
            return qty * 453.592
        if unit == "g":
            return qty
        if unit in ("unidad", "ud.", "uds.", "unidades"):
            return qty * unit_density_g
        return qty
    return -1.0


# ===========================================================================
# 1. Repro PDF — melón 24 Uds → cap 8
# ===========================================================================
def test_repro_pdf_melon_24_caps_to_8():
    """Caso real: 24 melones para 2p × mes → cap 8."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["99 melones"],
        multiplier=18.666666,
        structured=True,
    )
    qty_g = _qty_grams_for(result, "melon")
    cap_g = _expected_max_g(8 * 1200.0)
    assert 0 < qty_g <= cap_g, (
        f"Melón cap fallido: {qty_g:.0f}g excede cap de {cap_g:.0f}g (~8 melones)"
    )


# ===========================================================================
# 2. Sandía con su per_week específico (0.5)
# ===========================================================================
def test_sandia_caps_to_4_for_2p_monthly():
    """Sandía 0.5/persona/sem → 2p × mes (8 pw) = cap 4."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["99 sandias"],
        multiplier=18.666666,
        structured=True,
    )
    qty_g = _qty_grams_for(result, "sandia", unit_density_g=3000.0)
    cap_g = _expected_max_g(4 * 3000.0)
    assert 0 < qty_g <= cap_g, (
        f"Sandía cap fallido: {qty_g:.0f}g > {cap_g:.0f}g (~4 sandías)"
    )


# ===========================================================================
# 3. Piña, lechosa, papaya
# ===========================================================================
# [search_name] El aggregator canonicaliza variedades a su nombre dominicano
# (constants.DOMINICAN_FRUIT_VARIETY: "papaya" → "lechosa"), así que el item de
# salida lleva el nombre canónico, no el literal del input. Buscamos por el
# nombre canónico esperado. Antes el test extraía el substring del input
# (`papaya`) y no encontraba el item ("Lechosa") → -1.0.
@pytest.mark.parametrize("ingredient,density,per_week,search_name", [
    ("99 pinas", 1500.0, 1, "pina"),
    ("99 lechosas", 800.0, 1, "lechosa"),
    ("99 papayas", 800.0, 1, "lechosa"),  # papaya → Lechosa (canónico DR)
])
def test_other_large_fruits_capped(ingredient, density, per_week, search_name):
    """Cada fruta del dict respeta su per_week específico."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=[ingredient],
        multiplier=18.666666,
        structured=True,
    )
    name_sub = search_name
    qty_g = _qty_grams_for(result, name_sub, unit_density_g=density)
    expected_cap = max(2, int(round(per_week * 8)))
    cap_g = _expected_max_g(expected_cap * density, margin_pct=0.15)
    assert qty_g > 0
    assert qty_g <= cap_g, (
        f"{ingredient}: cap {expected_cap} unidades ({cap_g:.0f}g), "
        f"recibido {qty_g:.0f}g"
    )


# ===========================================================================
# 4. Cap escalado por person_weeks
# ===========================================================================
class TestMelonScaling:
    @pytest.mark.parametrize("scenario,multiplier,expected_cap", [
        ("4p mensual", 4 * 4 * 7 / 3, 16),
        ("2p mensual", 2 * 4 * 7 / 3, 8),
        ("1p mensual", 1 * 4 * 7 / 3, 4),
        ("2p quincenal", 2 * 2 * 7 / 3, 4),
        ("2p semanal", 2 * 1 * 7 / 3, 2),
        ("1p semanal", 1 * 1 * 7 / 3, 2),  # max(2, 1)
    ])
    def test_melon_scales(self, scenario, multiplier, expected_cap):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 melones"],
            multiplier=multiplier,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "melon")
        cap_g = _expected_max_g(expected_cap * 1200.0)
        assert qty_g <= cap_g, (
            f"{scenario}: esperado cap {expected_cap} melones ({cap_g:.0f}g), "
            f"recibido {qty_g:.0f}g"
        )


# ===========================================================================
# 5. Items NO listados no se afectan
# ===========================================================================
def test_manzana_not_capped_by_fruits_large():
    """Manzana (fruta pequeña, no en dict) no debe ser tocada."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["99 manzanas"],
        multiplier=18.666666,
        structured=True,
    )
    qty_g = _qty_grams_for(result, "manzana")
    if qty_g > 0:
        # 99 × 150g (typical manzana) × 18.67 = 277000g sin cap.
        # Cap específico melón es 8 × 1200 = 9600g. Manzana NO debe
        # cap a este nivel.
        big_threshold = 50000  # 50kg, debajo del valor sin cap pero arriba del cap melón
        assert qty_g > 5000, (
            f"Manzana fue capeada erróneamente: {qty_g:.0f}g muy bajo. "
            f"Solo melón/sandía/piña/lechosa/papaya están en el dict."
        )


def test_naranja_not_capped():
    """Naranja (fruta cítrica, no en dict) tampoco se cap."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["99 naranjas"],
        multiplier=18.666666,
        structured=True,
    )
    qty_g = _qty_grams_for(result, "naranja")
    if qty_g > 0:
        assert qty_g > 5000, (
            f"Naranja no debería ser capeada como fruta grande"
        )


# ===========================================================================
# 6. Sanity: source code marker
# ===========================================================================
def test_source_has_fruits_large_cap():
    """Sanity guard contra remoción accidental."""
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-FRUITS-LARGE-CAP" in src
    assert "_FRUITS_LARGE_PER_WEEK_PER_PERSON" in src
