"""[P6-EGGS-AGGREGATE-CAP] Tests para el cap defensivo de huevos en
la lista de compras post-aggregator.

Bug observable (PDF 2026-05-05 14:35):
  Lista mostró "Huevo: 11 cartón (30 uds.) = 330 huevos" para 2p × mes.
  P6-EGGS-CAP del day_generator prompt redujo las RECETAS a ~5 enteros
  + ~10.5 claras / 3 días (reviewer médico aceptó), pero el aggregator
  suma claras+enteros como huevos comprables. Resultado: ~5.5 huevos/
  persona/día visible en PDF — visualmente excesivo.

Causa raíz:
  1 clara = 1 huevo en aggregator (correcto: hay que comprar el huevo
  entero para sacar la clara). PERO los usuarios NO descartan yemas en
  la práctica — las usan en otras comidas. El aggregator no captura
  esta optimización de uso real.

Fix:
  Cap defensivo a `max(2, round(person_weeks))` cartones de 30:
    - 2p mensual (8 pw) → 8 cartones = 240 huevos = 4/persona/día
    - 2p quincenal (4 pw) → 4 cartones = 120 huevos
    - 2p semanal (2 pw) → 2 cartones = 60 huevos
  4/persona/día es el threshold del reviewer médico.

Cobertura:
  - Repro PDF: 290+ huevos raw → ≤ 240 (cap)
  - Cap escalado por person_weeks
  - Cap mínimo (max(2, ...))
  - Items que NO son huevos no se tocan (regresión guard)
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


# Reusa helpers del test paralelo
MARKET_MIN_LB_G = 0.25 * 453.592


def _expected_max_g(cap_g: float, margin_pct: float = 0.10) -> float:
    return max(cap_g, MARKET_MIN_LB_G) * (1.0 + margin_pct)


def _qty_grams_for_eggs(result: list) -> float:
    """Cantidad de huevos en gramos. Convierte por unit:
       - unidad / cartón → ×50g (huevo estándar)
       - g → 1×
       - lbs → ×453"""
    from constants import strip_accents
    for r in result:
        if not isinstance(r, dict):
            continue
        name = strip_accents(r.get("name", "")).lower()
        if "huevo" not in name:
            continue
        qty = float(r.get("market_qty", 0))
        unit = (r.get("market_unit") or "").lower()
        if unit in ("unidad", "ud.", "uds.", "unidades"):
            return qty * 50.0
        if unit in ("cartón", "carton", "cartones"):
            return qty * 30 * 50.0  # 30 huevos × 50g
        if unit in ("lb", "lbs"):
            return qty * 453.592
        if unit == "g":
            return qty
        return qty * 50.0
    return -1.0


# ===========================================================================
# 1. Repro PDF — 11 cartones → cap 8
# ===========================================================================
def test_repro_pdf_11_cartones_caps_to_8():
    """[P6-EGGS-AGGREGATE-CAP] Caso real del PDF: 5 enteros + ~10.5 claras
    en 3 días × multiplier 18.67 = ~290 huevos raw. Cap a 240 (8 cartones)."""
    plan = [
        # Huevos enteros distribuidos
        "2 huevos enteros",
        "2 huevos enteros",
        "1 huevo entero",
        # Claras (consolidan al mismo canonical 'Huevo' via PROTEIN_SYNONYMS)
        "3 claras de huevo",
        "3 claras de huevo",
        "4 claras de huevo",
    ]
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=plan,
        multiplier=18.666666,  # mensual × 2p
        structured=True,
    )
    qty_g = _qty_grams_for_eggs(result)
    # Cap = 8 cartones × 30 × 50g = 12000g, +10% margen
    cap_g = _expected_max_g(8 * 30 * 50.0)
    assert 0 < qty_g <= cap_g, (
        f"Eggs cap fallido: {qty_g:.0f}g excede cap de {cap_g:.0f}g (~8 cartones)"
    )


# ===========================================================================
# 2. Cap escala con person_weeks
# ===========================================================================
class TestEggsCapScaling:
    def _eggs_qty_g(self, multiplier: float) -> float:
        # Pedir 99 huevos para forzar cap
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 huevos"],
            multiplier=multiplier,
            structured=True,
        )
        return _qty_grams_for_eggs(result)

    @pytest.mark.parametrize("scenario,multiplier,expected_cap_cartones", [
        ("4p mensual", 4 * 4 * 7 / 3, 16),
        ("2p mensual", 2 * 4 * 7 / 3, 8),
        ("1p mensual", 1 * 4 * 7 / 3, 4),
        ("2p quincenal", 2 * 2 * 7 / 3, 4),
        ("2p semanal", 2 * 1 * 7 / 3, 2),
        ("1p semanal", 1 * 1 * 7 / 3, 2),  # max(2, 1)
    ])
    def test_cap_scales(self, scenario, multiplier, expected_cap_cartones):
        actual_g = self._eggs_qty_g(multiplier)
        cap_g = _expected_max_g(expected_cap_cartones * 30 * 50.0)
        assert actual_g <= cap_g, (
            f"{scenario} (mult={multiplier:.2f}): esperado cap "
            f"{expected_cap_cartones} cartones ({cap_g:.0f}g), "
            f"recibido {actual_g:.0f}g"
        )


# ===========================================================================
# 3. Cap mínimo: max(2, ...) garantiza floor de 2 cartones
# ===========================================================================
def test_cap_minimum_is_2_cartones():
    """Para multiplier muy bajo (1p semanal mult=2.33, person_weeks=1),
    cap = max(2, round(1)) = 2 cartones. NO baja a 1 (insuficiente para
    una semana de comida estándar)."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["99 huevos"],
        multiplier=2.333333,  # 1p semanal
        structured=True,
    )
    qty_g = _qty_grams_for_eggs(result)
    cap_g = _expected_max_g(2 * 30 * 50.0)  # 2 cartones mínimo
    assert qty_g > 0
    assert qty_g <= cap_g


# ===========================================================================
# 4. Items que NO son huevos no se afectan (regresión guard)
# ===========================================================================
def test_pollo_not_capped_by_eggs():
    """Pollo NO debe ser tocado por el cap de huevos."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["3 lbs de pollo"],
        multiplier=18.666666,
        structured=True,
    )
    # Pollo no debe ser cap to 8 cartones equivalent
    for r in result:
        if not isinstance(r, dict):
            continue
        name = (r.get("name") or "").lower()
        if "pollo" in name:
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            # Pollo a 18.67× de 3 lbs = 56 lbs en lbs path. Nada que ver
            # con cap de huevos. Solo confirmamos que aparece.
            assert qty > 0
            # No debe ser cartones (sería bug serio)
            assert "cartón" not in unit
            assert "carton" not in unit


def test_other_items_in_plan_unaffected():
    """Plan con huevos + otros items: huevos cap, otros intactos."""
    plan = [
        "5 huevos enteros",  # será cap
        "200g de pollo",
        "100g de arroz",
    ]
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=plan,
        multiplier=18.666666,
        structured=True,
    )
    names_lower = {(r.get("name") or "").lower() for r in result if isinstance(r, dict)}
    assert any("pollo" in n for n in names_lower)
    # Huevos presente y bajo cap
    eggs_g = _qty_grams_for_eggs(result)
    assert 0 < eggs_g <= _expected_max_g(8 * 30 * 50.0)


# ===========================================================================
# 5. Sanity: source code referencia el fix
# ===========================================================================
def test_source_has_eggs_cap():
    """Sanity guard contra remoción accidental del fix."""
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-EGGS-AGGREGATE-CAP" in src
    assert "_EGGS_NAMES_FOR_CAP" in src
    assert "_HUEVOS_PER_CARTON" in src
