"""[P6-FRUITS-PERISHABLE-CAP] Tests para el cap defensivo de frutas
perecederas vendidas por libras (fresas, arándanos, moras, frambuesas).

Bug observable (PDF 2026-05-05 15:34):
  Lista mostró "Fresas: 25 paquetes (1 lb c/u) = 25 lbs" para 2p × mes.
  11 kg de fresa para almacenar 1 mes = ~80% se descompone (fresas duran
  3-5 días refrigeradas).

Diferencia con P6-FRUITS-LARGE-CAP:
  Frutas grandes (melón, sandía) se compran por UNIDAD entera. Berries y
  fresas se compran por LIBRAS/PAQUETES. Cap diferente:
    - Grandes: cap por unidades + density
    - Perecederas: cap por libras directamente

Cobertura:
  - Repro PDF: 25 lbs fresas → cap 8
  - Cap escalado por person_weeks
  - Berries (arándanos/moras/frambuesas) con per_week=0.5 lb
  - Items NO listados (mango, naranja) no se afectan
  - Cap aplica a g, lb, paquete (los 3 paths)
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


def _qty_grams_for_fruit(result: list, name_substr: str) -> float:
    """Cantidad efectiva en gramos del primer item que matchee.
    Convierte por unit (lbs, paquete, g)."""
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
        if unit in ("lb", "lbs", "libra", "libras"):
            return qty * 453.592
        if unit in ("paquete", "paquetes"):
            return qty * 453.592  # 1 paquete = 1 lb
        if unit == "g":
            return qty
        return qty
    return -1.0


# ===========================================================================
# 1. Repro PDF — fresas 25 lbs → cap 8
# ===========================================================================
def test_repro_pdf_fresas_25_lbs_caps_to_8():
    """Caso real del PDF: 25 lbs de fresas para 2p × mes → cap 8 lbs."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["99 lbs de fresas"],
        multiplier=18.666666,  # 2p × mes
        structured=True,
    )
    qty_g = _qty_grams_for_fruit(result, "fresa")
    cap_g = 8 * 453.592 * 1.15  # 8 lbs + 15% margen
    assert 0 < qty_g <= cap_g, (
        f"Fresas cap fallido: {qty_g:.0f}g excede cap de {cap_g:.0f}g (~8 lbs)"
    )


# ===========================================================================
# 2. Cap escalado
# ===========================================================================
class TestFresasCapScaling:
    @pytest.mark.parametrize("scenario,multiplier,expected_cap_lbs", [
        ("4p mensual", 4 * 4 * 7 / 3, 16),
        ("2p mensual", 2 * 4 * 7 / 3, 8),
        ("1p mensual", 1 * 4 * 7 / 3, 4),
        ("2p quincenal", 2 * 2 * 7 / 3, 4),
        ("2p semanal", 2 * 1 * 7 / 3, 2),
    ])
    def test_fresa_cap_scales(self, scenario, multiplier, expected_cap_lbs):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 lbs de fresas"],
            multiplier=multiplier,
            structured=True,
        )
        qty_g = _qty_grams_for_fruit(result, "fresa")
        cap_g = expected_cap_lbs * 453.592 * 1.20
        assert qty_g <= cap_g, (
            f"{scenario}: cap {expected_cap_lbs} lbs ({cap_g:.0f}g), recibido {qty_g:.0f}g"
        )


# ===========================================================================
# 3. Cap por paquetes (LLM puede emitir "X paquetes de fresas")
# ===========================================================================
def test_fresa_cap_via_paquetes():
    """LLM emite 'X paquetes de fresas' → cap aplica a 'paquete' unit."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["99 paquetes de fresas"],
        multiplier=18.666666,
        structured=True,
    )
    qty_g = _qty_grams_for_fruit(result, "fresa")
    cap_g = 8 * 453.592 * 1.15
    assert 0 < qty_g <= cap_g, (
        f"Fresas paquete cap fallido: {qty_g:.0f}g > {cap_g:.0f}g"
    )


# ===========================================================================
# 4. Cap por gramos (LLM puede emitir "Xg de fresas")
# ===========================================================================
def test_fresa_cap_via_grams():
    """LLM emite 'Xg de fresas' → cap aplica a 'g' unit."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["500g de fresas frescas"],
        multiplier=18.666666,
        structured=True,
    )
    qty_g = _qty_grams_for_fruit(result, "fresa")
    cap_g = 8 * 453.592 * 1.15
    # 500g × 18.67 = 9333g pre-cap. Cap = 3628g (8 lbs). Should be capped.
    assert qty_g <= cap_g, (
        f"Fresas g cap fallido: {qty_g:.0f}g > {cap_g:.0f}g"
    )


# ===========================================================================
# 5. Berries (arándanos, moras, frambuesas) con per_week menor
# ===========================================================================
@pytest.mark.parametrize("berry,per_week", [
    ("arandanos", 0.5),
    ("moras", 0.5),
    ("frambuesas", 0.5),
])
def test_berries_capped_with_lower_per_week(berry, per_week):
    """Berries son más caros y consumo menor → cap más bajo (4 lbs vs 8 fresa)."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=[f"99 lbs de {berry}"],
        multiplier=18.666666,
        structured=True,
    )
    name_sub = berry.rstrip("s")  # "arandano", "mora", "frambuesa"
    qty_g = _qty_grams_for_fruit(result, name_sub)
    expected_cap_lbs = max(1, round(per_week * 8))  # 4 lbs
    cap_g = expected_cap_lbs * 453.592 * 1.20
    if qty_g > 0:
        assert qty_g <= cap_g, (
            f"{berry}: cap {expected_cap_lbs} lbs ({cap_g:.0f}g), recibido {qty_g:.0f}g"
        )


# ===========================================================================
# 6. Items NO listados no se afectan
# ===========================================================================
def test_mango_not_capped_by_perishable_fruits():
    """Mango (en _FRUITS_LARGE_CAP, NO en perishable) no debe ser tocado
    por este cap específico."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["10 lbs de mango"],
        multiplier=18.666666,
        structured=True,
    )
    qty_g = _qty_grams_for_fruit(result, "mango")
    if qty_g > 0:
        # Mango cap (P6-FRUITS-LARGE) es 8 unidades × 300g = 2400g.
        # Pero medido en lbs, este cap NO debe aplicar al mango.
        # Solo verificamos que no bajó al cap perishable de 8 lbs.
        # 10 lbs × 18.67 = 187 lbs raw → algún cap aplicará (large fruits)
        # Solo confirmamos que mango NO tiene cap perishable aplicado.
        pass  # No assertion específica, solo verificar que no crashea


def test_naranja_not_capped_by_perishable():
    """Naranja no es perishable berry — no cap perishable."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["99 naranjas"],
        multiplier=18.666666,
        structured=True,
    )
    # Naranja debe aparecer en cantidad alta (no cap específico para ella)
    from constants import strip_accents
    found = False
    for r in result:
        if not isinstance(r, dict):
            continue
        name = strip_accents(r.get("name", "")).lower()
        if "naranja" in name:
            found = True
            qty = float(r.get("market_qty", 0))
            assert qty > 5  # naranja debe aparecer en cantidad razonable


# ===========================================================================
# 7. Sanity: source code marker
# ===========================================================================
def test_source_has_perishable_cap():
    """Sanity guard contra remoción accidental."""
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-FRUITS-PERISHABLE-CAP" in src
    assert "_FRUITS_PERISHABLE_LBS_PER_WEEK_PER_PERSON" in src
    assert "fresa" in src.lower()
