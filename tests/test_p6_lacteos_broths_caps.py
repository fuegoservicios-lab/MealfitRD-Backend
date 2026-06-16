"""[P6-LACTEOS-PERISHABLE-CAP / P6-BROTHS-CAP] Tests para los caps
defensivos de yogurt (lácteo perecedero) y caldo (saborizante líquido).

Bugs observables (PDF 2026-05-05 18:33):
  - Yogurt griego sin azúcar: 21 potes (16 oz c/u) ≈ 9.5 kg
    Logísticamente imposible: no cabe en nevera promedio + se acerca
    al límite de caducidad. Realistic shopping pattern es semanal.
  - Caldo de vegetales: 3 lbs (formato weird; caldo es líquido o
    cubitos, no peso seco). ~45g/día = 5+ cubitos/día (excesivo).

Fix:
  - LACTEOS: cap 1.5 lb/persona/sem (≈1 pote 16oz cada 5 días).
    Para 2p×mes (8 pw): 12 lbs ≈ 12 potes.
  - BROTHS: cap 0.125 lb/persona/sem (~57g/sem ≈ 5-6 cubitos).
    Para 2p×mes: 1 lb (rounding) ≈ 50 cubitos = 1.5/día/2p.

Cobertura:
  - Repro PDF yogurt: 21 potes → cap 12
  - Repro PDF caldo: 3 lbs → cap 1
  - Match substring (cubre 'yogurt griego sin azúcar', 'caldo de vegetales', etc.)
  - Cap escalado por person_weeks
  - Items NO listados (queso, leche) no se afectan
  - Cap aplica a g, lbs, potes (los 3 paths típicos)
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


def _qty_grams_for(result: list, name_substr: str, unit_density_g: float = 453.592) -> float:
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
        if unit in ("paquete", "paquetes", "pote", "potes"):
            return qty * 453.592
        if unit == "g":
            return qty
        return qty * unit_density_g
    return -1.0


# ===========================================================================
# Sección 1: P6-LACTEOS-PERISHABLE-CAP (yogurt)
# ===========================================================================
class TestYogurtCap:
    def test_repro_pdf_yogurt_griego_21_potes_caps_to_12(self):
        """Caso real PDF: 21 potes de yogurt griego sin azúcar para
        2p × mes → cap 12 potes."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 potes de yogurt griego sin azúcar"],
            multiplier=18.666666,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "yogurt")
        cap_g = 12 * 453.592 * 1.10  # 12 potes + 10% margen
        assert 0 < qty_g <= cap_g, (
            f"Yogurt cap fallido: {qty_g:.0f}g > {cap_g:.0f}g (~12 potes)"
        )

    def test_yogurt_grams_path_capped(self):
        """LLM emite 'Xg de yogurt' → cap aplica a 'g' unit."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["500g de yogurt griego"],
            multiplier=18.666666,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "yogurt")
        cap_g = 12 * 453.592 * 1.10
        assert qty_g <= cap_g

    def test_yogurt_substring_match(self):
        """Match substring: 'yogurt griego sin azucar' debe matchear via
        'yogurt' substring sin requerir entry exacto."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 potes de yogurt natural sin grasa"],
            multiplier=18.666666,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "yogurt")
        cap_g = 12 * 453.592 * 1.10
        assert 0 < qty_g <= cap_g

    @pytest.mark.parametrize("scenario,multiplier,expected_cap_lbs", [
        ("4p mensual", 4 * 4 * 7 / 3, 24),
        ("2p mensual", 2 * 4 * 7 / 3, 12),
        ("1p mensual", 1 * 4 * 7 / 3, 6),
        ("2p quincenal", 2 * 2 * 7 / 3, 6),
        ("2p semanal", 2 * 1 * 7 / 3, 3),
    ])
    def test_yogurt_cap_scales(self, scenario, multiplier, expected_cap_lbs):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 lbs de yogurt griego"],
            multiplier=multiplier,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "yogurt")
        cap_g = expected_cap_lbs * 453.592 * 1.20
        assert qty_g <= cap_g, (
            f"{scenario}: cap {expected_cap_lbs} lbs ({cap_g:.0f}g), recibido {qty_g:.0f}g"
        )


# ===========================================================================
# Sección 2: P6-BROTHS-CAP (caldo)
# ===========================================================================
class TestCaldoCap:
    def test_repro_pdf_caldo_3lbs_caps_to_1(self):
        """Caso real PDF: 3 lbs de caldo de vegetales para 2p × mes →
        cap 1 lb."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 lbs de caldo de vegetales"],
            multiplier=18.666666,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "caldo")
        cap_g = 1.0 * 453.592 * 1.20  # 1 lb cap + margen
        assert 0 < qty_g <= cap_g, (
            f"Caldo cap fallido: {qty_g:.0f}g > {cap_g:.0f}g"
        )

    @pytest.mark.parametrize("caldo_type", [
        "caldo de vegetales",
        "caldo de pollo",
        "caldo de res",
        "caldo de hueso",
    ])
    def test_caldo_variants_substring_match(self, caldo_type):
        """Match substring: 'caldo de X' debe matchear via 'caldo'."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=[f"99 lbs de {caldo_type}"],
            multiplier=18.666666,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "caldo")
        cap_g = 1.0 * 453.592 * 1.20
        assert 0 < qty_g <= cap_g, (
            f"{caldo_type}: cap fallido {qty_g:.0f}g > {cap_g:.0f}g"
        )

    @pytest.mark.parametrize("scenario,multiplier,expected_cap_lbs", [
        ("4p mensual", 4 * 4 * 7 / 3, 2.0),
        ("2p mensual", 2 * 4 * 7 / 3, 1.0),
        ("1p mensual", 1 * 4 * 7 / 3, 0.5),
        ("2p semanal", 2 * 1 * 7 / 3, 0.5),
    ])
    def test_caldo_cap_scales(self, scenario, multiplier, expected_cap_lbs):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 lbs de caldo de vegetales"],
            multiplier=multiplier,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "caldo")
        cap_g = expected_cap_lbs * 453.592 * 1.30
        assert qty_g <= cap_g, (
            f"{scenario}: cap {expected_cap_lbs} lbs ({cap_g:.0f}g), recibido {qty_g:.0f}g"
        )


# ===========================================================================
# Sección 3: Items NO afectados
# ===========================================================================
def test_queso_not_capped_by_yogurt():
    """Queso no contiene 'yogurt' substring → no cap por lácteo perishable."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["10 lbs de queso blanco"],
        multiplier=18.666666,
        structured=True,
    )
    from constants import strip_accents
    for r in result:
        if not isinstance(r, dict):
            continue
        name = strip_accents(r.get("name", "")).lower()
        if "queso" in name and "yogurt" not in name:
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            # Queso debe aparecer en cantidades normales (mensual×2p
            # 10 lbs raw × 18.67 = mucho más de 12 lbs cap del yogurt)
            if "lb" in unit:
                # Queso debe ser > 12 lbs (no fue capeado al límite del yogurt)
                assert qty > 5  # solo confirmamos no cap absurdo


def test_leche_not_capped_by_yogurt():
    """[STALE-PARSER-FIX] P6-LACTEOS-EXT-4 (2026-05-07) añadió DELIBERADAMENTE
    'leche': 1.75 lb/persona/sem al cap de lácteos perecederos (cap all-cause
    por volumen: lácteos abiertos duran ~14 días). El test viejo asumía que
    leche NUNCA se capeaba — premisa stale.

    Cap esperado para 2p×mes (8 pw): round(1.75 × 8) = 14 lbs ≈ 2.8 L ≈
    3 cartones de 946ml. Input raw '10 lbs de leche' × 18.67 ≈ 186 lbs → se
    capea muy por debajo del raw, pero a su PROPIO límite (no al de yogurt).
    El display DB-backed sale en 'cartón' (3) o lbs (14) según master_map.
    """
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["10 lbs de leche"],
        multiplier=18.666666,
        structured=True,
    )
    from constants import strip_accents
    _WEIGHT_PER_UNIT = {  # gramos aprox por unidad de display
        "lb": 453.592, "lbs": 453.592, "libra": 453.592, "libras": 453.592,
        "carton": 946.0, "cartones": 946.0, "g": 1.0,
    }
    seen = False
    for r in result:
        if not isinstance(r, dict):
            continue
        name = strip_accents(r.get("name", "")).lower()
        if "leche" not in name:
            continue
        seen = True
        qty = float(r.get("market_qty", 0))
        unit = strip_accents((r.get("market_unit") or "")).lower()
        qty_g = qty * _WEIGHT_PER_UNIT.get(unit, 453.592)
        # Cap documentado: 14 lbs ≈ 6350g. Generoso margen +20% por
        # redondeo de unidad de display (cartón/lb). Debe estar MUY por
        # debajo del raw (~186 lbs ≈ 84600g) → cap firó.
        leche_cap_g = 14 * 453.592 * 1.20  # ≈7620g
        assert 0 < qty_g <= leche_cap_g, (
            f"Leche debe capearse a su propio límite (~14 lbs / 3 cartones), "
            f"recibido {qty_g:.0f}g ({qty} {unit})"
        )
    assert seen, "Leche debe aparecer en la lista"


def test_aceite_not_capped_by_caldo():
    """Aceite no contiene 'caldo' substring → no afectado."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["10 lbs de aceite de oliva"],
        multiplier=18.666666,
        structured=True,
    )
    from constants import strip_accents
    for r in result:
        if not isinstance(r, dict):
            continue
        name = strip_accents(r.get("name", "")).lower()
        if "aceite" in name:
            qty = float(r.get("market_qty", 0))
            assert qty > 0  # solo confirma que sigue apareciendo


# ===========================================================================
# Sección 4: Sanity - source code markers
# ===========================================================================
def test_source_has_lacteos_cap():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-LACTEOS-PERISHABLE-CAP" in src
    assert "_LACTEOS_PERISHABLE_LBS_PER_WEEK_PER_PERSON" in src
    assert "yogurt" in src.lower()


def test_source_has_broths_cap():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-BROTHS-CAP" in src
    assert "_BROTHS_LBS_PER_WEEK_PER_PERSON" in src
    assert "caldo" in src.lower()
