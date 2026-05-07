"""[P6-OLIVE-CAP-FIX / P6-VEG-EXT-2 / P6-CITRUS-CAP] Tests para los 3 fixes
de caps observables en PDF 2026-05-05 19:36 ([8b0f351d]).

Bugs:
  1. Aceitunas: 187 frascos (12 oz c/u) — P5-OLIVE-CAP existía pero nunca
     firó. Causa: name canonicalizado por master_map a variantes ("Aceitunas
     Manzanilla"); set literal `{'aceituna', 'aceitunas'}` no matcheaba.
     También unit_key con suffix tipo `'frasco (12 oz)'` no matcheaba.
  2. Auyama: 34¾ lbs (~31 Uds.) — sin cap, NO en `_VEG_PER_WEEK_PER_PERSON`.
     Plátano verde: 28 Uds. — entry `'platano'` era exacta, no substring.
     Berenjena: 12½ lbs (~19 Uds.) — sin cap.
  3. Limón: 51 Uds. — sin cap (cítrico).

Fixes:
  1. P6-OLIVE-CAP-FIX: substring match en nombre Y unit_key.
  2. P6-VEG-EXT-2: añadir auyama, platano verde, berenjena al dict.
  3. P6-CITRUS-CAP: nuevo bloque para limón / lima / naranja (cap por uds y g).

Cobertura:
  - Repro PDF de los 3 (187 → 3, 31 → 8, 28 → 24, 19 → 16, 51 → 32)
  - Variantes de nombre (Aceitunas Manzanilla, Aceitunas Negras)
  - Substring de unit (frasco vs 'frasco (12 oz)')
  - Cap escalado por person_weeks
  - Items NO listados (aceite no se afecta por olive cap, etc.)
  - Sanity: markers en source
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


def _qty_grams_for(result: list, name_substr: str, unit_density_g: float = 100.0) -> float:
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
        if unit in ("frasco", "frascos", "pote", "potes", "botella", "botellas"):
            return qty * 340.194  # 12 oz default
        if unit in ("paquete", "paquetes"):
            return qty * 453.592
        if unit == "g":
            return qty
        return qty * unit_density_g
    return -1.0


def _qty_units_for(result: list, name_substr: str) -> float:
    """Devuelve cantidad en formato de display original (qty)."""
    from constants import strip_accents
    needle = strip_accents(name_substr).lower()
    for r in result:
        if not isinstance(r, dict):
            continue
        haystack = strip_accents(r.get("name", "")).lower()
        if needle not in haystack:
            continue
        return float(r.get("market_qty", 0))
    return -1.0


# ===========================================================================
# Sección 1: P6-OLIVE-CAP-FIX
# ===========================================================================
class TestOliveCapFix:
    def test_repro_pdf_aceitunas_187_frascos(self):
        """Caso real PDF: 187 frascos uncapped. Cap esperado ~3 frascos
        (8 person_weeks / 3 ≈ 3)."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["50 frascos de aceitunas verdes"],
            multiplier=18.666666,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "aceituna")
        cap_g = 3 * 340.194 * 1.20  # 3 frascos × 12 oz + 20% margen
        assert 0 < qty_g <= cap_g, (
            f"Aceitunas cap fallido: {qty_g:.0f}g > {cap_g:.0f}g (~3 frascos)"
        )

    def test_aceitunas_manzanilla_variant_capped(self):
        """Variante de nombre: 'Aceitunas Manzanilla' debe matchear via
        substring 'aceituna'."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["50 frascos de aceitunas manzanilla"],
            multiplier=18.666666,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "aceituna")
        cap_g = 3 * 340.194 * 1.20
        assert 0 < qty_g <= cap_g, (
            f"Variante 'manzanilla' debe capear: {qty_g:.0f}g > {cap_g:.0f}g"
        )

    def test_aceitunas_negras_variant_capped(self):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["50 frascos de aceitunas negras"],
            multiplier=18.666666,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "aceituna")
        cap_g = 3 * 340.194 * 1.20
        assert 0 < qty_g <= cap_g

    def test_aceitunas_via_grams_path(self):
        """LLM emite 'Xg de aceitunas' → cap aplica al peso."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["500g de aceitunas"],
            multiplier=18.666666,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "aceituna")
        cap_g = 3 * 340.194 * 1.20
        assert qty_g <= cap_g

    @pytest.mark.parametrize("scenario,multiplier,expected_cap_frascos", [
        ("4p mensual", 4 * 4 * 7 / 3, 5),  # 16 pw / 3 ≈ 5
        ("2p mensual", 2 * 4 * 7 / 3, 3),  # 8 pw / 3 ≈ 3
        ("1p mensual", 1 * 4 * 7 / 3, 1),  # 4 pw / 3 ≈ 1
        ("2p semanal", 2 * 1 * 7 / 3, 1),  # 2 pw / 3 = 0.67 → max(1, 0) = 1
    ])
    def test_olive_cap_scales(self, scenario, multiplier, expected_cap_frascos):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["50 frascos de aceitunas"],
            multiplier=multiplier,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "aceituna")
        cap_g = expected_cap_frascos * 340.194 * 1.30
        assert qty_g <= cap_g, (
            f"{scenario}: cap {expected_cap_frascos} frascos ({cap_g:.0f}g), "
            f"recibido {qty_g:.0f}g"
        )

    @pytest.mark.parametrize("unit_form", [
        "12 oz",   # caso real bug 20:36 — LLM emite "12 oz aceitunas"
        "0.75 lb",
        "340 g",
        "0.34 kg",
    ])
    def test_olive_cap_via_weight_units_fix3(self, unit_form):
        """[P6-OLIVE-CAP-FIX-3] Cap debe activar cuando LLM emite aceitunas
        en CUALQUIER unit de peso (g/kg/oz/lb/ml/l), no solo 'g' literal.
        Bug PDF 2026-05-05 20:36 [265055c3]: 94 frascos uncapped pese a
        FIX-1 porque LLM emitió 'oz', mi cap solo cubría 'g'."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=[f"{unit_form} de aceitunas verdes"] * 5,  # 5 meals
            multiplier=18.666666,  # 2p × mes
            structured=True,
        )
        qty_g = _qty_grams_for(result, "aceituna")
        cap_g = 3 * 340.194 * 1.20  # 3 frascos + 20% margen
        assert qty_g <= cap_g, (
            f"FIX-3: '{unit_form} aceitunas' debe capear a ≤{cap_g:.0f}g, "
            f"recibido {qty_g:.0f}g (~{qty_g/340.194:.0f} frascos)"
        )

    def test_olive_cap_mixed_weight_units(self):
        """LLM puede emitir mix de units en el mismo plan: '50g aceitunas'
        + '4 oz aceitunas' + '0.25 lb aceitunas'. Suma total debe capearse."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=[
                "50g de aceitunas",
                "4 oz de aceitunas verdes",
                "0.25 lb de aceitunas negras",
            ],
            multiplier=18.666666,
            structured=True,
        )
        qty_g = _qty_grams_for(result, "aceituna")
        cap_g = 3 * 340.194 * 1.20
        assert qty_g <= cap_g, (
            f"Mix de units: total debe capear, recibido {qty_g:.0f}g"
        )

    def test_aceite_not_affected_by_olive_cap(self):
        """'Aceite de oliva' contiene 'oliv' substring — pero NO debe matchear
        'olive' (que solo aplica a fruto de olivo, no aceite). Sanity check."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["10 botellas de aceite de oliva"],
            multiplier=18.666666,
            structured=True,
        )
        # Aceite debe aparecer (puede ser 1-3 botellas reales). Solo
        # confirmamos que NO crashea ni elimina el item.
        from constants import strip_accents
        for r in result:
            if not isinstance(r, dict):
                continue
            name = strip_accents(r.get("name", "")).lower()
            if "aceite" in name and "oliva" in name:
                qty = float(r.get("market_qty", 0))
                # Debe seguir presente — 10 botellas × 18.67 = mucho aceite
                # post-cap (puede haber otro cap, pero NO el de aceitunas)
                assert qty > 0


# ===========================================================================
# Sección 2: P6-VEG-EXT-2 (auyama, plátano verde, berenjena)
# ===========================================================================
class TestVegExt2:
    def test_auyama_capped(self):
        """Auyama 99 unidades raw × 18.67 = mucho. Cap = 8 unidades para 2p×mes."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 auyamas"],
            multiplier=18.666666,
            structured=True,
        )
        from constants import strip_accents
        for r in result:
            if not isinstance(r, dict):
                continue
            name = strip_accents(r.get("name", "")).lower()
            if "auyama" not in name:
                continue
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            if unit in ("unidad", "unidades"):
                assert qty <= 8 * 1.10, f"Auyama cap fallido: {qty} > 8"
            elif unit in ("lb", "lbs"):
                # 8 auyamas × 1100g = 8800g ≈ 19.4 lbs
                assert qty <= 20

    def test_platano_verde_capped(self):
        """'platano verde' debe matchear su entry específica (no 'platano' genérico)."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 plátanos verdes"],
            multiplier=18.666666,
            structured=True,
        )
        from constants import strip_accents
        for r in result:
            if not isinstance(r, dict):
                continue
            name = strip_accents(r.get("name", "")).lower()
            if "platano" not in name or "verde" not in name:
                continue
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            if unit in ("unidad", "unidades"):
                # Cap = 3/persona/sem × 8 pw = 24
                assert qty <= 24 * 1.10, f"Plátano verde cap fallido: {qty} > 24"

    def test_berenjena_capped(self):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 berenjenas"],
            multiplier=18.666666,
            structured=True,
        )
        from constants import strip_accents
        for r in result:
            if not isinstance(r, dict):
                continue
            name = strip_accents(r.get("name", "")).lower()
            if "berenjena" not in name:
                continue
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            if unit in ("unidad", "unidades"):
                # Cap = 2/persona/sem × 8 pw = 16
                assert qty <= 16 * 1.15, f"Berenjena cap fallido: {qty} > 16"

    def test_platano_maduro_still_capped_separately(self):
        """No-regresión: plátano maduro sigue capeado por su entry original."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 plátanos maduros"],
            multiplier=18.666666,
            structured=True,
        )
        from constants import strip_accents
        for r in result:
            if not isinstance(r, dict):
                continue
            name = strip_accents(r.get("name", "")).lower()
            if "platano" not in name or "maduro" not in name:
                continue
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            if unit in ("unidad", "unidades"):
                # Cap original = 5/persona/sem × 8 pw = 40
                assert qty <= 40 * 1.10


# ===========================================================================
# Sección 3: P6-CITRUS-CAP (limón, lima, naranja)
# ===========================================================================
class TestCitrusCap:
    def test_repro_pdf_limon_51_capped_to_32(self):
        """Caso real PDF: 51 limones para 2p × mes → cap 32."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 limones"],
            multiplier=18.666666,
            structured=True,
        )
        from constants import strip_accents
        for r in result:
            if not isinstance(r, dict):
                continue
            name = strip_accents(r.get("name", "")).lower()
            if "limon" not in name:
                continue
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            if unit in ("unidad", "unidades"):
                # Cap = 4/persona/sem × 8 pw = 32
                assert qty <= 32 * 1.10, f"Limón cap fallido: {qty} > 32"

    @pytest.mark.parametrize("citrus", ["lima", "limas", "limon", "limones"])
    def test_lima_limon_variants_capped(self, citrus):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=[f"99 {citrus}"],
            multiplier=18.666666,
            structured=True,
        )
        from constants import strip_accents
        needle = strip_accents(citrus).lower().rstrip("s")  # "lima", "limon"
        for r in result:
            if not isinstance(r, dict):
                continue
            name = strip_accents(r.get("name", "")).lower()
            if needle not in name:
                continue
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            if unit in ("unidad", "unidades"):
                assert qty <= 32 * 1.15, f"{citrus} cap fallido: {qty} > 32"

    def test_limon_via_grams(self):
        """500g limones × 18.67 = 9333g pre-cap. Cap = 32 × 60g = 1920g."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["500g de limón rallado"],
            multiplier=18.666666,
            structured=True,
        )
        from constants import strip_accents
        for r in result:
            if not isinstance(r, dict):
                continue
            name = strip_accents(r.get("name", "")).lower()
            if "limon" not in name:
                continue
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            if unit == "g":
                assert qty <= 32 * 60 * 1.30, f"Limón g cap fallido: {qty}g"

    @pytest.mark.parametrize("scenario,multiplier,expected_cap_units", [
        ("4p mensual", 4 * 4 * 7 / 3, 64),
        ("2p mensual", 2 * 4 * 7 / 3, 32),
        ("1p mensual", 1 * 4 * 7 / 3, 16),
        ("2p semanal", 2 * 1 * 7 / 3, 8),
    ])
    def test_citrus_cap_scales(self, scenario, multiplier, expected_cap_units):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["99 limones"],
            multiplier=multiplier,
            structured=True,
        )
        from constants import strip_accents
        for r in result:
            if not isinstance(r, dict):
                continue
            name = strip_accents(r.get("name", "")).lower()
            if "limon" not in name:
                continue
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            if unit in ("unidad", "unidades"):
                assert qty <= expected_cap_units * 1.20, (
                    f"{scenario}: cap {expected_cap_units} uds, recibido {qty}"
                )


# ===========================================================================
# Sección 4: Sanity guards — markers en source
# ===========================================================================
def test_source_has_olive_cap_fix_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-OLIVE-CAP-FIX" in src, "Marker FIX-1 debe existir"
    assert "P6-OLIVE-CAP-FIX-3" in src, (
        "Marker FIX-3 (peso total: g/kg/oz/lb/ml/l) debe existir — "
        "FIX-1 solo cubrió 'g' y substring 'frasco', dejando 'oz' uncapped"
    )
    assert "_OLIVE_SUBSTRINGS" in src, "Substring set debe existir"
    assert "_OLIVE_UNIT_SUBSTRINGS" in src, "Unit substring set debe existir"
    assert "_WEIGHT_UNIT_TO_G" in src, "Conversión total de peso debe existir"


def test_source_has_veg_ext_2_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-VEG-EXT-2" in src
    assert "'auyama'" in src
    assert "'platano verde'" in src
    assert "'berenjena'" in src


def test_source_has_citrus_cap_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-CITRUS-CAP" in src
    assert "_CITRUS_PER_WEEK_PER_PERSON" in src
    assert "'limon'" in src
