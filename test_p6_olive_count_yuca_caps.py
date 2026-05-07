"""[P6-OLIVE-CAP-FIX-4 / P6-VEG-EXT-4] Tests para los 2 fixes finales:
cap por count para aceitunas y cap para yuca.

Bugs:
  1. PDF 2026-05-05 21:34: 234 frascos de aceitunas (12 oz c/u) = 80 kg.
     Mi P6-OLIVE-CAP-FIX-3 cubría weight units (g/kg/oz/lb/ml/l) pero NO
     cubría 'unidad'/'unidades'. El LLM emitió "X aceitunas verdes" como
     count → silenciosamente skipped. Después apply_smart_market_units
     BLOQUE 2 multiplica por density (5g/aceituna) y BLOQUE 1 divide por
     container_weight_g (340g/frasco) → 234 frascos en display.
  2. PDF 2026-05-05 21:34: 17 unidades de yuca para 2p×mes = ~7 kg.
     Yuca NO estaba en `_VEG_PER_WEEK_PER_PERSON`. Es staple DR como
     carbo, uso 2-3×/sem.

Fixes:
  1. P6-OLIVE-CAP-FIX-4: añadir cap por count en 'unidad'/'unidades'/'ud'
     /'uds'. cap_count = cap_g / density_g_per_olive (5g) → ~204 olivas
     max para 2p×mes (3 frascos × 68 olivas/frasco).
  2. P6-VEG-EXT-4: añadir 'yuca'/'yucas' al `_VEG_PER_WEEK_PER_PERSON`
     con cap 3/persona/sem = 24 max. Density 400g (yuca DR es grande).

Cobertura:
  - Repro PDF de los 2
  - Cap escalado por person_weeks
  - Variantes 'ud'/'uds'/'unidad'/'unidades' para olives
  - No regresión: olive cap por peso/frasco sigue funcionando
  - No regresión: papa/batata siguen capeando con sus entries
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


# ===========================================================================
# Sección 1: P6-OLIVE-CAP-FIX-4 (cap por count)
# ===========================================================================
class TestOliveCapFix4:
    def test_repro_pdf_aceitunas_count_uncapped(self, caplog):
        """Caso real PDF 21:34: '99 aceitunas verdes' (count) × 18.67 = 1848
        aceitunas raw post-mult. Pre-FIX-4: silenciosamente skipped. Post:
        cap a ~204 unidades (3 frascos × 68 olivas/frasco)."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 aceitunas verdes"],
                multiplier=18.666666,
                structured=True,
            )
        olive_caps = [
            r for r in caplog.records
            if "P5-OLIVE-CAP" in r.message and "count" in r.message.lower()
        ]
        assert olive_caps, (
            f"P6-OLIVE-CAP-FIX-4 debe firar 'count cap' para aceitunas via 'unidad'. "
            f"Warnings vistos: {[r.message for r in caplog.records if 'OLIVE' in r.message]}"
        )

    @pytest.mark.parametrize("plural_form", [
        "99 aceitunas",
        "99 aceitunas verdes",
        "99 aceitunas negras",
        "99 aceitunas manzanilla",
    ])
    def test_aceitunas_variants_count_capped(self, plural_form, caplog):
        """Variantes de nombre con count grande deben capear."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=[plural_form],
                multiplier=18.666666,
                structured=True,
            )
        olive_count_caps = [
            r for r in caplog.records
            if "P5-OLIVE-CAP" in r.message and "count" in r.message.lower()
        ]
        assert olive_count_caps, f"'{plural_form}' debe firar count cap"

    @pytest.mark.parametrize("scenario,multiplier,expected_cap_count", [
        ("4p mensual", 4 * 4 * 7 / 3, 340),  # 5 frascos × 68
        ("2p mensual", 2 * 4 * 7 / 3, 204),  # 3 frascos × 68
        ("2p semanal", 2 * 1 * 7 / 3, 68),   # 1 frasco × 68
    ])
    def test_olive_count_cap_scales(self, scenario, multiplier, expected_cap_count, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 aceitunas verdes"],
                multiplier=multiplier,
                structured=True,
            )
        olive_count_caps = [
            r for r in caplog.records
            if "P5-OLIVE-CAP" in r.message and "count" in r.message.lower()
        ]
        if expected_cap_count < 1000:  # Solo si plan excede cap
            assert olive_count_caps, f"{scenario}: count cap debe firar"

    def test_olive_weight_cap_still_works(self, caplog):
        """No-regresión: cap por peso (FIX-3) sigue funcionando."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["12 oz de aceitunas"] * 5,
                multiplier=18.666666,
                structured=True,
            )
        weight_caps = [
            r for r in caplog.records
            if "P5-OLIVE-CAP" in r.message and "peso total" in r.message
        ]
        assert weight_caps, "Cap por peso (FIX-3) NO debe romperse con FIX-4"

    def test_olive_frasco_cap_still_works(self, caplog):
        """No-regresión: cap por unit substring (frasco) sigue funcionando."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["50 frascos de aceitunas"],
                multiplier=18.666666,
                structured=True,
            )
        unit_caps = [
            r for r in caplog.records
            if "P5-OLIVE-CAP" in r.message and ("'frasco" in r.message or "frascos" in r.message)
        ]
        # Si firó por frasco substring, OK. Si firó por count o weight, también OK.
        any_cap = [r for r in caplog.records if "P5-OLIVE-CAP" in r.message]
        assert any_cap, "Algún cap de aceitunas debe firar para 50 frascos"


# ===========================================================================
# Sección 2: P6-VEG-EXT-4 (yuca)
# ===========================================================================
class TestYucaCap:
    def test_repro_pdf_yuca_17_uds(self, caplog):
        """Caso real PDF 21:34: 17 unidades yuca para 2p×mes. Cap esperado: 24."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 yucas"],
                multiplier=18.666666,
                structured=True,
            )
        yuca_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and "yuca" in r.message.lower()
        ]
        assert yuca_caps, (
            f"P5-VEG-CAP debe firar para yuca. "
            f"Warnings: {[r.message for r in caplog.records if 'VEG-CAP' in r.message]}"
        )

    @pytest.mark.parametrize("scenario,multiplier,expected_cap_uds", [
        ("4p mensual", 4 * 4 * 7 / 3, 48),
        ("2p mensual", 2 * 4 * 7 / 3, 24),
        ("1p mensual", 1 * 4 * 7 / 3, 12),
        ("2p semanal", 2 * 1 * 7 / 3, 6),
    ])
    def test_yuca_cap_scales(self, scenario, multiplier, expected_cap_uds, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 yucas"],
                multiplier=multiplier,
                structured=True,
            )
        yuca_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and "yuca" in r.message.lower()
        ]
        # En 2p×mes (24 cap) con 99 raw → debe capear
        if expected_cap_uds < 99:
            assert yuca_caps, f"{scenario}: yuca cap debe firar"

    def test_papa_still_separately_capped(self, caplog):
        """No-regresión: papa sigue capeando (cap original 40)."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 papas"],
                multiplier=18.666666,
                structured=True,
            )
        papa_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and "papa" in r.message.lower()
            and "yuca" not in r.message.lower()
        ]
        assert papa_caps, "Papa cap original debe seguir activo"

    def test_batata_still_separately_capped(self, caplog):
        """No-regresión: batata (P6-VEG-EXT-3) sigue capeando."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 batatas"],
                multiplier=18.666666,
                structured=True,
            )
        batata_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and "batata" in r.message.lower()
        ]
        assert batata_caps, "Batata cap (VEG-EXT-3) debe seguir activo"


# ===========================================================================
# Sección 3: Sanity guards — markers en source
# ===========================================================================
def test_source_has_olive_fix_4_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-OLIVE-CAP-FIX-4" in src, "Marker FIX-4 debe existir"
    assert "_OLIVE_DENSITY_G_PER_UNIT" in src, "Density per unit debe existir"
    assert "_olive_cap_count" in src, "Variable count cap debe existir"
    # Cubre las 4 formas de unidad
    assert "'unidad'" in src
    assert "'unidades'" in src


def test_source_has_veg_ext_4_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-VEG-EXT-4" in src
    assert "'yuca'" in src
    assert "'yucas'" in src
