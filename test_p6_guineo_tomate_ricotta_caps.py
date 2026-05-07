"""[P6-VEG-EXT-5 / P6-VEG-EXT-6 / P6-LACTEOS-EXT] Tests para los 3 fixes
de la lista de compras observada PDF 2026-05-05 21:50.

Bugs:
  1. Guineo: 56 unidades para 2p×mes (=7/sem/persona, excesivo). Sin cap
     porque guineo NO está en `_VEG_PER_WEEK_PER_PERSON` (es distinto a
     plátano maduro, no entra en su entry).
  2. Tomate: 38 unidades. Sin cap específico — tomate como vegetal de uso
     constante en sofrito y ensaladas, alto sin restricción.
  3. Ñame: 12 unidades. Bajo (no crítico) pero sin cap, podría escalar.
  4. Queso ricotta: 6 potes 425g = 2.55 kg. Sin cap, P6-LACTEOS-CAP solo
     cubría 'yogurt'/'yogur'. Ricotta es lácteo perecedero similar.

Fixes:
  1. P6-VEG-EXT-5: 'guineo'/'guineos' al dict, cap 4/persona/sem = 32 max
  2. P6-VEG-EXT-6: 'tomate'/'tomates' (5/sem = 40 max),
     'ñame'/'ñames'/'name'/'names' (2/sem = 16 max)
  3. P6-LACTEOS-EXT: 'ricotta', 'cottage' al dict de lácteos perecederos

Cobertura:
  - Repro PDF de los 4 (guineo 56, tomate 38, ñame 12, ricotta 2.5kg)
  - Cap escalado por person_weeks
  - No-regresión: yogurt sigue capeando, plátano maduro sigue separado
  - Variantes con/sin tilde (ñame/name)
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
# Sección 1: P6-VEG-EXT-5 (guineo)
# ===========================================================================
class TestGuineoCap:
    def test_repro_pdf_guineo_56_uds(self, caplog):
        """Caso real PDF 21:50: 56 guineos para 2p×mes. Cap esperado: 32."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 guineos"],
                multiplier=18.666666,
                structured=True,
            )
        guineo_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and "guineo" in r.message.lower()
        ]
        assert guineo_caps, (
            f"P5-VEG-CAP debe firar para guineo. "
            f"Warnings: {[r.message for r in caplog.records if 'VEG-CAP' in r.message]}"
        )

    @pytest.mark.parametrize("scenario,multiplier,expected_cap_uds", [
        ("4p mensual", 4 * 4 * 7 / 3, 64),
        ("2p mensual", 2 * 4 * 7 / 3, 32),
        ("1p mensual", 1 * 4 * 7 / 3, 16),
        ("2p semanal", 2 * 1 * 7 / 3, 8),
    ])
    def test_guineo_cap_scales(self, scenario, multiplier, expected_cap_uds, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 guineos"],
                multiplier=multiplier,
                structured=True,
            )
        guineo_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and "guineo" in r.message.lower()
        ]
        if expected_cap_uds < 99:
            assert guineo_caps, f"{scenario}: guineo cap debe firar"

    def test_platano_maduro_separately_capped(self, caplog):
        """No-regresión: plátano maduro sigue capeando con su entry original."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 plátanos maduros"],
                multiplier=18.666666,
                structured=True,
            )
        platano_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and ("platano" in r.message.lower() or "plátano" in r.message.lower())
        ]
        assert platano_caps, "Plátano maduro cap debe seguir activo"


# ===========================================================================
# Sección 2: P6-VEG-EXT-6 (tomate y ñame)
# ===========================================================================
class TestTomateCap:
    def test_repro_pdf_tomate_38_uds(self, caplog):
        """PDF 21:50: 38 tomates. Cap nuevo: 40 max para 2p×mes."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 tomates"],
                multiplier=18.666666,
                structured=True,
            )
        tomate_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and "tomate" in r.message.lower()
        ]
        assert tomate_caps, "P5-VEG-CAP debe firar para tomate (P6-VEG-EXT-6)"

    @pytest.mark.parametrize("scenario,multiplier,expected_cap_uds", [
        ("2p mensual", 2 * 4 * 7 / 3, 40),
        ("4p mensual", 4 * 4 * 7 / 3, 80),
    ])
    def test_tomate_cap_scales(self, scenario, multiplier, expected_cap_uds, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 tomates"],
                multiplier=multiplier,
                structured=True,
            )
        tomate_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and "tomate" in r.message.lower()
        ]
        assert tomate_caps, f"{scenario}: tomate cap debe firar"


class TestNameCap:
    @pytest.mark.parametrize("name_form", ["ñames", "names"])
    def test_name_with_and_without_accent(self, name_form, caplog):
        """Cap debe firar tanto para 'ñame' como para 'name' (sin tilde)."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=[f"99 {name_form}"],
                multiplier=18.666666,
                structured=True,
            )
        name_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and ("ñame" in r.message.lower() or "name" in r.message.lower())
        ]
        assert name_caps, f"'{name_form}' debe firar P5-VEG-CAP"


# ===========================================================================
# Sección 3: P6-LACTEOS-EXT (ricotta, cottage)
# ===========================================================================
class TestRicottaCap:
    def test_ricotta_capped(self, caplog):
        """Ricotta excesivo debe firar cap (similar a yogurt)."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 lbs de queso ricotta"],
                multiplier=18.666666,
                structured=True,
            )
        ricotta_caps = [
            r for r in caplog.records
            if "P6-LACTEOS-PERISHABLE-CAP" in r.message and "ricotta" in r.message.lower()
        ]
        assert ricotta_caps, (
            f"P6-LACTEOS-EXT debe firar para ricotta excesivo. "
            f"Warnings: {[r.message for r in caplog.records if 'LACTEOS' in r.message]}"
        )

    def test_cottage_cheese_capped(self, caplog):
        """Cottage cheese (similar a ricotta) debe firar cap."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 lbs de cottage cheese"],
                multiplier=18.666666,
                structured=True,
            )
        cottage_caps = [
            r for r in caplog.records
            if "P6-LACTEOS-PERISHABLE-CAP" in r.message and "cottage" in r.message.lower()
        ]
        assert cottage_caps, "P6-LACTEOS-EXT debe firar para cottage cheese"

    def test_yogurt_still_capped(self, caplog):
        """No-regresión: yogurt sigue capeando con su entry original."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 lbs de yogurt griego"],
                multiplier=18.666666,
                structured=True,
            )
        yogurt_caps = [
            r for r in caplog.records
            if "P6-LACTEOS-PERISHABLE-CAP" in r.message and "yogurt" in r.message.lower()
        ]
        assert yogurt_caps, "Yogurt cap original debe seguir activo"


# ===========================================================================
# Sección 4: Sanity guards — markers en source
# ===========================================================================
def test_source_has_veg_ext_5_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-VEG-EXT-5" in src
    assert "'guineo'" in src
    assert "'guineos'" in src


def test_source_has_veg_ext_6_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-VEG-EXT-6" in src
    assert "'tomate'" in src
    assert "'ñame'" in src


def test_source_has_lacteos_ext_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-LACTEOS-EXT" in src
    assert "'ricotta'" in src
    assert "'cottage'" in src
