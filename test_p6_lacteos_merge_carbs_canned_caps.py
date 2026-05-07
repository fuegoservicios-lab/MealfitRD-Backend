"""[P6-LACTEOS-MERGE / P6-VEG-EXT-7 / P6-CARBS-CAP / P6-CANNED-PROTEIN-CAP /
P6-LACTEOS-EXT-2] Tests para los 5 fixes de la lista PDF 2026-05-05 22:42.

Bugs:
  1. Yogurt: "Yogurt griego: 13 potes" Y "Yogurt: 7 Uds" en lista → suma
     20 potes (>>cap 12). Aggregator no unificó variantes.
  2. Brócoli: 14 cabezas (~15 lbs) — sin cap, no estaba en _VEG_PER_WEEK_PER_PERSON.
  3. Tortilla integral: 7 paquetes (288g) = 2 kg — sin cap.
  4. Atún en agua: 19 latas (184g) = 3.5 kg — sin cap.
  5. Queso mozzarella: 5 paquetes (1 lb) = 5 lbs — sin cap (P6-LACTEOS-EXT
     cubría ricotta/cottage pero no mozzarella).

Fixes:
  1. P6-LACTEOS-MERGE: si hay key "Yogurt" genérico Y variante "Yogurt X",
     mergear el genérico en la específica antes de aplicar cap.
  2. P6-VEG-EXT-7: añadir 'brocoli'/'brocolis' al dict (cap 1/persona/sem
     = 8 cabezas para 2p×mes).
  3. P6-CARBS-CAP: nuevo bloque para tortillas/pan empaquetado (cap 1
     paquete por 2 person-weeks = 4 paquetes para 2p×mes).
  4. P6-CANNED-PROTEIN-CAP: nuevo bloque para atún/sardinas/salmón en
     lata (cap 1 lata/persona/sem = 8 latas para 2p×mes).
  5. P6-LACTEOS-EXT-2: añadir 'mozzarella' al dict de lácteos perecederos
     (cap 0.5 lb/persona/sem = 4 lbs para 2p×mes).

Cobertura:
  - Repro PDF de los 5
  - No-regresión: yogurt griego sigue capeando (no rompe LACTEOS-CAP existente)
  - Variantes (sardinas, mostaza, pan integral, etc.)
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
# Sección 1: P6-LACTEOS-MERGE (yogurt unificado)
# ===========================================================================
class TestLacteosMerge:
    def test_yogurt_generico_se_merge_en_especifico(self, caplog):
        """Si hay 'Yogurt' genérico Y 'Yogurt griego', el genérico se folds."""
        import logging
        with caplog.at_level(logging.INFO):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=[
                    "1 pote de yogurt griego sin azúcar",
                    "1 pote de yogurt",  # genérico
                    "1 pote de yogurt griego sin azúcar",
                ],
                multiplier=18.666666,
                structured=True,
            )
        merge_logs = [
            r for r in caplog.records
            if "P6-LACTEOS-MERGE" in r.message
        ]
        assert merge_logs, (
            f"P6-LACTEOS-MERGE debe firar. "
            f"INFO logs: {[r.message for r in caplog.records if 'MERGE' in r.message]}"
        )


# ===========================================================================
# Sección 2: P6-VEG-EXT-7 (brócoli)
# ===========================================================================
class TestBrocoliCap:
    def test_repro_pdf_brocoli_excesivo(self, caplog):
        """PDF 22:42: 14 cabezas brócoli para 2p×mes. Cap esperado: 8."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 brócolis"],
                multiplier=18.666666,
                structured=True,
            )
        brocoli_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and ("broc" in r.message.lower() or "bróc" in r.message.lower())
        ]
        assert brocoli_caps, (
            f"P5-VEG-CAP debe firar para brócoli. "
            f"Warnings: {[r.message for r in caplog.records if 'VEG-CAP' in r.message]}"
        )

    @pytest.mark.parametrize("scenario,multiplier", [
        ("2p mensual", 2 * 4 * 7 / 3),
        ("4p mensual", 4 * 4 * 7 / 3),
    ])
    def test_brocoli_cap_scales(self, scenario, multiplier, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 brócolis"],
                multiplier=multiplier,
                structured=True,
            )
        brocoli_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and ("broc" in r.message.lower() or "bróc" in r.message.lower())
        ]
        assert brocoli_caps, f"{scenario}: brócoli cap debe firar"


# ===========================================================================
# Sección 3: P6-CARBS-CAP (tortilla, pan integral)
# ===========================================================================
class TestCarbsCap:
    def test_repro_pdf_tortilla_integral_7_paquetes(self, caplog):
        """PDF 22:42: 7 paquetes tortilla integral. Cap esperado: 4."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["1 paquete de tortilla integral"] * 5,
                multiplier=18.666666,
                structured=True,
            )
        carb_caps = [
            r for r in caplog.records
            if "P6-CARBS-CAP" in r.message
        ]
        assert carb_caps, (
            f"P6-CARBS-CAP debe firar para tortilla integral. "
            f"Warnings: {[r.message for r in caplog.records if 'CARBS' in r.message]}"
        )

    @pytest.mark.parametrize("carb", [
        "tortilla integral",
        "pan integral",
        "pan de molde",
    ])
    def test_other_carbs_capped(self, carb, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=[f"1 paquete de {carb}"] * 5,
                multiplier=18.666666,
                structured=True,
            )
        carb_caps = [r for r in caplog.records if "P6-CARBS-CAP" in r.message]
        assert carb_caps, f"{carb} debe firar P6-CARBS-CAP"


# ===========================================================================
# Sección 4: P6-CANNED-PROTEIN-CAP (atún en agua)
# ===========================================================================
class TestCannedProteinCap:
    def test_repro_pdf_atun_19_latas(self, caplog):
        """PDF 22:42: 19 latas atún en agua. Cap esperado: 8."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["1 lata de atún en agua"] * 5,
                multiplier=18.666666,
                structured=True,
            )
        atun_caps = [
            r for r in caplog.records
            if "P6-CANNED-PROTEIN-CAP" in r.message
        ]
        assert atun_caps, (
            f"P6-CANNED-PROTEIN-CAP debe firar para atún. "
            f"Warnings: {[r.message for r in caplog.records if 'CANNED' in r.message]}"
        )

    @pytest.mark.parametrize("canned", [
        "atún",
        "sardinas",
        "atún en agua",
    ])
    def test_other_canned_proteins_capped(self, canned, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=[f"1 lata de {canned}"] * 5,
                multiplier=18.666666,
                structured=True,
            )
        atun_caps = [r for r in caplog.records if "P6-CANNED-PROTEIN-CAP" in r.message]
        assert atun_caps, f"{canned} debe firar P6-CANNED-PROTEIN-CAP"


# ===========================================================================
# Sección 5: P6-LACTEOS-EXT-2 (mozzarella)
# ===========================================================================
class TestMozzarellaCap:
    def test_mozzarella_capped(self, caplog):
        """Mozzarella excesivo debe firar cap (similar a ricotta)."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 lbs de queso mozzarella"],
                multiplier=18.666666,
                structured=True,
            )
        mozzarella_caps = [
            r for r in caplog.records
            if "P6-LACTEOS-PERISHABLE-CAP" in r.message and "mozzarella" in r.message.lower()
        ]
        assert mozzarella_caps, (
            f"P6-LACTEOS-EXT-2 debe firar para mozzarella excesivo. "
            f"Warnings: {[r.message for r in caplog.records if 'LACTEOS' in r.message]}"
        )

    def test_yogurt_still_capped(self, caplog):
        """No-regresión: yogurt sigue capeando."""
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
        assert yogurt_caps, "Yogurt cap debe seguir activo"


# ===========================================================================
# Sección 6: Sanity guards — markers en source
# ===========================================================================
def test_source_has_lacteos_merge_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-LACTEOS-MERGE" in src
    assert "_specific_yogurt_keys" in src


def test_source_has_veg_ext_7_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-VEG-EXT-7" in src
    assert "'brocoli'" in src


def test_source_has_carbs_cap_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-CARBS-CAP" in src
    assert "_CARBS_PACKAGE_NAMES_FOR_CAP" in src
    assert "tortilla integral" in src


def test_source_has_canned_protein_cap_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-CANNED-PROTEIN-CAP" in src
    assert "_CANNED_PROTEIN_NAMES_FOR_CAP" in src
    assert "atun" in src


def test_source_has_lacteos_ext_2_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-LACTEOS-EXT-2" in src
    assert "'mozzarella'" in src
