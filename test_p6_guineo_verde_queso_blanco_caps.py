"""[P6-VEG-EXT-5-FIX / P6-LACTEOS-EXT-3] Tests para los 2 fixes finales
de la corrida PDF 2026-05-05 23:12.

Bugs:
  1. "Guineo verde: 168 Uds" para 2p × mes — absurdo. Mi P6-VEG-EXT-5
     capeó 'Guineo' (32 Uds en la lista) pero NO capturó la variante
     'Guineo verde' (item distinto en master_map, exact match no funciona).
  2. "Queso blanco: 9 paquetes (1 lb c/u)" = 9 lbs — sin cap. P6-LACTEOS-CAP
     cubría yogurt/ricotta/cottage/mozzarella pero NO queso blanco.

Fixes:
  1. P6-VEG-EXT-5-FIX: añadir 'guineo verde'/'guineos verdes' al dict
     `_VEG_PER_WEEK_PER_PERSON`. Cap mismo que guineo común (4/persona/sem
     = 32 max para 2p×mes).
  2. P6-LACTEOS-EXT-3: añadir 'queso blanco'/'queso fresco' al dict de
     lácteos perecederos. Cap 0.75 lb/persona/sem = 6 lbs para 2p×mes
     (queso blanco DR uso ~daily como acompañante).

Cobertura:
  - Repro PDF de los 2
  - Variantes con/sin plural
  - No-regresión: 'guineo' (sin verde) sigue capeando, mozzarella sigue capeando
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
# Sección 1: P6-VEG-EXT-5-FIX (guineo verde)
# ===========================================================================
class TestGuineoVerdeCap:
    def test_repro_pdf_guineo_verde_168_uds(self, caplog):
        """PDF 23:12: 168 guineos verdes para 2p×mes. Cap esperado: 32."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 guineos verdes"],
                multiplier=18.666666,
                structured=True,
            )
        guineo_caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message
            and "guineo" in r.message.lower() and "verde" in r.message.lower()
        ]
        assert guineo_caps, (
            f"P5-VEG-CAP debe firar para guineo verde. "
            f"Warnings: {[r.message for r in caplog.records if 'VEG-CAP' in r.message]}"
        )

    @pytest.mark.parametrize("name", [
        "99 guineo verde",
        "99 guineos verdes",
    ])
    def test_guineo_verde_variants(self, name, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=[name],
                multiplier=18.666666,
                structured=True,
            )
        caps = [
            r for r in caplog.records
            if "P5-VEG-CAP" in r.message and "guineo" in r.message.lower()
        ]
        assert caps, f"'{name}' debe firar P5-VEG-CAP"

    def test_guineo_comun_still_capped_separately(self, caplog):
        """No-regresión: 'Guineo' (sin verde) sigue capeando con su entry original."""
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
            and "verde" not in r.message.lower()
        ]
        assert guineo_caps, "Guineo común cap debe seguir activo"


# ===========================================================================
# Sección 2: P6-LACTEOS-EXT-3 (queso blanco)
# ===========================================================================
class TestQuesoBlancoCap:
    def test_repro_pdf_queso_blanco_9_lbs(self, caplog):
        """PDF 23:12: 9 lbs queso blanco. Cap esperado: 6 lbs."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 lbs de queso blanco"],
                multiplier=18.666666,
                structured=True,
            )
        queso_caps = [
            r for r in caplog.records
            if "P6-LACTEOS-PERISHABLE-CAP" in r.message
            and "queso blanco" in r.message.lower()
        ]
        assert queso_caps, (
            f"P6-LACTEOS-EXT-3 debe firar para queso blanco. "
            f"Warnings: {[r.message for r in caplog.records if 'LACTEOS' in r.message]}"
        )

    def test_queso_blanco_fresco_compound_capped(self, caplog):
        """Compound 'queso blanco fresco' (LLM emite frecuente con ambos
        adjetivos) debe firar via substring 'queso blanco'."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 lbs de queso blanco fresco"],
                multiplier=18.666666,
                structured=True,
            )
        queso_caps = [
            r for r in caplog.records
            if "P6-LACTEOS-PERISHABLE-CAP" in r.message
            and "queso" in r.message.lower()
        ]
        assert queso_caps, "Queso blanco fresco debe firar via substring"

    def test_mozzarella_still_capped(self, caplog):
        """No-regresión: mozzarella sigue capeando con P6-LACTEOS-EXT-2."""
        import logging
        with caplog.at_level(logging.WARNING):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["99 lbs de queso mozzarella"],
                multiplier=18.666666,
                structured=True,
            )
        mozz_caps = [
            r for r in caplog.records
            if "P6-LACTEOS-PERISHABLE-CAP" in r.message
            and "mozzarella" in r.message.lower()
        ]
        assert mozz_caps, "Mozzarella cap debe seguir activo"

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
            if "P6-LACTEOS-PERISHABLE-CAP" in r.message
            and "yogurt" in r.message.lower()
        ]
        assert yogurt_caps, "Yogurt cap debe seguir activo"


# ===========================================================================
# Sección 3: Sanity guards — markers en source
# ===========================================================================
def test_source_has_guineo_verde_fix_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-VEG-EXT-5-FIX" in src
    assert "'guineo verde'" in src
    assert "'guineos verdes'" in src


def test_source_has_lacteos_ext_3_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-LACTEOS-EXT-3" in src
    assert "'queso blanco'" in src
    assert "'queso fresco'" in src
