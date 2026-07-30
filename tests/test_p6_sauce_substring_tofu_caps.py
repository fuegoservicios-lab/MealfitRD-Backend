"""[P6-SAUCE-CAP-FIX / P6-TOFU-CAP] Tests para los 2 fixes adicionales
post PDF 2026-05-05 23:33.

Bugs:
  1. "Salsa de soya baja en sodio: 10 botellas" — P6-SAUCE-CAP exact match
     del set no captura variantes. Master_map preserva "baja en sodio" en
     el name canonicalizado.
  2. "Tofu: 31 lbs" — sin cap específico. Tofu se vende en paquetes/lbs y
     LLM tiende a pedirlo en cantidades altas como sustituto de carne.

Fixes:
  1. P6-SAUCE-CAP-FIX: cambio de exact match (set) a substring match (tuple).
     Patrón análogo a P6-OLIVE-CAP-FIX. Captura "salsa de soya baja en sodio",
     "salsa de soya light", "ketchup picante", etc.
  2. P6-TOFU-CAP: añadir 'tofu'/'tofu firme'/'tofu suave' al
     `_VEG_PER_WEEK_PER_PERSON` (cap 1 lb/persona/sem = 8 lbs para 2p×mes).

Cobertura:
  - Repro PDF: salsa de soya baja en sodio + tofu 31 lbs
  - Variantes (mayonesa light, mostaza dijon, ketchup picante)
  - No-regresión: salsa de tomate sin variantes sigue capeando
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


def _is_item_cap(record, marker: str, food: str = "") -> bool:
    """[P2-CAP-LOG-LEVEL · 2026-07-29] La línea PER-ÍTEM del cap, no el resumen agregado.

    Estos tests capturaban en WARNING y comprobaban `marker in message`. Tras bajar los emisores de
    cap a INFO siguieron en verde — **por accidente**: lo que cazaban ya no era el cap del ítem sino
    el resumen agregado `[P2-CAP-LOG-LEVEL]`, que es WARNING y cita dentro el nombre del alimento y
    el `reason` del cap. Al de-duplicar ese resumen por contenido, 5 de los 14 se pusieron rojos y
    destaparon el acoplamiento. Un test que pasa por una línea distinta a la que cree estar mirando
    no protege nada: aquí se exige la del ítem y se excluye el resumen explícitamente.
    """
    msg = record.message
    if "[P2-CAP-LOG-LEVEL]" in msg:      # el resumen agregado NO cuenta como cap del ítem
        return False
    return marker in msg and (not food or food in msg.lower())


# ===========================================================================
# Sección 1: P6-SAUCE-CAP-FIX (substring match)
# ===========================================================================
class TestSauceCapFix:
    def test_repro_pdf_salsa_de_soya_baja_en_sodio(self, caplog):
        """PDF 23:33: 'Salsa de soya baja en sodio: 10 botellas'.
        Pre-fix: exact match del set falla, no capea. Post-fix: substring
        'salsa de soya' matchea variante."""
        import logging
        with caplog.at_level(logging.INFO):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["1 botella de salsa de soya baja en sodio"] * 5,
                multiplier=18.666666,
                structured=True,
            )
        sauce_caps = [
            r for r in caplog.records
            if _is_item_cap(r, "P6-SAUCE-CAP", "soya")
        ]
        assert sauce_caps, (
            f"P6-SAUCE-CAP-FIX debe firar para 'salsa de soya baja en sodio'. "
            f"Warnings: {[r.message for r in caplog.records if 'SAUCE' in r.message]}"
        )

    @pytest.mark.parametrize("variant", [
        "mayonesa light",
        "mostaza dijon",
        "ketchup picante",
        "salsa de tomate organica",
    ])
    def test_other_sauce_variants_now_capped(self, variant, caplog):
        """Variantes con adjetivos también deben firar (substring match)."""
        import logging
        with caplog.at_level(logging.INFO):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=[f"1 lata de {variant}"] * 5,
                multiplier=18.666666,
                structured=True,
            )
        sauce_caps = [r for r in caplog.records if _is_item_cap(r, "P6-SAUCE-CAP")]
        assert sauce_caps, f"'{variant}' debe firar P6-SAUCE-CAP via substring"

    def test_salsa_tomate_basica_still_capped(self, caplog):
        """No-regresión: caso base sin variantes sigue funcionando."""
        import logging
        with caplog.at_level(logging.INFO):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["1 lata de salsa de tomate"] * 5,
                multiplier=18.666666,
                structured=True,
            )
        sauce_caps = [
            r for r in caplog.records
            if _is_item_cap(r, "P6-SAUCE-CAP", "tomate")
        ]
        assert sauce_caps, "Salsa de tomate (caso base) debe seguir capeando"


# ===========================================================================
# Sección 2: P6-TOFU-CAP (tofu)
# ===========================================================================
class TestTofuCap:
    def test_repro_pdf_tofu_31_lbs(self, caplog):
        """PDF 23:33: 31 lbs tofu para 2p × mes — absurdo. Cap esperado: 8 lbs."""
        import logging
        with caplog.at_level(logging.INFO):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["500g de tofu firme"] * 5,
                multiplier=18.666666,
                structured=True,
            )
        tofu_caps = [
            r for r in caplog.records
            if _is_item_cap(r, "P5-VEG-CAP", "tofu")
        ]
        assert tofu_caps, (
            f"P5-VEG-CAP debe firar para tofu via P6-TOFU-CAP. "
            f"Warnings: {[r.message for r in caplog.records if 'VEG-CAP' in r.message]}"
        )

    @pytest.mark.parametrize("tofu_variant", [
        "tofu",
        "tofu firme",
        "tofu suave",
    ])
    def test_tofu_variants_capped(self, tofu_variant, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=[f"500g de {tofu_variant}"] * 5,
                multiplier=18.666666,
                structured=True,
            )
        tofu_caps = [
            r for r in caplog.records
            if _is_item_cap(r, "P5-VEG-CAP", "tofu")
        ]
        assert tofu_caps, f"'{tofu_variant}' debe firar cap"

    @pytest.mark.parametrize("scenario,multiplier", [
        ("2p mensual", 2 * 4 * 7 / 3),
        ("4p mensual", 4 * 4 * 7 / 3),
    ])
    def test_tofu_cap_scales(self, scenario, multiplier, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            aggregate_and_deduct_shopping_list(
                plan_ingredients=["500g de tofu firme"] * 5,
                multiplier=multiplier,
                structured=True,
            )
        tofu_caps = [
            r for r in caplog.records
            if _is_item_cap(r, "P5-VEG-CAP", "tofu")
        ]
        assert tofu_caps, f"{scenario}: tofu cap debe firar"


# ===========================================================================
# Sección 3: Sanity guards — markers en source
# ===========================================================================
def test_source_has_sauce_cap_fix_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-SAUCE-CAP-FIX" in src
    assert "_SAUCE_NAME_SUBSTRINGS" in src
    # Substring match logic (any())
    assert "any(s in _name_norm for s in _SAUCE_NAME_SUBSTRINGS)" in src


def test_source_has_tofu_cap_marker():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P6-TOFU-CAP" in src
    assert "'tofu'" in src
    assert "'tofu firme'" in src
