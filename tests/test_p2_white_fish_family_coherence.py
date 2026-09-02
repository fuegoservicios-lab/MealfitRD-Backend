"""[P2-WHITE-FISH-FAMILY-COHERENCE · 2026-09-02] Mero/tilapia/chillo y «Filete de pescado blanco»
son la MISMA compra para el guard de coherencia.

Medido 3 veces el 02-sep (planes 6c6989d9, adc792f0, cc8136e0): la receta decía «145 g de filete
de mero», la lista compraba «Filete de pescado blanco» (fila genérica del catálogo cuyos alias
incluyen mero/tilapia/chillo) y el guard reportaba `Mero [expected_only]` + `Pescado
[aggregated_only]` como divergencia crítica marginal. Sin reintento, pero ruido en cada plan.

Tooltip-anchor: P2-WHITE-FISH-FAMILY-COHERENCE | _white_fish_family_canonical
"""
import pytest

import shopping_calculator
from shopping_calculator import _canonicalize_for_coherence, _white_fish_family_canonical, canonicalize_fish_seafood


@pytest.fixture
def no_master_db(monkeypatch):
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [])
    if hasattr(shopping_calculator, "_COHERENCE_ALIAS_MAP_CACHE"):
        monkeypatch.setattr(shopping_calculator, "_COHERENCE_ALIAS_MAP_CACHE", None, raising=False)


@pytest.mark.parametrize("raw", ["Mero", "Tilapia", "Chillo", "Dorado", "Pescado", "Filete de pescado blanco", "pescado blanco", "MERO"])
def test_family_collapses_to_pescado(raw):
    assert _white_fish_family_canonical(raw) == "Pescado"


@pytest.mark.parametrize("raw", ["Salmón", "Bacalao", "Atún", "Sardinas en lata", "Camarones", "Pulpo", "Pollo", "Huevo"])
def test_distinct_products_untouched(raw):
    assert _white_fish_family_canonical(raw) == raw


def test_recipe_mero_and_list_white_fish_meet_in_the_guard(no_master_db):
    canon = _canonicalize_for_coherence(["filete de mero", "Filete de pescado blanco"])
    assert canon == {"Pescado"}, canon


def test_species_helper_still_returns_species():
    # el agregador y test_p1_audit_2 dependen de la especie; solo el guard colapsa
    assert canonicalize_fish_seafood("filete de mero") == "Mero"


def test_knob_off_restores_species_split(monkeypatch, no_master_db):
    monkeypatch.setenv("MEALFIT_COHERENCE_WHITE_FISH_FAMILY", "false")
    canon = _canonicalize_for_coherence(["filete de mero", "Filete de pescado blanco"])
    assert "Mero" in canon and len(canon) == 2


def test_salmon_stays_apart_in_the_guard(no_master_db):
    canon = _canonicalize_for_coherence(["filete de salmón", "Filete de pescado blanco"])
    assert canon == {"Salmón", "Pescado"}, canon
