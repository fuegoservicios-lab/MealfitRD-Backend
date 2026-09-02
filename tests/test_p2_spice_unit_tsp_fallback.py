"""[P2-SPICE-UNIT-TSP-FALLBACK · 2026-09-02] «unidad» de una especia = una cucharadita
(density_g_per_cup / 48), acotado a condimentos y con knob.

Medido en prod: Canela en polvo, Orégano dominicano y Pimienta negra tienen
`density_g_per_cup` (124/24/96) y NO `density_g_per_unit`; las recetas las piden en
«unidad» y cada recálculo de la lista emitía 3-4 WARNING de convert_amount y saltaba la
deducción. Para Fideos/Arroz «unidad» no es una cucharadita: siguen estrictos.

Tooltip-anchor: P2-SPICE-UNIT-TSP-FALLBACK | _spice_unit_as_teaspoon
"""
import logging

import pytest

import db_inventory


def test_spice_unit_is_a_teaspoon_from_cup_density():
    assert db_inventory._resolve_unit_weight({"name": "Canela en polvo", "density_g_per_cup": 124}) == pytest.approx(2.583, abs=0.001)
    assert db_inventory._resolve_unit_weight({"name": "Orégano dominicano", "density_g_per_cup": 24}) == pytest.approx(0.5)
    assert db_inventory._resolve_unit_weight({"name": "Pimienta negra", "density_g_per_cup": 96}) == pytest.approx(2.0)


def test_explicit_per_unit_density_still_wins():
    assert db_inventory._resolve_unit_weight({"name": "Canela en polvo", "density_g_per_unit": 3, "density_g_per_cup": 124}) == 3.0


def test_non_spice_pantry_item_stays_strict():
    assert db_inventory._resolve_unit_weight({"name": "Fideos", "density_g_per_cup": 150}) is None
    assert db_inventory._resolve_unit_weight({"name": "Arroz integral", "density_g_per_cup": 190}) is None


def test_spice_without_cup_density_stays_strict():
    assert db_inventory._resolve_unit_weight({"name": "Comino", "density_g_per_cup": None}) is None


def test_knob_off_restores_previous_behaviour(monkeypatch):
    monkeypatch.setenv("MEALFIT_SPICE_UNIT_AS_TSP", "false")
    assert db_inventory._resolve_unit_weight({"name": "Canela en polvo", "density_g_per_cup": 124}) is None


def test_convert_amount_no_longer_warns_for_spices(caplog):
    item = {"name": "Orégano dominicano", "density_g_per_cup": 24}
    with caplog.at_level(logging.WARNING, logger="db_inventory"):
        g = db_inventory.convert_amount(4.666666666666667, "unidad", "g", item)
    assert g == pytest.approx(2.333, abs=0.001)
    assert not [r for r in caplog.records if "convert_amount" in r.getMessage()]
