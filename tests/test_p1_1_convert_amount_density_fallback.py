"""[P1-1 · 2026-05-08] Tests de `convert_amount` con cadena de fallback de densidad.

Bug original (audit 2026-05-07):
  `convert_amount` en backend/db_inventory.py asumía `density_g_per_cup or 150.0`
  como default cuando `master_item` no tenía la densidad. Para grasas/líquidos
  densos (aceite ~218, miel ~340, peanut butter ~258 g/taza), eso producía
  conversiones cruzadas masa↔volumen con error ~30-50%, propagándose a
  deducción de inventario y a la coherencia receta↔lista.

Fix:
  1. Cadena: master["density_g_per_cup"] → VOLUMETRIC_DENSITIES[name] →
     knob `MEALFIT_CROSS_UNIT_CONVERSION_STRICT`.
  2. Default knob = True → retorna None + WARNING (caller skipea fila).
  3. Knob = False → cae a 150 g/taza legacy (escape hatch operacional).
  4. Migración backfill de top items con densidad NULL (aceite, miel, leche,
     mantequilla, salsas, quesos, huevo, especias).

Cobertura de este test:
  - master tiene density_g_per_cup → usa ese valor (camino feliz).
  - master vacío + nombre matchea VOLUMETRIC_DENSITIES → usa fallback constants.
  - master vacío + nombre desconocido + STRICT=True (default) → None + WARNING.
  - master vacío + nombre desconocido + STRICT=False → 150 g/taza + WARNING.
  - count→mass: master tiene density_g_per_unit → lo usa.
  - count→mass: master vacío + nombre matchea UNIT_WEIGHTS → fallback.
  - count→mass: nombre desconocido + STRICT=True → None.
  - Mass↔mass y vol↔vol no requieren densidad (no se afectan).
"""
import logging
import os
import sys
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_env():
    """Garantiza que cada test arranca con el knob limpio."""
    prev = os.environ.pop("MEALFIT_CROSS_UNIT_CONVERSION_STRICT", None)
    yield
    if prev is not None:
        os.environ["MEALFIT_CROSS_UNIT_CONVERSION_STRICT"] = prev


def _import_convert_amount():
    """Importa fresco para que el knob lazy lea el env actual."""
    if "db_inventory" in sys.modules:
        del sys.modules["db_inventory"]
    if "graph_orchestrator" in sys.modules:
        # forzar re-registro del knob
        del sys.modules["graph_orchestrator"]
    from db_inventory import convert_amount
    return convert_amount


def test_same_unit_returns_qty_unchanged():
    convert_amount = _import_convert_amount()
    assert convert_amount(100, "g", "g", {}) == 100


def test_mass_to_mass_no_density_needed():
    convert_amount = _import_convert_amount()
    assert convert_amount(1, "kg", "g", {}) == pytest.approx(1000.0)
    assert convert_amount(500, "g", "kg", {}) == pytest.approx(0.5)


def test_volume_to_volume_no_density_needed():
    convert_amount = _import_convert_amount()
    assert convert_amount(1, "l", "ml", {}) == pytest.approx(1000.0)
    assert convert_amount(2, "tazas", "ml", {}) == pytest.approx(480.0)


def test_cross_domain_uses_master_density_when_present():
    """Aceite con density_g_per_cup=218 (real) debe convertir 218g → ~1 taza."""
    convert_amount = _import_convert_amount()
    master = {"name": "Aceite de oliva", "density_g_per_cup": 218}
    result = convert_amount(218, "g", "taza", master)
    assert result == pytest.approx(1.0, rel=0.01)


def test_cross_domain_falls_back_to_volumetric_densities_constants():
    """Si master_item no trae densidad pero el nombre matchea VOLUMETRIC_DENSITIES.

    aceite en VOLUMETRIC_DENSITIES = 0.920 g/ml × 236.588 ≈ 217.66 g/taza.
    Convertir 217.66g → ~1 taza.
    """
    convert_amount = _import_convert_amount()
    master = {"name": "aceite"}
    result = convert_amount(217.66, "g", "taza", master)
    assert result == pytest.approx(1.0, rel=0.01)


def test_cross_domain_strict_default_returns_none_for_unknown(caplog):
    """STRICT=True (default) + densidad irresoluble → None + WARNING."""
    convert_amount = _import_convert_amount()
    master = {"name": "ingrediente_inexistente_zxq"}
    with caplog.at_level(logging.WARNING):
        result = convert_amount(100, "g", "taza", master)
    assert result is None
    assert any("[P1-1]" in rec.message for rec in caplog.records)


def test_cross_domain_non_strict_falls_to_legacy_150(caplog):
    """STRICT=False → cae a 150 g/taza con WARNING."""
    os.environ["MEALFIT_CROSS_UNIT_CONVERSION_STRICT"] = "false"
    convert_amount = _import_convert_amount()
    master = {"name": "ingrediente_inexistente_zxq"}
    with caplog.at_level(logging.WARNING):
        result = convert_amount(150, "g", "taza", master)
    assert result is not None
    assert result == pytest.approx(1.0, rel=0.01)
    assert any("strict=False" in rec.message for rec in caplog.records)


def test_cross_domain_empty_master_strict_returns_none():
    convert_amount = _import_convert_amount()
    assert convert_amount(100, "ml", "g", {}) is None


def test_count_to_mass_uses_master_unit_weight():
    convert_amount = _import_convert_amount()
    master = {"name": "Huevo", "density_g_per_unit": 50}
    result = convert_amount(2, "unidad", "g", master)
    assert result == pytest.approx(100.0)


def test_count_to_mass_falls_back_to_unit_weights_constants():
    """Sin density_g_per_unit pero nombre matchea UNIT_WEIGHTS."""
    convert_amount = _import_convert_amount()
    master = {"name": "platano verde"}
    result = convert_amount(1, "unidad", "g", master)
    assert result == pytest.approx(280.0, rel=0.01)


def test_count_to_mass_strict_returns_none_for_unknown():
    convert_amount = _import_convert_amount()
    master = {"name": "fruta_desconocida_yyq"}
    assert convert_amount(1, "unidad", "g", master) is None


def test_rebanada_overrides_density_to_25g():
    """Rebanada hardcoded 25g (típico pan)."""
    convert_amount = _import_convert_amount()
    master = {"name": "pan", "density_g_per_unit": 999}
    result = convert_amount(2, "rebanada", "g", master)
    assert result == pytest.approx(50.0)


def test_oil_old_default_would_have_been_off_by_30pct():
    """Caso real del bug: 218g de aceite ↔ 1 taza usando densidad correcta."""
    convert_amount = _import_convert_amount()
    master = {"name": "aceite", "density_g_per_cup": 218}
    correct = convert_amount(218, "g", "taza", master)
    legacy = convert_amount(218, "g", "taza", {"name": "aceite", "density_g_per_cup": 150})
    assert correct == pytest.approx(1.0, rel=0.01)
    assert legacy == pytest.approx(1.453, rel=0.01)
    assert abs(legacy - correct) / correct > 0.30


def test_knob_registered_in_registry():
    """Asegura que MEALFIT_CROSS_UNIT_CONVERSION_STRICT pasa por _env_bool."""
    if "graph_orchestrator" in sys.modules:
        del sys.modules["graph_orchestrator"]
    convert_amount = _import_convert_amount()
    # disparar lectura del knob via path strict
    convert_amount(100, "g", "taza", {"name": "x_unknown_zzz"})
    from graph_orchestrator import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_CROSS_UNIT_CONVERSION_STRICT" in snap
    assert snap["MEALFIT_CROSS_UNIT_CONVERSION_STRICT"]["type"] == "bool"
    assert snap["MEALFIT_CROSS_UNIT_CONVERSION_STRICT"]["default"] is True
