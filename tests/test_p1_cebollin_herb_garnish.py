"""[P1-CEBOLLIN-HERB-GARNISH · 2026-07-07] "Cebollín al gusto → 1 paquete 375g RD$229 ×ciclo".

Review visual plan 30d: la lista mostraba "Cebollín 1 paquete (25 unid (~375 g)) RD$229"
para una receta que pedía "Cebollín fresco al gusto" (guarnición). Perecedero → ×4.286 =
~RD$982/ciclo por cebollín de adorno, empujando el plan sobre presupuesto.

Causa (2 capas):
  DATOS — cebollín estaba mal-modelado en master_ingredients como EMPAQUE
  (market_container='paquete', container_weight_g=375, price_per_unit=229) en vez de
  hierba de MAZO como cilantro/perejil (default_unit='mazo', sin container,
  density=50 g/mazo, price_per_unit por mazo). Cualquier cantidad redondeaba a 1 paquete.
  Fix de datos: UPDATE master → modelo mazo (default_unit='mazo', container NULL,
  density 50, price_per_unit 31). [aplicado en Neon 2026-07-07]

  CÓDIGO — cebollín estaba OMITIDO de is_herb_mazo (routing a mazo ~50g) y de
  _HERB_NAMES_FOR_CAP (cap por persona-semana). Añadido a ambos.

Resultado: "cebollín al gusto" → "1 Mazo RD$31", cap por ciclo → sin ×multiplicación.
tooltip-anchor: P1-CEBOLLIN-HERB-GARNISH
"""
from __future__ import annotations

from pathlib import Path

import pytest

import shopping_calculator as sc

_SRC = (Path(sc.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")

# Modelo mazo del master (espejo del UPDATE de datos aplicado a Neon).
_CEBOLLIN_MAZO_MASTER = [{
    "name": "Cebollín", "category": "Vegetales", "default_unit": "mazo",
    "price_per_lb": 277.0, "price_per_unit": 31.0, "density_g_per_unit": 50.0,
    "market_container": None, "container_weight_g": None, "available_sizes_g": None,
    "market_packages": None, "shelf_life_days": 7, "aliases": [],
}]


@pytest.fixture()
def cebollin_master(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_CEBOLLIN_MAZO_MASTER))
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


def _ceb(result):
    return next((r for r in result if isinstance(r, dict) and "cebollin" in str(r.get("name", "")).lower()
                 or "cebollín" in str(r.get("name", "")).lower()), None)


# --- Parser-based: cebollín en ambos paths de hierba ---
def test_cebollin_in_herb_mazo_regex():
    assert "P1-CEBOLLIN-HERB-GARNISH" in _SRC
    import re
    m = re.search(r"is_herb_mazo = bool\(re\.search\(r'\\b\(([^)]*)\)", _SRC)
    assert m, "no se aisló el regex is_herb_mazo"
    assert "cebollin" in m.group(1), "cebollín debe estar en is_herb_mazo (routing a mazo)"


def test_cebollin_in_herb_cap_set():
    import re
    m = re.search(r"_HERB_NAMES_FOR_CAP = \{(.*?)\}", _SRC, re.DOTALL)
    assert m and "'cebollin'" in m.group(1), "cebollín debe estar en _HERB_NAMES_FOR_CAP (cap por ciclo)"


# --- Funcional: al gusto → mazo barato, con cap por ciclo ---
def test_al_gusto_becomes_mazo_not_package(cebollin_master):
    result = sc.aggregate_and_deduct_shopping_list(
        ["Cebollín fresco al gusto"], [], structured=True, multiplier=1.0,
    )
    it = _ceb(result)
    assert it is not None, f"cebollín ausente: {result}"
    disp = str(it.get("display_string") or it.get("display_qty")).lower()
    assert "mazo" in disp, f"debe ser un mazo, no un paquete: {disp}"
    assert "paquete" not in disp and "375" not in disp, f"no debe ser el paquete de 375g: {disp}"
    cost = float(it.get("estimated_cost_rd") or 0)
    assert cost < 60.0, f"1 mazo ~RD$31, NO el paquete RD$229: RD${cost}"


def test_cycle_multiplier_capped(cebollin_master):
    """A ×4.286 (ciclo mensual) el cebollín de guarnición NO se multiplica: el cap
    de hierbas lo mantiene en ~1 mazo (era RD$229 × 4.286 = ~RD$982)."""
    result = sc.aggregate_and_deduct_shopping_list(
        ["Cebollín fresco al gusto"], [], structured=True, multiplier=4.286,
    )
    it = _ceb(result)
    assert it is not None
    cost = float(it.get("estimated_cost_rd") or 0)
    assert cost < 120.0, f"guarnición 'al gusto' no debe escalar ×ciclo: RD${cost}"
