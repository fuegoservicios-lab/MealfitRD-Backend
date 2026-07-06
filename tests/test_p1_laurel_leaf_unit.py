"""[P1-LAUREL-LEAF-UNIT · 2026-07-06] "4.67 hojas de laurel → RD$701" arreglado.

PDF del owner (plan ff673061): "Laurel: 4.67 hojas RD$701". Forensic SQL: la fila
master de Laurel estaba COMPLETA (pote 100 g, RD$150, market_packages,
density_g_per_unit=0.6 g/hoja) — el bug era de código: "hojas" es count unit y
nada la convertía a peso, así que el Bloque 1 de envases jamás corría y el costeo
caía a count × price_per_unit = 4.67 × RD$150 (el precio del POTE entero cobrado
POR HOJA). Pre-paso hojas→gramos (como Ajo dientes→cabezas y Huevo→cartones) →
1 pote (100 g) = RD$150 una sola vez.
"""
import pytest

import shopping_calculator as sc

_LAUREL_MASTER = [{
    "name": "Laurel", "category": "Despensa", "default_unit": "frasco",
    "price_per_lb": 680.39, "price_per_unit": 150.0, "density_g_per_unit": 0.6,
    "market_container": "pote", "container_weight_g": 100.0,
    "available_sizes_g": [100],
    "market_packages": [{"unit": "pote", "grams": 100, "label": "100 g", "price": 150}],
    "shelf_life_days": 365, "aliases": [],
}]


@pytest.fixture()
def laurel_master(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_LAUREL_MASTER))
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


def _laurel(result):
    return next((r for r in result if isinstance(r, dict) and "laurel" in str(r.get("name", "")).lower()), None)


def test_leaves_buy_one_pote_not_per_leaf(laurel_master):
    result = sc.aggregate_and_deduct_shopping_list(
        ["2 hojas de laurel"], [], structured=True,
    )
    item = _laurel(result)
    assert item is not None, f"laurel ausente: {result}"
    disp = str(item.get("display_qty", ""))
    assert "hoja" not in disp.lower(), f"las hojas deben unitarizarse a envase: {disp}"
    assert "pote" in disp.lower() and "100" in disp, f"esperaba '1 pote (100 g)': {disp}"
    cost = float(item.get("estimated_cost_rd") or 0)
    assert cost == 150.0, f"1 pote = RD$150 (NO count×150={2 * 150}): RD${cost}"


def test_pdf_repro_4_67_leaves_costs_one_pote(laurel_master):
    """El caso exacto del PDF: 4.67 hojas costaban RD$701. Ahora: RD$150."""
    result = sc.aggregate_and_deduct_shopping_list(
        ["1 hoja de laurel", "2 hojas de laurel"], [], structured=True, multiplier=1.5555,
    )
    item = _laurel(result)
    assert item is not None
    cost = float(item.get("estimated_cost_rd") or 0)
    assert cost == 150.0, f"cualquier cantidad razonable de hojas = 1 pote RD$150: RD${cost}"


def test_density_fallback_when_master_missing(monkeypatch):
    stub = [dict(_LAUREL_MASTER[0], density_g_per_unit=None)]
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: stub)
    sc.invalidate_master_cache()
    try:
        result = sc.aggregate_and_deduct_shopping_list(
            ["3 hojas de laurel"], [], structured=True,
        )
        item = _laurel(result)
        assert item is not None
        assert float(item.get("estimated_cost_rd") or 0) == 150.0, (
            "sin density en master → fallback 0.5 g/hoja, mismo pote único"
        )
    finally:
        sc.invalidate_master_cache()


def test_anchor_in_source():
    from pathlib import Path
    src = (Path(sc.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "P1-LAUREL-LEAF-UNIT" in src
    assert "'laurel' in name.lower()" in src, "pre-paso gated al laurel (no toca hojas de otros alimentos)"
