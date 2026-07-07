"""[P1-CASABE-HOJA-UNIT · 2026-07-07] "18.67 hojas de casabe → RD$1,755" arreglado.

PDF del owner (plan 5f80f797, review visual 30d): "Casabe: 18.67 hojas RD$1,754.98"
(= RD$94/hoja). Forensic SQL: la fila master de Casabe estaba COMPLETA (paquete 283 g,
RD$94, market_packages [{283,94},{1134,199}], density_g_per_unit=30) — el bug era de
código: "hojas" es count unit y nada la convertía a peso (a diferencia de Ajo
dientes→cabezas, Huevo→cartones, Laurel hojas→gramos), así que el Bloque 1 de envases
(exige weight_in_lbs>0) jamás corría y el costeo caía a count × price_per_unit =
18.67 × RD$94 (el precio del PAQUETE entero cobrado POR HOJA). Pre-paso hojas→gramos →
1-2 paquetes = RD$94-199, como cualquier despensa (paridad con el primer plan que
mostró "1 paquete RD$85" cuando la receta usó "torta").

MISMO patrón que P1-LAUREL-LEAF-UNIT (test_p1_laurel_leaf_unit.py).
"""
import pytest

import shopping_calculator as sc

# Espejo de la fila real de master_ingredients (forensic 2026-07-07).
_CASABE_MASTER = [{
    "name": "Casabe", "category": "Despensa", "default_unit": "paquete",
    "price_per_lb": 0.0, "price_per_unit": 94.0, "density_g_per_unit": 30.0,
    "market_container": "paquete", "container_weight_g": 283.0,
    "available_sizes_g": [283, 1134],
    "market_packages": [
        {"grams": 283, "label": "10 oz", "price": 94},
        {"grams": 1134, "label": "40 oz", "price": 199},
    ],
    "shelf_life_days": 14, "aliases": ["casabe tradicional", "casabe dominicano"],
}]


@pytest.fixture()
def casabe_master(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_CASABE_MASTER))
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


def _casabe(result):
    return next((r for r in result if isinstance(r, dict) and "casabe" in str(r.get("name", "")).lower()), None)


def test_hojas_buy_package_not_per_leaf(casabe_master):
    """2 hojas de casabe → 1 paquete, NO '2 hojas × RD$94'."""
    result = sc.aggregate_and_deduct_shopping_list(
        ["2 hojas de casabe"], [], structured=True,
    )
    item = _casabe(result)
    assert item is not None, f"casabe ausente: {result}"
    disp = str(item.get("display_qty", ""))
    assert "hoja" not in disp.lower(), f"las hojas deben unitarizarse a envase: {disp}"
    assert "paquete" in disp.lower(), f"esperaba '1 paquete (…)': {disp}"
    cost = float(item.get("estimated_cost_rd") or 0)
    assert cost == 94.0, f"1 paquete = RD$94 (NO count×94={2 * 94}): RD${cost}"


def test_pdf_repro_18_67_hojas_not_1755(casabe_master):
    """El caso exacto del PDF: 18.67 hojas costaban RD$1,755. Ahora: precio de paquete(s)."""
    # 18.67 hojas ≈ 9.33 comidas × "2 hojas"; usamos multiplier para escalar.
    result = sc.aggregate_and_deduct_shopping_list(
        ["2 hojas de casabe"], [], structured=True, multiplier=9.335,
    )
    item = _casabe(result)
    assert item is not None
    cost = float(item.get("estimated_cost_rd") or 0)
    assert cost < 300.0, (
        f"18.67 hojas deben costar 1-2 paquetes (RD$94-199), NO count×94=RD$1,755: RD${cost}"
    )
    assert "hoja" not in str(item.get("display_qty", "")).lower()


def test_density_fallback_when_master_missing(monkeypatch):
    """Sin density en master → fallback 15 g/hoja, sigue costeando por paquete (no per-hoja)."""
    stub = [dict(_CASABE_MASTER[0], density_g_per_unit=None)]
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: stub)
    sc.invalidate_master_cache()
    try:
        result = sc.aggregate_and_deduct_shopping_list(
            ["3 hojas de casabe"], [], structured=True,
        )
        item = _casabe(result)
        assert item is not None
        cost = float(item.get("estimated_cost_rd") or 0)
        assert cost < 300.0, f"sin density → fallback 15 g/hoja, sigue por paquete: RD${cost}"
    finally:
        sc.invalidate_master_cache()


def test_anchor_in_source():
    from pathlib import Path
    src = (Path(sc.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "P1-CASABE-HOJA-UNIT" in src
    assert "'casabe' in name.lower()" in src, "pre-paso gated al casabe (no toca hojas de otros alimentos)"
