"""[P1-CYCLE-REPURCHASE-HONEST · 2026-07-25] El ciclo cobraba 5 fundas de manzanas para 5 manzanas.

"Costo real del ciclo de 30 Días" multiplicaba **todos** los perecederos por las semanas del ciclo
(×4,286 en mensual), como si cada uno se re-comprara en cada ida al súper. Pero el envase mínimo
del mercado dominicano a veces cubre el mes entero: la funda de manzanas más pequeña es de 3 lb y
el plan usa ~1 manzana por semana.

Medido en el plan vivo `1d3c6643` (sobre-oferta del envase respecto a la necesidad):

    Manzana        6,5×      Mozzarella   4,9×
    Avena          4,3×      Queso ricotta 3,7×

El número que veía el owner (RD$14.266 de ciclo) incluía cuatro re-compras que nunca ocurren.

El límite lo pone lo que pase primero: que se acabe **o que se dañe**. Una funda de 3 lb de
manzanas cubre el mes en consumo pero no aguanta 30 días en la nevera, así que `shelf_life_days`
recorta la cobertura. Sin ese tope, el arreglo sub-declararía el costo — el error opuesto.
"""
import math

import pytest

import shopping_calculator as sc


def _item(cost, *, perishable=True, ratio=None, shelf=None):
    it = {"name": "X", "estimated_cost_rd": cost, "is_perishable": perishable}
    if ratio is not None:
        it["pkg_cover_ratio"] = ratio
    if shelf is not None:
        it["shelf_life_days"] = shelf
    return it


# ───────────── 1. el caso medido ─────────────

def test_la_funda_de_manzanas_no_se_compra_cinco_veces():
    """3 lb para ~1 manzana/semana: cubre 6,5 idas, y aguanta (shelf 30d)."""
    n = sc._item_cycle_repurchases(_item(265, ratio=6.5, shelf=30), cycle_days=30)
    assert n == pytest.approx(1.0), n


def test_lo_que_se_consume_en_una_semana_SI_se_recompra_cada_semana():
    """La costilla, la leche, la lechuga: el envase cubre justo la ida."""
    n = sc._item_cycle_repurchases(_item(95, ratio=1.0, shelf=5), cycle_days=30)
    assert n == pytest.approx(30 / 7, rel=0.01), n


def test_la_vida_util_recorta_la_cobertura():
    """Un envase que cubre el mes en CONSUMO pero se daña en 10 días se compra 3 veces."""
    n = sc._item_cycle_repurchases(_item(100, ratio=6.0, shelf=10), cycle_days=30)
    assert n == pytest.approx(3.0, rel=0.01), n


def test_jamas_sube_el_costo():
    """Este pase sólo puede BAJAR lo declarado; no puede inventar compras."""
    plano = 30 / 7
    for ratio in (0.1, 0.5, 1.0, 2.0, 12.0):
        for shelf in (1, 3, 7, 30, 400):
            n = sc._item_cycle_repurchases(_item(10, ratio=ratio, shelf=shelf), cycle_days=30)
            assert 1.0 <= n <= plano + 1e-9, (ratio, shelf, n)


def test_sin_senal_conserva_el_comportamiento_previo():
    """Ítems viejos o sin envase resuelto: no adivinar."""
    assert sc._item_cycle_repurchases(_item(50), cycle_days=30) == pytest.approx(30 / 7, rel=0.01)
    assert sc._item_cycle_repurchases({}, cycle_days=30) == pytest.approx(30 / 7, rel=0.01)


def test_knob_de_rollback(monkeypatch):
    monkeypatch.setattr(sc, "CYCLE_REPURCHASE_HONEST", False)
    n = sc._item_cycle_repurchases(_item(265, ratio=6.5, shelf=30), cycle_days=30)
    assert n == pytest.approx(30 / 7, rel=0.01), "con el knob OFF debe volver al plano"


# ───────────── 2. el total del ciclo ─────────────

def test_el_total_baja_y_reporta_cuanto():
    items = [_item(265, ratio=6.5, shelf=30),      # manzanas: 1 compra
             _item(95, ratio=1.0, shelf=5),        # costilla: 4,29 compras
             _item(2300, perishable=False)]        # despensa: no entra acá
    total, ahorro = sc._perishable_cycle_cost(items, cycle_days=30, weeks_flat=30 / 7)
    assert total == pytest.approx(265 * 1 + 95 * 30 / 7, rel=0.01)
    assert ahorro == pytest.approx(265 * (30 / 7 - 1), rel=0.01)


def test_los_estables_no_se_tocan():
    """La despensa ya se cuenta 1× aparte; contarla aquí la duplicaría."""
    total, _ = sc._perishable_cycle_cost([_item(999, perishable=False, ratio=1.0)],
                                         cycle_days=30, weeks_flat=30 / 7)
    assert total == 0.0


def test_items_sin_precio_no_suman():
    for malo in (None, "", "abc", float("nan"), float("inf"), 0, -5):
        total, _ = sc._perishable_cycle_cost([_item(malo, ratio=1.0)], cycle_days=30, weeks_flat=4.286)
        assert total == 0.0, malo


# ───────────── 3. el ratio nace donde conviven necesidad y envase ─────────────

def test_apply_smart_market_units_expone_el_ratio():
    from pathlib import Path
    src = (Path(sc.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")
    i_fn = src.index("def apply_smart_market_units(")
    # `"def "` a secas casa dentro del docstring; el siguiente def de MÓDULO empieza en columna 0.
    i_ret = src.index("\ndef ", i_fn + 10)
    cuerpo = src[i_fn:i_ret]
    assert 'result["pkg_cover_ratio"]' in cuerpo, \
        "el ratio debe calcularse donde aún se conoce la necesidad (weight_in_lbs)"
    assert "weight_in_lbs" in cuerpo
