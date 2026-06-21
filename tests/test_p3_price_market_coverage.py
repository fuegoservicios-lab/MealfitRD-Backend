"""[P3-PRICE-MARKET-COVERAGE · 2026-06-20] El costo por ítem de la lista de compras
(`estimated_cost_rd`) debe reflejar la cantidad DISPLAY que el usuario compra (el
paquete/cartón/Ud que `apply_smart_market_units` redondeó), no los gramos crudos que
usa la receta.

Bug pre-fix (PDF real, plan 2026-06-20): staples por-peso sub-costeaban — Arroz blanco
'1 paquete (1 lb)' = RD$4 (cobraba ~0.12 lb de receta, no la bolsa); Garbanzos RD$12;
Nueces RD$4; Lechosa '1 Ud' = RD$23 (no la lechosa entera ~66). Y el huevo AL REVÉS:
'medio cartón (15 uds.)' = RD$280 (cobraba cartón completo, debía ~140).

Fix: helper `_cost_from_market` costea sobre `market_qty` × precio-del-market-unit:
(A) libras → market_qty × price_per_lb; (B) envase nombrado → market_qty × price_per_unit
(o convierte envase→lb con container_weight_g si solo hay price_per_lb); (C) Ud./Cabeza →
market_qty × price_per_unit; (D) huevo 'cartón (N uds.)' → (market_qty × N) × (price_per_unit/30).

Datos reales verificados en master_ingredients (prod Neon): huevo ud=280 cont=900 dens=50;
arroz-blanco lb=32.70 cont=907; garbanzos lb=69.74 cont=453; camarones lb=299 (sin container);
lechosa lb=20 ud=66.14; tofu lb=168.38 ud=147 cont=396; melón lb=21.77 ud=72.

[P3-EGG-REAL-CARTONS · 2026-06-20] El pre-processing del huevo (aggregate_and_deduct)
ahora SOLO crea 'cartón (30 uds.)' redondeando hacia arriba — en el mercado DR no existen
cartones de 6 ni 15 (confirmado por el owner con fotos de la tienda; los huevos se compran
por cartón completo). El helper sigue siendo robusto a cualquier 'cartón (N uds.)' (defensa),
pero producción solo verá cartones de 30; precio actual del huevo = 295/cartón → completo 295.
Los tests de 6/15 abajo verifican la ROBUSTEZ de la fórmula del helper, no buckets que
producción cree (ver test_preprocessing_huevo_solo_crea_cartones_de_30).
"""
from __future__ import annotations

import pytest

from shopping_calculator import _cost_from_market


def _mk(qty, unit):
    return {"market_qty": qty, "market_unit": unit}


# ── (D) HUEVO — la corrección estrella (medio cartón debe costear medio, no cartón completo)
def test_huevo_medio_carton_costea_15_huevos_no_carton_completo():
    # 'medio cartón (15 uds.)' × 1; price_per_unit=280 por cartón de 30 → 15 × (280/30) = 140
    cost = _cost_from_market(_mk(1, "medio cartón (15 uds.)"),
                             {"container_weight_g": 900, "density_g_per_unit": 50}, 0, 280)
    assert cost == pytest.approx(140.0), f"huevo medio cartón debe ser 140, no {cost}"


def test_huevo_carton_completo_x2():
    # 'cartón (30 uds.)' × 2 → 60 huevos × (280/30) = 560
    assert _cost_from_market(_mk(2, "cartón (30 uds.)"), {}, 0, 280) == pytest.approx(560.0)


def test_huevo_carton_6():
    assert _cost_from_market(_mk(1, "cartón (6 uds.)"), {}, 0, 280) == pytest.approx(56.0)


# ── (B) STAPLES POR-PESO — la otra mitad: costear la BOLSA, no los gramos
def test_arroz_blanco_paquete_costea_la_bolsa():
    # container 907g (≈2 lb), price_per_lb=32.7 → 907/453.592 × 32.7 ≈ 65.4 (no RD$4)
    cost = _cost_from_market(_mk(1, "paquete"), {"container_weight_g": 907}, 32.7, 0)
    assert cost == pytest.approx(65.4, abs=0.6), f"arroz 1 paquete(907g) debe ser ~65, no {cost}"


def test_garbanzos_paquete_costea_la_bolsa():
    cost = _cost_from_market(_mk(1, "paquete"), {"container_weight_g": 453}, 69.74, 0)
    assert cost == pytest.approx(69.65, abs=0.6)


# ── (C) UNIDAD NATURAL — lechosa/melón enteros
def test_lechosa_ud_costea_la_unidad_entera():
    cost = _cost_from_market(_mk(1, "Ud."), {"density_g_per_unit": 1500}, 20, 66.14)
    assert cost == pytest.approx(66.14)


def test_melon_2ud():
    assert _cost_from_market(_mk(2, "Ud."), {}, 21.77, 72) == pytest.approx(144.0)


# ── (A) LIBRAS — NO-REGRESIÓN: camarones ya costeaba bien
def test_camarones_lb_no_regresion():
    # 2 lbs × price_per_lb=299 = 598 (el screenshot mostraba 584 con precio viejo 292)
    assert _cost_from_market(_mk(2, "lbs"), {}, 299, 0) == pytest.approx(598.0)


def test_lb_fraccional():
    # 1/4 lb maní × price_per_lb=279.72 = 69.93
    assert _cost_from_market(_mk(0.25, "lb"), {}, 279.72, 0) == pytest.approx(69.93, abs=0.1)


# ── (B) ENVASE NOMBRADO con price_per_unit — NO-REGRESIÓN: tofu/aceite ya costeaban bien
def test_tofu_paquete_por_unidad_no_regresion():
    # price_per_unit=147 toma precedencia sobre price_per_lb → 2 × 147 = 294
    assert _cost_from_market(_mk(2, "paquete"), {"container_weight_g": 396}, 168.38, 147) == pytest.approx(294.0)


def test_aceite_botella_no_regresion():
    assert _cost_from_market(_mk(1, "botella"), {}, 0, 195) == pytest.approx(195.0)


# ── EDGE CASES
def test_sin_precio_devuelve_cero():
    assert _cost_from_market(_mk(1, "paquete"), {}, 0, 0) == 0.0


def test_market_qty_cero_devuelve_cero():
    assert _cost_from_market(_mk(0, "lbs"), {}, 299, 0) == 0.0


def test_al_gusto_devuelve_cero():
    assert _cost_from_market(_mk(1, "al gusto"), {}, 0, 0) == 0.0


def test_market_qty_no_numerico_no_crashea():
    assert _cost_from_market(_mk("xx", "lbs"), {}, 299, 0) == 0.0


def test_master_item_none_no_crashea():
    assert _cost_from_market(_mk(2, "lbs"), None, 299, 0) == pytest.approx(598.0)


def test_envase_solo_price_per_lb_sin_container_devuelve_cero():
    # envase nombrado, solo price_per_lb, SIN container_weight_g → no se puede convertir → 0 (honesto)
    assert _cost_from_market(_mk(1, "paquete"), {}, 50.0, 0) == 0.0


# ── [P3-EGG-REAL-CARTONS · 2026-06-20] El pre-processing solo crea cartones REALES de 30
def test_huevo_carton_30_precio_actual_295():
    # precio actual en prod = 295/cartón → cartón completo (30 uds.) = 295
    assert _cost_from_market(_mk(1, "cartón (30 uds.)"), {}, 0, 295) == pytest.approx(295.0)


def test_huevo_dos_cartones_30_precio_actual():
    # 2 cartones de 30 (61+ huevos) → 2 × 295 = 590
    assert _cost_from_market(_mk(2, "cartón (30 uds.)"), {}, 0, 295) == pytest.approx(590.0)


def test_preprocessing_huevo_solo_crea_cartones_de_30():
    """En el mercado DR no existen cartones de 6 ni 15; el huevo se compra por cartón
    completo. El pre-processing del huevo (aggregate_and_deduct) debe redondear HACIA
    ARRIBA a 'cartón (30 uds.)' y NUNCA crear los buckets falsos 'medio cartón (15 uds.)'
    ni 'cartón (6 uds.)'. Ancla la corrección reportada por el owner con fotos de la tienda."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "shopping_calculator.py").read_text(encoding="utf-8")
    idx = src.index("P3-EGG-REAL-CARTONS")
    block = src[idx:idx + 650]  # cubre el comentario + la línea de código del bucket
    assert "cartón (30 uds.)" in block, "el pre-processing debe crear 'cartón (30 uds.)'"
    assert "math.ceil" in block, "debe redondear HACIA ARRIBA (ceil) a cartones completos"
    assert "medio cartón (15 uds.)" not in block, "no debe crear el bucket falso de 15 uds."
    assert "cartón (6 uds.)" not in block, "no debe crear el bucket falso de 6 uds."
