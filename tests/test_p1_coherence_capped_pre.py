"""[P1-COHERENCE-CAPPED-PRE · 2026-07-26] El guard reportaba el tope de perecederos como si fuera incoherencia.

## El caso

Tras cerrar el factor 7/3 ([[P1-COHERENCE-DAY-BASIS]]), en el plan vivo `fbe53a5b` — nevera
**vacía**, 0 items, así que `pantry_overdeduct` está descartado — quedaban divergencias que
parecían compras a la baja:

    Yogur    esperado=2324.8  lista=907.2   (39%)
    Cebolla  esperado=1459.5  lista=600.0   (41%)
    Tomate   esperado=1575.0  lista=750.0   (48%)

No lo eran. Cada item lleva su propio registro del tope:

    Yogurt   capped_by=P6-LACTEOS-PERISHABLE-CAP  capped_pre=2324.8  capped_post=907.2
    Cebolla  capped_by=P5-VEG-CAP                 capped_pre=1459.5  capped_post=600.0
    Tomate   capped_by=P5-VEG-CAP                 capped_pre=1575.0  capped_post=750.0
    Lechosa  capped_by=P6-FRUITS-LARGE-CAP        capped_pre=3613.3  capped_post=3000.0

Es un tope **deliberado** —nadie compra 30 días de tomate fresco de una vez— y además se le
comunica al usuario en el propio item: *"1 pote (1.96 kg) · alcanza ~12 de 30 días — recompra"*.

## Lo que confirma de paso

`capped_pre` coincide con el lado esperado **al decimal** en los cuatro. Es una verificación
independiente de que [[P1-COHERENCE-DAY-BASIS]] quedó exacto, desde un campo que el guard no
miraba: el esperado reproduce justo lo que el agregador calculó ANTES de topar.

## La regla

El guard verifica que el agregador calculó BIEN a partir de las recetas. El tope se aplica
después y por diseño, así que la comparación correcta es contra `capped_pre`. **No es un
mute**: si el agregador calcula mal, `capped_pre` diverge y se reporta igual.

## Efecto medido (19 planes vivos, guard REAL en block)

    fbe53a5b:  11 -> 6 divergencias

Y las 6 que quedan son TODAS de la clase sin pareja (condimentos "al gusto" + unidades de
empaque): **cero incoherencias reales de cantidad** en el único plan con el pipeline moderno
completo.

tooltip-anchor: P1-COHERENCE-CAPPED-PRE
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc


def _item(**kw) -> dict:
    base = {"name": "Tomate", "base_qty": 750.0, "base_unit": "g",
            "market_qty_numeric": 1.75, "market_unit": "lbs"}
    base.update(kw)
    return base


# ───────────── 1. el efecto ─────────────

def test_compara_contra_lo_calculado_antes_del_tope():
    out = sc._extract_aggregated_food_dict([
        _item(capped_by="P5-VEG-CAP", capped_pre=1575.0, capped_post=750.0)])
    assert out == {"Tomate": {"g": 1575.0}}, "debe usar capped_pre, no la cantidad topada"


def test_el_tope_deja_de_reportarse_como_divergencia():
    """El caso vivo: recetas 1575 g, lista topada a 750 g. Con el fix, coinciden."""
    expected = {"Tomate": {"g": 1575.0}}
    agg = sc._extract_aggregated_food_dict([
        _item(capped_by="P5-VEG-CAP", capped_pre=1575.0, capped_post=750.0)])
    assert sc.compare_expected_vs_aggregated(expected, agg, tolerance=0.10) == []


@pytest.mark.parametrize("cap,pre,post", [
    ("P6-LACTEOS-PERISHABLE-CAP", 2324.8, 907.2),
    ("P5-VEG-CAP", 1459.5, 600.0),
    ("P6-FRUITS-LARGE-CAP", 3613.3, 3000.0),
])
def test_los_tres_topes_vivos(cap, pre, post):
    out = sc._extract_aggregated_food_dict([
        _item(capped_by=cap, capped_pre=pre, capped_post=post, base_qty=post)])
    assert out["Tomate"]["g"] == pytest.approx(pre)


# ───────────── 2. NO es un mute ─────────────

def test_un_calculo_MALO_del_agregador_sigue_divergiendo():
    """Si el agregador calcula mal desde las recetas, `capped_pre` diverge y se reporta.
    Éste es el test que separa 'excluir un falso positivo' de 'silenciar el guard'."""
    expected = {"Tomate": {"g": 1575.0}}
    agg = sc._extract_aggregated_food_dict([
        _item(capped_by="P5-VEG-CAP", capped_pre=700.0, capped_post=700.0)])
    divs = sc.compare_expected_vs_aggregated(expected, agg, tolerance=0.10)
    assert divs, "un capped_pre equivocado DEBE seguir reportándose"
    assert divs[0]["actual_qty"] == pytest.approx(700.0)


def test_un_item_sin_tope_no_cambia():
    out = sc._extract_aggregated_food_dict([_item()])
    assert out == {"Tomate": {"g": 750.0}}


# ───────────── 3. fail-safe ─────────────

@pytest.mark.parametrize("pre", [None, 0, -5, "no-numero"])
def test_capped_pre_invalido_cae_a_base_qty(pre):
    out = sc._extract_aggregated_food_dict([
        _item(capped_by="P5-VEG-CAP", capped_pre=pre)])
    assert out == {"Tomate": {"g": 750.0}}, f"pre={pre!r} inválido → comportamiento previo"


def test_capped_by_vacio_no_activa_la_rama():
    out = sc._extract_aggregated_food_dict([
        _item(capped_by="", capped_pre=1575.0)])
    assert out == {"Tomate": {"g": 750.0}}


def test_sin_base_unit_no_se_usa_capped_pre():
    """`capped_pre` es un número sin unidad propia: solo sirve si el item declara `base_unit`."""
    out = sc._extract_aggregated_food_dict([
        {"name": "Tomate", "capped_by": "P5-VEG-CAP", "capped_pre": 1575.0,
         "market_qty_numeric": 1.75, "market_unit": "lbs"}])
    assert out == {"Tomate": {"lbs": 1.75}}


def test_listas_legacy_intactas():
    out = sc._extract_aggregated_food_dict([
        {"name": "Miel", "market_qty_numeric": 1.0, "market_unit": "pote"}])
    assert out == {"Miel": {"pote": 1.0}}


# ───────────── 4. el knob ─────────────

def test_el_knob_nace_en_true():
    import os
    os.environ.pop("MEALFIT_COHERENCE_COMPARE_CAPPED_PRE", None)
    assert sc._get_coherence_compare_capped_pre_knob() is True


def test_apagarlo_restaura_la_comparacion_contra_lo_topado(monkeypatch):
    monkeypatch.setenv("MEALFIT_COHERENCE_COMPARE_CAPPED_PRE", "false")
    out = sc._extract_aggregated_food_dict([
        _item(capped_by="P5-VEG-CAP", capped_pre=1575.0, capped_post=750.0)])
    assert out == {"Tomate": {"g": 750.0}}


# ───────────── 5. ancla ─────────────

def test_la_rama_del_tope_va_antes_de_leer_base_qty():
    import inspect
    src = inspect.getsource(sc._extract_aggregated_food_dict)
    i_cap = src.index('item.get("capped_by")')
    i_use = src.index("if isinstance(_bq, (int, float)) and float(_bq) > 0 and _bu:")
    assert i_cap < i_use, "el override del tope debe evaluarse ANTES de consumir _bq"
