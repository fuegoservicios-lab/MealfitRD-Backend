"""[P1-COVERAGE-VS-PURCHASE · 2026-07-27] El aviso mandaba recomprar lo que sobra.

## Lo que veía el owner en su lista de 30 días

    Cúrcuma  ¼ lb · alcanza ~19 de 30 días — recompra

Un cuarto de libra son 113 g y el plan necesita 43,6. Le sobra por cuatro y el sistema le dice
que recompre.

## La causa

El aviso se calcula como `capado / necesario`: cuánto de lo que hace falta sobrevive al tope de
perecederos. Correcto en sí mismo, pero **ignora lo que el usuario realmente COMPRA**. El envase
de mercado redondea hacia ARRIBA y muchas veces se pasa de lo necesario:

    Yogurt   necesita 1480.7 g   compra 1960.0 g   "alcanza ~18 de 30 días — recompra"
    Cúrcuma  necesita   43.6 g   compra  113.4 g   "alcanza ~19 de 30 días — recompra"
    Puerro   necesita  207.7 g   compra  300.0 g   "alcanza ~7 de 30 días — recompra"

Medido sobre 6 planes vivos: **6 de 39 avisos (15%)** eran así, y el yogurt salía 3 veces.

## Por qué importa más de lo que parece

Un aviso que manda recomprar lo que sobra no es solo ruido: **enseña al usuario a ignorar los
avisos**, y los otros 33 sí son ciertos. La credibilidad del canal es el activo.

## La regla

Se calcula lo comprado en gramos (envase × cantidad, o la unidad si ya es de peso) y se calla el
aviso solo cuando cubre lo necesario ANTES del tope.

⚠️ Fail-safe: sin datos suficientes para decidir —10 de 39 no los traen— se CONSERVA el aviso.
Nunca peor que el comportamiento previo.

tooltip-anchor: P1-COVERAGE-VS-PURCHASE
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc


# ───────────── 1. los casos reales que mentían ─────────────

@pytest.mark.parametrize("nombre,item,necesita", [
    ("Yogurt",  {"market_qty_numeric": 1.0, "market_unit": "pote", "package_grams": 1960.0}, 1480.7),
    ("Cúrcuma", {"market_qty_numeric": 0.25, "market_unit": "lb", "package_grams": None}, 43.6),
    ("Puerro",  {"market_qty_numeric": 1.0, "market_unit": "paquete", "package_grams": 300.0}, 207.7),
])
def test_cuando_el_envase_ya_cubre_no_se_avisa(nombre, item, necesita):
    assert sc._purchase_covers_need(item, necesita) is True, (
        f"{nombre}: el envase cubre lo necesario y aun así se avisaría de recompra"
    )


def test_la_curcuma_del_owner():
    """¼ lb = 113,4 g para una necesidad de 43,6 g. Sobra por cuatro."""
    assert sc._purchase_covers_need(
        {"market_qty_numeric": 0.25, "market_unit": "lb", "package_grams": None}, 43.6) is True


# ───────────── 2. los avisos LEGÍTIMOS se conservan ─────────────

def test_envase_que_se_queda_corto_sigue_avisando():
    """Comino: pote de 28,35 g para 47 g de necesidad. Aquí el aviso es cierto y debe salir."""
    assert sc._purchase_covers_need(
        {"market_qty_numeric": 1.0, "market_unit": "pote", "package_grams": 28.35}, 47.0) is False


def test_el_yogurt_cuando_de_verdad_falta():
    """Mismo envase de 1.96 kg, pero con 2324.8 g de necesidad: el aviso es correcto."""
    assert sc._purchase_covers_need(
        {"market_qty_numeric": 1.0, "market_unit": "pote", "package_grams": 1960.0}, 2324.8) is False


# ───────────── 3. fail-safe: la duda conserva el aviso ─────────────

@pytest.mark.parametrize("item", [
    {"market_qty_numeric": 1.0, "market_unit": "mazo", "package_grams": None},   # unidad no-peso
    {"market_qty_numeric": 0, "market_unit": "lb", "package_grams": None},        # sin cantidad
    {"market_qty_numeric": "x", "market_unit": "lb", "package_grams": None},      # basura
    {},                                                                            # vacío
])
def test_sin_datos_se_conserva_el_aviso(item):
    assert sc._purchase_covers_need(item, 50.0) is False


@pytest.mark.parametrize("necesita", [0, -5, None, "no-numero"])
def test_necesidad_invalida_conserva_el_aviso(necesita):
    assert sc._purchase_covers_need(
        {"market_qty_numeric": 1.0, "market_unit": "kg", "package_grams": None}, necesita) is False


def test_unidades_de_peso_directas():
    assert sc._purchase_covers_need({"market_qty_numeric": 2.0, "market_unit": "kg"}, 1500.0) is True
    assert sc._purchase_covers_need({"market_qty_numeric": 500.0, "market_unit": "g"}, 400.0) is True
    assert sc._purchase_covers_need({"market_qty_numeric": 1.0, "market_unit": "lbs"}, 500.0) is False


def test_el_envase_manda_sobre_la_unidad():
    """Si hay `package_grams`, es el dato bueno: 'pote' no dice cuánto pesa."""
    assert sc._purchase_covers_need(
        {"market_qty_numeric": 3.0, "market_unit": "pote", "package_grams": 100.0}, 250.0) is True


# ───────────── 4. ancla: el aviso consulta la regla ─────────────

def test_el_aviso_consulta_lo_comprado():
    """Ancla de la clase: si el aviso vuelve a mirar solo `post/pre`, regresa el 15% de avisos
    que mandan recomprar lo que sobra."""
    import inspect
    src = inspect.getsource(sc.apply_smart_market_units)
    assert "_purchase_covers_need(" in src, (
        "el aviso de recompra debe consultar lo COMPRADO, no solo la fracción capada"
    )
    i = src.index("_purchase_covers_need(")
    j = src.index("alcanza ~", i)
    assert i < j, "la comprobación debe ir ANTES de construir el mensaje"
