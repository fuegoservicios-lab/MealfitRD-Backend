# -*- coding: utf-8 -*-
"""[P2-TRAILING-QTY-NO-COLON · 2026-09-06] «Comino 1/4 cdta»: la cantidad va al final y sin dos
puntos, y el parser devolvía cantidad CERO.

Es el hallazgo del auditor sobre «un ingrediente sin gramos puede quedar como plantilla válida»,
**verificado en vez de descartado**: existe, y llega por esta vía. Antes del arreglo,
`_split_qty_unit_name("Comino 1/4 cdta")` devolvía `(0.0, 'unidad', 'Comino 1/4 cdta')` — cantidad
cero y un nombre que ningún catálogo resuelve, así que el ingrediente aportaba 0 macros y la lista
de compras no podía pedirlo.

Medido sobre los planes servidos: 255 líneas así de 10.953 (2,3 %), de las que 223 tienen esta
forma exacta y 210 llevan un nombre que sí está en `master_ingredients`. Todas son condimentos y
aromáticos —canela 46, comino 35, tomillo 21, perejil 20, cilantro 18— así que el error de macros
es minúsculo; lo que se perdía de verdad es decirle al usuario que compre el comino.

La maquinaria ya existía: `canonicalize_trailing_qty_line` («Cebada: 50 g» → «50 g de Cebada»,
P2-INGREDIENT-TRAILING-QTY). Solo exigía los dos puntos, y las líneas vivas no los llevan.

El backstop de alérgenos NO estaba afectado y este test lo ancla: lee el texto crudo, no el
nombre parseado, así que caza «Maní 30 g» igual que «30 g de maní». Sin esa comprobación alguien
podría leer este P-fix como si hubiera habido un agujero de seguridad, y no lo hubo.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import nutrition_db as nd  # noqa: E402


# ── la forma que el modelo escribe de verdad ─────────────────────────────────────────────────
@pytest.mark.parametrize("linea, qty, unidad, nombre", [
    ("Comino 1/4 cdta", 0.25, "cdta", "Comino"),
    ("Canela en polvo 1/2 cucharadita", 0.5, "cucharadita", "Canela en polvo"),
    ("Vinagre blanco 1 cda", 1.0, "cda", "Vinagre blanco"),
    ("Orégano dominicano 1/2 cucharadita", 0.5, "cucharadita", "Orégano dominicano"),
    ("Leche 2 tazas", 2.0, "tazas", "Leche"),
    ("Pan integral 2 rebanadas", 2.0, "rebanadas", "Pan integral"),
])
def test_cantidad_al_final_se_resuelve(linea, qty, unidad, nombre):
    q, u, n = nd._split_qty_unit_name(linea)
    assert (q, u, n) == (qty, unidad, nombre), f"{linea!r} sigue sin resolver"


def test_el_de_intermedio(linea="Canela en polvo 1/4 de cucharadita"):
    """«1/4 de cucharadita» — el «de» entre cantidad y unidad es común en español."""
    assert nd._split_qty_unit_name(linea) == (0.25, "cucharadita", "Canela en polvo")


@pytest.mark.parametrize("linea, nombre", [
    ("Perejil 1 cucharada picado", "Perejil picado"),
    ("Cilantro 1 cucharada picada", "Cilantro picada"),
])
def test_el_participio_final_vuelve_al_nombre(linea, nombre):
    """«Perejil 1 cucharada picado»: el participio cierra la línea y pertenece al alimento, no a
    la unidad. Se devuelve al final del nombre en vez de descartarlo."""
    q, u, n = nd._split_qty_unit_name(linea)
    assert (q, u) == (1.0, "cucharada") and n == nombre


# ── lo que NO se debe tocar: el número final es parte del nombre ─────────────────────────────
@pytest.mark.parametrize("linea", [
    "Yogurt griego 0%",       # porcentaje de grasa
    "Omega 3",                # el número ES el nombre
    "Vitamina D 1000 UI",     # 'UI' no es unidad de cocina
    "Sal al gusto",           # sin cantidad, legítimo
    "Canela en polvo",        # sin cantidad
    "Zumo de 1/2 limón",      # la cantidad no está al final
])
def test_lo_que_no_es_una_unidad_queda_intacto(linea):
    """El gate es `canonicalize_unit`: si el último token no es una unidad de cocina, el número
    pertenece al nombre. Sin ese gate, «Yogurt griego 0%» se convertiría en «0 % de Yogurt
    griego» — un ingrediente de cero gramos inventado a partir de una etiqueta."""
    assert nd.canonicalize_trailing_qty_line(linea) == linea


def test_la_forma_con_dos_puntos_sigue_funcionando():
    """P2-INGREDIENT-TRAILING-QTY es quien trajo la maquinaria; este P-fix la amplía, no la
    sustituye."""
    assert nd.canonicalize_trailing_qty_line("Cebada: 50 g") == "50 g de Cebada"
    assert nd._split_qty_unit_name("Cebada: 50 g") == (50.0, "g", "Cebada")


def test_la_forma_canonica_no_se_altera():
    assert nd.canonicalize_trailing_qty_line("50 g de avena") == "50 g de avena"
    assert nd._split_qty_unit_name("50 g de avena") == (50.0, "g", "avena")


def test_entrada_no_string_es_no_op():
    for basura in (None, 42, ["Comino 1/4 cdta"], {"a": 1}):
        assert nd.canonicalize_trailing_qty_line(basura) is basura


# ── el backstop de alérgenos nunca dependió de este parser ───────────────────────────────────
def test_el_backstop_de_alergenos_lee_el_texto_crudo():
    """Los cuatro órdenes posibles se cazan igual. Es la comprobación que baja la severidad de
    este hallazgo de «agujero de seguridad» a «macros y lista de compras»: el backstop escanea la
    línea entera, no el nombre que devuelve `_split_qty_unit_name`."""
    import graph_orchestrator as go

    def _plan(linea):
        return {"days": [{"day": 1, "meals": [
            {"meal": "Merienda", "name": "prueba", "ingredients": [linea],
             "recipe": ["Montaje: sirve."]}]}]}

    for linea in ("2 cucharadas de maní", "Maní 2 cucharadas", "30 g de maní", "Maní 30 g"):
        assert go._scan_allergen_violations(_plan(linea), ["maní"]), (
            f"el backstop dejó pasar {linea!r}")


# ── remate 2026-09-06: la fracción unicode ───────────────────────────────────────────────────
# «Comino 1/4 cdta» resolvía y «Cúrcuma ½ cucharadita» no. La inversión ocurría igual, pero
# `_LEAD_QTY_RE` (nutrition_db.py:136) solo entiende dígitos, así que la línea reescrita seguía
# dando cantidad CERO. Medido: 63 de las 255 líneas se quedaban fuera SOLO por esto.

@pytest.mark.parametrize("linea, qty, unidad, nombre", [
    ("Cúrcuma ½ cucharadita", 0.5, "cucharadita", "Cúrcuma"),
    ("Canela en polvo ¼ cucharadita", 0.25, "cucharadita", "Canela en polvo"),
    ("Comino ½ cdta", 0.5, "cdta", "Comino"),
    ("Canela en polvo ¼ de cucharadita", 0.25, "cucharadita", "Canela en polvo"),
    ("Orégano ⅓ cucharadita", 1.0 / 3.0, "cucharadita", "Orégano"),
])
def test_la_fraccion_unicode_al_final_tambien_resuelve(linea, qty, unidad, nombre):
    q, u, n = nd._split_qty_unit_name(linea)
    assert (u, n) == (unidad, nombre)
    assert abs(q - qty) < 1e-6


def test_la_fraccion_se_emite_en_ascii_no_en_decimal():
    """«1/2» es tan legible como «½» para el usuario y además la entienden los dos parsers.
    Emitirla como «0.5» habría resuelto igual y empeorado el texto."""
    assert nd.canonicalize_trailing_qty_line("Cúrcuma ½ cucharadita") == "1/2 cucharadita de Cúrcuma"


def test_la_cantidad_en_medio_sigue_intacta():
    """«Zumo de ½ limón» no tiene la cantidad al final: no es esta forma y no se toca."""
    assert nd.canonicalize_trailing_qty_line("Zumo de ½ limón") == "Zumo de ½ limón"
