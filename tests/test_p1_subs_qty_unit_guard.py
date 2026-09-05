# -*- coding: utf-8 -*-
"""[P1-SUBS-QTY-UNIT-GUARD · 2026-09-05] La sustitución preservaba la «cantidad» copiando el número Y la palabra
que lo seguía, fuera lo que fuera. Con un alimento CONTADO —«4 ciruelas», sin «de» de por medio— esa palabra es el
alimento, no una unidad, y el resultado fue el ingrediente imposible «4 ciruelas Soya texturizada» (plan vivo
18326457, merienda del día 3, 2026-09-05). Aquí se ancla que el prefijo solo sobrevive si es una unidad real."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import graph_orchestrator as go  # noqa: E402


def _sub(orig, replacement="Soya texturizada"):
    """Ejecuta la sustitución real sobre un plan de una comida y devuelve el ingrediente resultante."""
    plan = {"days": [{"meals": [{"name": "Vasito", "ingredients": [orig], "ingredients_raw": [orig]}]}]}
    subs = [{"tokens": ["jamon", "jamón", "pollo"], "replacement": replacement, "label": "carne",
             "negatives": [], "condition": "vegetarian", "preserve_qty": True}]
    go._apply_substitutions_core(plan, subs, lambda u, c: "nota", "Ajuste", lambda m, u, c: None)
    return plan["days"][0]["meals"][0]["ingredients"][0]


@pytest.mark.parametrize("token, es_unidad", [
    ("g", True), ("gramos", True), ("taza", True), ("cucharadas", True), ("lonjas", True),
    ("lb", True), ("onzas", True), ("unidades", True), ("porción", True), ("puñado", True),
    ("", True),                       # sin palabra tras el número: no hay alimento que confundir
    ("ciruelas", False), ("huevos", False), ("manzanas", False), ("pechuga", False),
    ("ciruela", False), ("aguacates", False),
])
def test_unit_whitelist(token, es_unidad):
    assert go._sub_qty_token_is_unit(token) is es_unidad


def test_alimento_contado_no_hereda_su_cantidad():
    """El caso vivo: la cantidad pertenecía a las ciruelas, así que NO puede viajar al reemplazo."""
    out = _sub("4 ciruelas rellenas de jamón")
    assert "ciruela" not in out.lower(), f"el alimento viejo sobrevivió en «{out}»"
    assert out.strip() == "Soya texturizada", out


def test_unidad_de_verdad_si_conserva_la_cantidad():
    """La razón de ser de preserve_qty: la lista de compras necesita el peso comprable."""
    assert _sub("120 g de jamón") == "120 g de Soya texturizada"
    assert _sub("2 lonjas de jamón") == "2 lonjas de Soya texturizada"


def test_la_regex_sigue_exponiendo_el_prefijo_completo():
    """Contrato con test_p3_condition_subs_fix: group(1) es el prefijo, group(2) la palabra a juzgar."""
    m = go._COND_SUB_QTY_PREFIX_RE.match("100g de longaniza")
    assert m.group(1).strip() == "100g de" and m.group(2) == "g"
    m2 = go._COND_SUB_QTY_PREFIX_RE.match("4 ciruelas")
    assert m2.group(2) == "ciruelas"
    assert go._COND_SUB_QTY_PREFIX_RE.match("Soya/Tofu") is None
