"""[P3-TOMATE-SAUCE-FIX · 2026-06-22] `canonicalize_tomate` ya NO colapsa las formas
PROCESADAS de tomate (salsa/pasta/puré/ketchup/seco) a "Tomate" fresco.

Bug pre-existente (revelado por test_p6_sauce_substring_tofu_caps): el regex `\btomates?\b`
no estaba anclado y capturaba "salsa de tomate" → "Tomate". En planes reales la salsa de
tomate enlatada se mezclaba y costeaba como tomate de ensalada (disparaba P5-VEG-CAP en vez
de P6-SAUCE-CAP). El docstring decía que se excluían pero el código no lo hacía.
"""
from __future__ import annotations

import pytest

from shopping_calculator import canonicalize_tomate


@pytest.mark.parametrize("name", [
    "salsa de tomate",
    "salsa de tomate organica",
    "salsa de tomate orgánica",
    "pasta de tomate",
    "puré de tomate",
    "pure de tomate",
    "ketchup de tomate",
    "kétchup",
    "tomate seco",
    "tomates secos",
    "tomate deshidratado",
])
def test_processed_tomato_not_canonicalized_to_fresh(name):
    """Formas procesadas → None (conservan su nombre → resuelven a su master item)."""
    assert canonicalize_tomate(name) is None, f"{name!r} no debe colapsar a 'Tomate' fresco"


@pytest.mark.parametrize("name,expected", [
    ("tomate", "Tomate"),
    ("tomates", "Tomate"),
    ("tomate perita", "Tomate"),
    ("tomate criollo", "Tomate"),
    ("tomate maduro", "Tomate"),
    ("tomate cherry", "Tomate cherry"),
    ("tomates cherry", "Tomate cherry"),
    ("tomate uva", "Tomate cherry"),
])
def test_fresh_tomato_still_consolidates(name, expected):
    """Tomate fresco (cualquier variedad) sigue consolidando como antes."""
    assert canonicalize_tomate(name) == expected


def test_non_tomato_returns_none():
    assert canonicalize_tomate("cebolla") is None
    assert canonicalize_tomate("") is None
    assert canonicalize_tomate(None) is None


def test_marker_present():
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.canonicalize_tomate)
    assert "P3-TOMATE-SAUCE-FIX" in src
