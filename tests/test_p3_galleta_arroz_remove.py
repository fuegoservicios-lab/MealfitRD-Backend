"""[P3-GALLETA-ARROZ-REMOVE · 2026-06-22] Las galletas de arroz se eliminaron del catálogo
(La Sirena no las vende). El swap GF de galletas de trigo ya NO apunta a "Galletas de arroz"
sino a "Casabe" (cracker de yuca GF, en catálogo).

Ancla la asimetría: el target del swap es un GF disponible; las galletas de arroz no se ofrecen.
"""
from __future__ import annotations

from pathlib import Path

from condition_rules import _ALLERGEN_GLUTEN_SUBS

_BACKEND = Path(__file__).resolve().parent.parent


def _find_cracker_rule():
    for tokens, repl, label, _flag in _ALLERGEN_GLUTEN_SUBS:
        if any("galleta" in t for t in tokens):
            return tokens, repl
    return None, None


def test_cracker_gluten_sub_targets_casabe_not_rice_cakes():
    tokens, repl = _find_cracker_rule()
    assert tokens is not None, "no se encontró la regla GF de galletas"
    assert repl == "Casabe", f"el swap GF de galletas debe ir a Casabe, no a {repl!r}"
    assert repl != "Galletas de arroz"


def test_no_gluten_sub_targets_rice_cakes():
    """Ninguna sustitución GF debe apuntar a galletas de arroz (item eliminado)."""
    for _tokens, repl, _label, _flag in _ALLERGEN_GLUTEN_SUBS:
        assert "arroz" not in repl.lower() or repl.lower() == "arroz blanco", \
            f"swap GF apunta a un item de arroz removido: {repl!r}"


def test_marker_present():
    src = (_BACKEND / "condition_rules.py").read_text(encoding="utf-8")
    assert "P3-GALLETA-ARROZ-REMOVE" in src
