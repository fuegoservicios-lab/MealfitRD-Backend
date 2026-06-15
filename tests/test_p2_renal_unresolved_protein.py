"""[P2-RENAL-UNRESOLVED-PROTEIN · 2026-06-15] Señal observable: cap renal incompleto sobre proteína no-resuelta.

El cap renal recorta el NÚMERO de proteína + ingredientes RESUELTOS, pero un plato proteico COMPUESTO
no-resuelto (sancocho/mondongo) deja su proteína FÍSICA intacta → el cap es incompleto para ese plan.
La señal lo marca observable (telemetría: `_clinical_layer_incomplete` + `renal_protein_cap.cap_complete=False`
+ `unresolved_protein_items`), NO escalado ciego (encogería vegetales). Define "no-resuelto" vía `db.lookup`
(NO macros, que también es None por cantidad 'al gusto' → falso positivo).

Validación determinista con stub (sin LLM/créditos).
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


class _StubDB:
    """lookup por nombre: pollo/res/arroz resuelven; sancocho/mondongo/yuca NO."""
    def lookup(self, name):
        n = str(name).lower()
        if any(k in n for k in ("pollo", "arroz", "res molida")):
            return object()  # resuelve por nombre
        return None


# ── _name_matches_protein_hint ──
@pytest.mark.parametrize("s,expected", [
    ("sancocho de res", True), ("mondongo", False), ("200g de pollo guisado", True),
    ("1 taza de arroz", False), ("yuca hervida", False), ("queso de freir", True),
])
def test_name_matches_protein_hint(go, s, expected):
    assert go._name_matches_protein_hint(s) is expected


# ── _ingredient_is_unresolved_protein (lookup-based, NO macros) ──
def test_unresolved_protein_dish_flagged(go):
    # 'sancocho de res' no resuelve por nombre + 'res' es hint de proteína → True.
    assert go._ingredient_is_unresolved_protein("sancocho de res", _StubDB()) is True


def test_resolved_protein_not_flagged(go):
    # 'pollo' resuelve por nombre → el cap lo cubre → False.
    assert go._ingredient_is_unresolved_protein("200g de pollo", _StubDB()) is False


def test_resolved_protein_with_unconvertible_qty_not_flagged(go):
    """Correción del verificador: 'pollo al gusto' resuelve por NOMBRE (lookup) aunque su cantidad no
    sea convertible → NO falso positivo (no se basa en macros, que sería None por la cantidad)."""
    assert go._ingredient_is_unresolved_protein("pollo al gusto", _StubDB()) is False


def test_unresolved_non_protein_not_flagged(go):
    # 'yuca hervida' no resuelve pero NO es proteína → no dispara (anti-falso-positivo).
    assert go._ingredient_is_unresolved_protein("yuca hervida", _StubDB()) is False


def test_failsafe_no_db(go):
    assert go._ingredient_is_unresolved_protein("sancocho de res", None) is False


# ── Knob default ON (telemetría segura) + anchors ──
def test_signal_knob_default_on(go):
    assert go.RENAL_UNRESOLVED_PROTEIN_SIGNAL is True  # telemetría pura, sin mutación


def test_marker_and_signal_present(go):
    from pathlib import Path
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert "P2-RENAL-UNRESOLVED-PROTEIN" in src
    assert "def _ingredient_is_unresolved_protein(" in src
    # La señal NO pisa meals_enforced (usa cap_complete) — corrección del verificador.
    assert 'cap_complete' in src
    assert "unresolved_protein_items" in src
