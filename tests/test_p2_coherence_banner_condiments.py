"""[P2-COHERENCE-BANNER-CONDIMENTS · 2026-09-03] El banner «Lista revisada — N ítems pueden
necesitar ajuste manual» mostraba «Ajo (Compra menor que la receta, 76 %)». El ajo es un
condimento: su compra se acota a propósito (topes de condimentos/hierbas) y va por envase, así
que no es una compra que el usuario deba ajustar a mano. El banner omite condimentos (SSOT
`constants.is_allowed_condiment`); la telemetría del guard conserva todas las divergencias.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from constants import is_allowed_condiment
from shopping_calculator import summarize_divergences_for_ui

_BACKEND = Path(__file__).resolve().parents[1]


def _div(food, hypothesis="magnitude_undersupply", delta=0.76):
    return {"food": food, "hypothesis": hypothesis, "side": "left", "magnitude": True, "delta_pct": delta}


def test_helper_reconoce_condimentos_sin_depender_de_acentos_ni_mayusculas():
    for n in ("Ajo", "ajo", "Limón", "Orégano", "Cilantro", "Sal", "dientes de ajo"):
        assert is_allowed_condiment(n), n
    for n in ("Pollo", "Pechuga de pollo", "Sardinas en lata", "Batata", "", None):
        assert not is_allowed_condiment(n), n


def test_el_banner_omite_condimentos_aunque_sean_accionables(monkeypatch):
    monkeypatch.delenv("MEALFIT_COHERENCE_BANNER_ACTIONABLE_ONLY", raising=False)
    out = summarize_divergences_for_ui([_div("Ajo"), _div("Cilantro", "cap_swallowed_modifier"), _div("Pollo")])
    assert [o["food"] for o in out] == ["Pollo"]


def test_el_knob_de_rollback_devuelve_todo(monkeypatch):
    monkeypatch.setenv("MEALFIT_COHERENCE_BANNER_ACTIONABLE_ONLY", "0")
    out = summarize_divergences_for_ui([_div("Ajo"), _div("Pollo")])
    assert [o["food"] for o in out] == ["Ajo", "Pollo"]


def test_anclas_en_el_codigo():
    sc = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "Tooltip-anchor: P2-COHERENCE-BANNER-CONDIMENTS" in sc
    assert re.search(r"if _actionable_only and _is_condiment\(d\.get\(\"food\"\) or d\.get\(\"name\"\) or \"\"\):", sc)
    assert 'from constants import is_allowed_condiment as _is_condiment' in sc
    assert '_LAST_KNOWN_PFIX = "P2-COHERENCE-BANNER-CONDIMENTS · 2026-09-03"' in (_BACKEND / "app.py").read_text(encoding="utf-8")


def test_el_ssot_de_condimentos_sigue_siendo_uno():
    """No debe nacer una segunda lista de condimentos para el banner (lección P1-DIET-CANON-SSOT)."""
    sc = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    i = sc.index("def summarize_divergences_for_ui")
    body = sc[i:i + 6000]
    assert "_ALLOWED_CONDIMENTS = (" not in body
    assert '"ajo",' not in body  # la tupla vive en constants.py, no aquí
