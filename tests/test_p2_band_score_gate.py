"""[P2-BAND-SCORE-GATE · 2026-06-15] Gate por-plan sobre el clinical_band_score.

Hoy un plan con la mitad de las celdas día×macro fuera de banda (band_score bajo) se entrega IDÉNTICO
a uno preciso, sin aviso (audit P2-14). `_maybe_mark_low_band_degraded` marca el plan como
_quality_degraded (reason=low_band_score) cuando la precisión medida cae bajo el umbral → dispara el
banner de degradación YA EXISTENTE en el frontend. NO fuerza retry. Default OFF (user-facing, opt-in).

Validación determinista del helper puro (sin LLM/DB/créditos).
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


@pytest.fixture
def gate_on(go, monkeypatch):
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "BAND_SCORE_GATE_THRESHOLD", 0.5)
    return go


def test_default_knob_off(go):
    assert go.BAND_SCORE_GATE_ENABLED is False, "default OFF (validate-first, user-facing)"


def test_off_does_not_mark(go, monkeypatch):
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", False)
    plan = {}
    assert go._maybe_mark_low_band_degraded(plan, 0.2, False, 1) is False
    assert "_quality_degraded" not in plan


def test_low_band_marks_degraded(gate_on):
    plan = {}
    assert gate_on._maybe_mark_low_band_degraded(plan, 0.30, False, 2) is True
    assert plan["_quality_degraded"] is True
    assert plan["_quality_degraded_reason"] == "low_band_score"
    assert plan["_quality_degraded_band_score"] == 0.30
    assert plan["_quality_degraded_attempts"] == 2


def test_high_band_not_marked(gate_on):
    plan = {}
    assert gate_on._maybe_mark_low_band_degraded(plan, 0.85, False, 1) is False
    assert "_quality_degraded" not in plan


def test_fallback_not_marked(gate_on):
    """Un fallback ya se trata aparte (holistic clamp); no lo re-marcamos por band score."""
    plan = {}
    assert gate_on._maybe_mark_low_band_degraded(plan, 0.10, True, 1) is False
    assert "_quality_degraded" not in plan


def test_does_not_override_existing_worse_reason(gate_on):
    plan = {"_quality_degraded": True, "_quality_degraded_reason": "max_attempts"}
    assert gate_on._maybe_mark_low_band_degraded(plan, 0.10, False, 3) is False
    assert plan["_quality_degraded_reason"] == "max_attempts"  # no se pisa


def test_none_band_val(gate_on):
    plan = {}
    assert gate_on._maybe_mark_low_band_degraded(plan, None, False, 1) is False


def test_marker_and_helper_present(go):
    from pathlib import Path
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert "P2-BAND-SCORE-GATE" in src
    assert "def _maybe_mark_low_band_degraded(" in src
