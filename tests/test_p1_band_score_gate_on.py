"""[P1-BAND-SCORE-GATE-ON · 2026-06-15] Activación del gate de honestidad por band-score (gap-audit G6).

Bug original (gap-audit 2026-06-15, G6):
    `compute_clinical_band_score` MIDE la precisión de macros por-plan, la persiste y la emite como
    métrica, pero el único mecanismo que la convierte en honestidad user-facing (banner
    _quality_degraded reason=low_band_score) estaba default OFF (`MEALFIT_BAND_SCORE_GATE=False`).
    Un plan con la mitad de las celdas día×macro fuera de banda se entregaba sin señal.

Cierre:
    Activado por default (`BAND_SCORE_GATE_ENABLED=True`) con umbral 0.5, tuneado contra la
    distribución REAL de prod (pipeline_metrics.node='clinical_band', n=152: avg 0.707, p50 0.667,
    p25 0.500) → marca solo la cola genuinamente pobre (~27% no-fallback, todos ≤~0.42).
    `_maybe_mark_low_band_degraded` es puro y fail-safe; corre post-scoring (NO fuerza retry).

Test funcional rápido (sin DB/LLM): importa el módulo y verifica la lógica del gate + el default ON.
"""
from __future__ import annotations

import pytest

import graph_orchestrator as go


def test_gate_enabled_by_default():
    """G6: el gate debe estar ON por default tras tunear el umbral contra prod."""
    assert go.BAND_SCORE_GATE_ENABLED is True, (
        "MEALFIT_BAND_SCORE_GATE debe defaultear True (gap-audit G6): el mecanismo de honestidad "
        "ya estaba cableado, solo faltaba activarlo con un umbral data-informado."
    )
    assert 0.0 < go.BAND_SCORE_GATE_THRESHOLD <= 1.0


def test_low_band_marks_degraded(monkeypatch):
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "BAND_SCORE_GATE_THRESHOLD", 0.5)
    plan: dict = {}
    marked = go._maybe_mark_low_band_degraded(plan, 0.333, False, attempt=1)
    assert marked is True
    assert plan["_quality_degraded"] is True
    assert plan["_quality_degraded_reason"] == "low_band_score"
    assert plan["_quality_degraded_band_score"] == 0.333


def test_high_band_not_marked(monkeypatch):
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "BAND_SCORE_GATE_THRESHOLD", 0.5)
    plan: dict = {}
    assert go._maybe_mark_low_band_degraded(plan, 0.80, False, attempt=1) is False
    assert "_quality_degraded" not in plan


def test_fallback_never_marked(monkeypatch):
    """Los fallbacks matemáticos son band-perfectos por construcción; aunque no lo fueran, el gate
    NO debe marcarlos (la honestidad del fallback la lleva su propio disclaimer)."""
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "BAND_SCORE_GATE_THRESHOLD", 0.5)
    plan: dict = {}
    assert go._maybe_mark_low_band_degraded(plan, 0.10, True, attempt=1) is False
    assert "_quality_degraded" not in plan


def test_does_not_override_worse_reason(monkeypatch):
    """Si el plan ya está marcado por una razón peor (max_attempts/critical), el band-gate NO la pisa."""
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "BAND_SCORE_GATE_THRESHOLD", 0.5)
    plan: dict = {"_quality_degraded": True, "_quality_degraded_reason": "max_attempts"}
    assert go._maybe_mark_low_band_degraded(plan, 0.10, False, attempt=2) is False
    assert plan["_quality_degraded_reason"] == "max_attempts"


def test_disabled_knob_is_noop(monkeypatch):
    """Rollback: con el knob OFF, el gate es no-op aunque el score sea ínfimo."""
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", False)
    plan: dict = {}
    assert go._maybe_mark_low_band_degraded(plan, 0.10, False, attempt=1) is False
    assert plan == {}
