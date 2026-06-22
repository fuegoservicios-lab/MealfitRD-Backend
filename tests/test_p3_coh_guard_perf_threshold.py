"""[P3-COH-GUARD-PERF-THRESHOLD · 2026-06-22] El umbral del warning de perf del coherence guard
(`MEALFIT_COHERENCE_GUARD_SLOW_MS`) debe ser > baseline normal del guard.

Medición en vivo (3 generaciones 2026-06-22: 1530/4818/1557 ms con 40-45 recetas) mostró que el baseline
normal es ~1.5s, con spikes ocasionales por autoscale del VPS — no por regresión. Con el umbral en 1000ms
el warning "Posible regresión perf — investigar" disparaba en CADA generación (false-positive). Subido a
3000ms: deja pasar el baseline, solo avisa en outliers genuinos. El metric numérico SIEMPRE se persiste.
"""
from __future__ import annotations

from pathlib import Path

_SRC = (Path(__file__).resolve().parent.parent / "shopping_calculator.py").read_text(encoding="utf-8")


def test_slow_threshold_default_raised():
    assert 'MEALFIT_COHERENCE_GUARD_SLOW_MS", "3000"' in _SRC, (
        "el umbral de slow-guard debe ser 3000ms (>baseline ~1.5s) para no gritar lobo en cada generación"
    )
    # Anti-regresión: el default viejo 1000 (false-positive en cada gen) no debe volver.
    assert 'MEALFIT_COHERENCE_GUARD_SLOW_MS", "1000"' not in _SRC
    assert "P3-COH-GUARD-PERF-THRESHOLD" in _SRC


def test_metric_still_emitted_unconditionally():
    # El umbral SOLO modula el warning; el INSERT del metric a pipeline_metrics NO debe gatearse por él.
    assert "_emit_coherence_guard_metric" in _SRC
