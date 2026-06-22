"""[P3-COH-GUARD-PERF-THRESHOLD · 2026-06-22] El umbral del warning de perf del coherence guard
(`MEALFIT_COHERENCE_GUARD_SLOW_MS`) debe ser > baseline normal del guard.

Medición en vivo 2026-06-22: la distribución de pipeline_metrics reveló que el guard de PLAN COMPLETO
(47 recetas × 47 ingredientes + ~70 divergencias) cuesta ~3-3.8s de forma CONSISTENTE (9 calls: p50=3027,
p90=3676, max=3844) — costo de canonicalización O(n²), no regresión. El umbral pasó 1000→3000→5000ms: con
3000 el warning disparaba en CADA recálculo de plan completo (baseline ~3s); 5000 queda por encima del p90
observado (3676) y solo avisa regresiones reales (>5s). El metric numérico SIEMPRE se persiste.
"""
from __future__ import annotations

from pathlib import Path

_SRC = (Path(__file__).resolve().parent.parent / "shopping_calculator.py").read_text(encoding="utf-8")


def test_slow_threshold_default_raised():
    assert 'MEALFIT_COHERENCE_GUARD_SLOW_MS", "5000"' in _SRC, (
        "el umbral de slow-guard debe ser 5000ms (>p90 ~3.7s del guard de plan completo) para no gritar "
        "lobo en cada recálculo normal"
    )
    # Anti-regresión: los defaults viejos (false-positives en cada recálculo) no deben volver.
    assert 'MEALFIT_COHERENCE_GUARD_SLOW_MS", "1000"' not in _SRC
    assert 'MEALFIT_COHERENCE_GUARD_SLOW_MS", "3000"' not in _SRC
    assert "P3-COH-GUARD-PERF-THRESHOLD" in _SRC


def test_metric_still_emitted_unconditionally():
    # El umbral SOLO modula el warning; el INSERT del metric a pipeline_metrics NO debe gatearse por él.
    assert "_emit_coherence_guard_metric" in _SRC
