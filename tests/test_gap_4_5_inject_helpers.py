"""[S10-1 / S10-2 · GAP-4/GAP-5 · 2026-05-29] Unit tests de los helpers SSOT extraídos
de `_inject_advanced_learning_signals` (cron path) e `inject_learning_signals_from_profile`
(API path).

Pre-fix, la lógica de ambos vivía copy-pasteada en los dos paths; un cambio de threshold
en uno divergía silenciosamente para planes cron-generados vs manuales. Estos tests pinan
el comportamiento de los helpers para que esa clase de drift falle aquí.

Tooltip-anchor: GAP-4-5-INJECT-HELPERS.
"""
from __future__ import annotations

import cron_tasks


# ---------------------------------------------------------------------------
# _quality_trend_hint (S10-2): clasificación trend→hint con prioridad
# ---------------------------------------------------------------------------

def test_trend_drastic_change_on_three_low():
    assert cron_tasks._quality_trend_hint([0.2, 0.1, 0.25], []) == 'drastic_change'


def test_trend_increase_complexity_on_three_high():
    assert cron_tasks._quality_trend_hint([0.9, 0.85, 0.95], []) == 'increase_complexity'


def test_trend_break_plateau_low_variance_low_mean():
    # 4+ valores, varianza < 0.01, media < 0.6, sin 3 consecutivos <0.3 ni >0.8.
    assert cron_tasks._quality_trend_hint([0.55, 0.56, 0.54, 0.55], []) == 'break_plateau'


def test_trend_simplify_urgently_on_dropping_adherence():
    # quality sin señal; adherencia cae consistentemente y termina < 0.65.
    assert cron_tasks._quality_trend_hint([0.6, 0.7], [0.8, 0.72, 0.6]) == 'simplify_urgently'


def test_trend_none_when_no_signal():
    assert cron_tasks._quality_trend_hint([0.6, 0.65], [0.9, 0.9, 0.9]) is None


def test_trend_priority_drastic_over_plateau():
    # 4 valores todos < 0.3 → drastic_change gana sobre break_plateau (prioridad).
    assert cron_tasks._quality_trend_hint([0.1, 0.1, 0.1, 0.1], []) == 'drastic_change'


def test_trend_handles_non_list_inputs():
    assert cron_tasks._quality_trend_hint(None, None) is None
    assert cron_tasks._quality_trend_hint("garbage", 5) is None


# ---------------------------------------------------------------------------
# _should_auto_activate_adversarial (S10-1)
# ---------------------------------------------------------------------------

def test_adversarial_quality_low_sustained():
    activate, reasons = cron_tasks._should_auto_activate_adversarial(
        {"quality_history_chunks": [0.4, 0.3]}
    )
    assert activate is True
    assert "quality_low_sustained" in reasons


def test_adversarial_high_variance():
    activate, reasons = cron_tasks._should_auto_activate_adversarial(
        {"attribution_tracker": {"a": {"avg_score": 0.1}, "b": {"avg_score": 0.9}, "c": {"avg_score": 0.5}}}
    )
    assert activate is True
    assert any("attribution_high_variance" in r for r in reasons)


def test_adversarial_frequent_rejections():
    activate, reasons = cron_tasks._should_auto_activate_adversarial(
        {"rejection_patterns": [1, 2, 3, 4, 5]}
    )
    assert activate is True
    assert any("frequent_rejections" in r for r in reasons)


def test_adversarial_none_on_healthy_profile():
    activate, reasons = cron_tasks._should_auto_activate_adversarial(
        {"quality_history_chunks": [0.9, 0.85], "attribution_tracker": {}, "rejection_patterns": []}
    )
    assert activate is False
    assert reasons == []


def test_adversarial_tolerates_garbage():
    activate, reasons = cron_tasks._should_auto_activate_adversarial({})
    assert activate is False
