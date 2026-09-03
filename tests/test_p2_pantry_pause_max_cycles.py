"""[P2-PANTRY-PAUSE-MAX-CYCLES · 2026-09-03] Un chunk pausado por nevera vacía no resucita para siempre.

Medido (cuenta de prueba sin actividad desde el 5-ago, plan de 15 días): chunks 10/11/12 en
`pending_user_action` con 15/6/0 intentos, 34 pausas/re-encolados en una semana. El recovery los
re-encolaba en flexible cada 12 h y el worker los pausaba otra vez en el mismo tick (0 frescos).
Ahora, al tope (`MEALFIT_PANTRY_PAUSE_MAX_CYCLES`, default 6 ≈ 3 días), el chunk espera la compra
y se reanuda solo cuando la Nevera viva supera `CHUNK_MIN_FRESH_PANTRY_ITEMS` (criterio P0-4, que
hasta hoy sólo cubría `empty_pantry_proactive`).
"""
from __future__ import annotations

from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
CT = (BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


def _ct():
    return pytest.importorskip("cron_tasks")


def test_helper_respects_cap_and_zero_means_no_cap(monkeypatch):
    ct = _ct()
    monkeypatch.setattr(ct, "CHUNK_PANTRY_EMPTY_MAX_CYCLES", 6)
    assert ct._pantry_pause_over_cap(6) is True and ct._pantry_pause_over_cap(15) is True
    assert ct._pantry_pause_over_cap(5) is False and ct._pantry_pause_over_cap(None) is False
    assert ct._pantry_pause_over_cap("x") is False
    monkeypatch.setattr(ct, "CHUNK_PANTRY_EMPTY_MAX_CYCLES", 0)
    assert ct._pantry_pause_over_cap(999) is False


def test_knob_default_and_clamp():
    from constants import CHUNK_PANTRY_EMPTY_MAX_CYCLES
    assert 0 <= CHUNK_PANTRY_EMPTY_MAX_CYCLES <= 100
    src = (BACKEND / "constants.py").read_text(encoding="utf-8")
    assert 'CHUNK_PANTRY_EMPTY_MAX_CYCLES = max(0, min(100, _env_int("MEALFIT_PANTRY_PAUSE_MAX_CYCLES", 6)))' in src


def test_recovery_reads_attempts_and_caps_before_the_flexible_reenqueue():
    """Parser: el SELECT del recovery trae `attempts`; la rama del tope va ANTES del re-encolado
    flexible y reanuda con el mismo criterio de Nevera viva que P0-4."""
    assert "SELECT id, user_id, meal_plan_id, week_number, days_offset, attempts, pipeline_snapshot," in CT
    i_cap = CT.index('if _pantry_pause_over_cap(row.get("attempts")):')
    i_flex = CT.index("Re-encolando en modo flexible.")
    assert i_cap < i_flex
    block = CT[i_cap:i_flex]
    assert "_count_meaningful_pantry_items(_cap_live) >= CHUNK_MIN_FRESH_PANTRY_ITEMS" in block
    assert "_resolve_pantry_pause_markers(resumed_snapshot, \"pantry_restocked\")" in block
    assert "_pantry_pause_cycles_exhausted_at" in block
    # el aviso de agotamiento se escribe UNA vez (marcador en el snapshot), no en cada tick
    assert 'if not snap.get("_pantry_pause_cycles_exhausted_at"):' in block


def test_marker_present():
    assert "P2-PANTRY-PAUSE-MAX-CYCLES" in CT
