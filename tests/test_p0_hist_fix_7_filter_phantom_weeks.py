"""[P0-HIST-FIX-7 · 2026-05-09] Cross-link backend del filter de
chunks fuera del alcance del plan en el tab Métricas.

Bug reportado en producción 2026-05-09:
    Plan "Plan Sintético 7 días" — 1 semana = 7 días. La queue tiene
    chunks fantasma para `week_number = 2` (rolling refill que el
    cron creó ahead-of-time pero NO pertenece a este plan corto).
    El tab Métricas mostraba el chunk de week 2 pendiente, confundiendo
    al usuario que reportó "la semana 2 no debe ni existir".

Fix (frontend-only):
    Filter en el render del tab Métricas:
      _maxValidWeek = ceil(_displayTotal / 7)
      _list = _rawList.filter(c => c.week_number <= _maxValidWeek)

    Para plan healthy de 7 días: max week 1 → filtra week 2+.
    Para plan de 14 días: max 2 → filtra week 3+.
    Chunks sin week_number numérico se preservan (legacy/edge).

    También ajusta el truncated notice: `_adjustedTotal = _totalCount
    - _filteredOutCount` para que "Mostrando X de Y" sea coherente
    con el alcance real del plan.

Cobertura backend (cross-link del marker):
    1. Anchor del marker en History.jsx.
    2. Endpoint /chunk-metrics expone `week_number` (input del filter).
    3. Endpoint /history-list expone `total_days_requested` y
       `totalDays` legacy (inputs del cómputo de _maxValidWeek).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"


def test_marker_present_in_history_jsx():
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-7" in text


def test_history_jsx_computes_max_valid_week():
    """`_maxValidWeek = Math.ceil(_displayTotal / 7)` — 1 semana
    para plan ≤7 días, 2 para 8-14, etc."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"_maxValidWeek\s*=\s*[\s\S]{0,200}?Math\.ceil\([^)]+\/\s*7\s*\)",
        text,
    ), "Frontend debe computar _maxValidWeek = ceil(N/7)."


def test_history_jsx_filters_chunks_by_week_number():
    """El filter `c.week_number <= _maxValidWeek` debe estar en el
    render del tab Métricas para excluir chunks fantasma."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"c\.week_number\s*<=\s*_maxValidWeek",
        text,
    ), "Filter de week_number > _maxValidWeek requerido."


def test_chunk_metrics_endpoint_exposes_week_number():
    """Input del filter — endpoint debe seguir exponiéndolo."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "q.week_number" in src


def test_history_list_exposes_total_days_for_max_week_calc():
    """Inputs del cómputo de _maxValidWeek — total_days_requested y
    totalDays legacy."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "total_days_requested" in src
    assert "totalDays" in src
