"""[P0-HIST-FIX-10 · 2026-05-09] Cross-link backend del fix del filter
del tab Métricas: filtrar chunks por rango de días, NO por week_number.

Bug reportado tras crear plan sintético de 30 días con 8 chunks reales:
    `split_with_absorb(30, 3) = [3,4,4,4,4,4,4,3]` — 8 chunks, week_numbers
    1..8 (asignación secuencial MAX+1 en routers/plans.py:1745). FIX-7
    anterior aplicaba `c.week_number <= ceil(displayTotal/7) = 5` y
    ocultaba weeks 6,7,8 — chunks completamente legítimos cubriendo
    días 24-30 dentro del plan.

Causa raíz:
    week_number en producción es **secuencial por orden de creación**
    (cron rolling refill incrementa MAX+1), NO cronológico. FIX-7
    asumía cronológico — falso. Para planes ≥6 chunks la función
    `Math.ceil(N/7)` infra-cuenta los slots disponibles.

Fix (frontend-only):
    Reemplazar el predicado por `days_offset + days_count <= total`,
    que es el invariante real ("el chunk cubre días dentro del plan").
    week_number queda como fallback para chunks legacy sin
    days_offset/days_count.

Cobertura backend (cross-link del marker):
    1. Anchor del marker FIX-10 en History.jsx.
    2. Predicado `days_offset + days_count <= _filterDisplayTotal`
       presente en el render del tab Métricas.
    3. Endpoint /chunk-metrics expone days_offset y days_count
       (inputs del nuevo filter).
    4. Fallback legacy a week_number preservado para chunks viejos.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"


def test_marker_present_in_history_jsx():
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-10" in text


def test_history_jsx_filters_chunks_by_days_range():
    """El predicado canónico del FIX-10:
    `(c.days_offset + c.days_count) <= _filterDisplayTotal`."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"\(c\.days_offset\s*\+\s*c\.days_count\)\s*<=\s*_filterDisplayTotal",
        text,
    ), "Filter por rango de días requerido (FIX-10)."


def test_history_jsx_preserves_week_number_fallback():
    """Chunks legacy sin days_offset/days_count caen al filtro
    por week_number — preservar el branch defensivo."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"c\.week_number\s*<=\s*_maxValidWeek",
        text,
    ), "Fallback legacy por week_number requerido."


def test_history_jsx_checks_typeof_for_days_range_filter():
    """Defensa contra payloads mal-formados: typeof === 'number'
    para days_offset Y days_count antes de hacer la suma."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    # Buscar dentro del bloque del filter (entre marker FIX-10 y
    # el cierre del filter).
    fix10_idx = text.find("[P0-HIST-FIX-10")
    assert fix10_idx > -1
    block = text[fix10_idx:fix10_idx + 3000]
    assert re.search(
        r"typeof\s+c\.days_offset\s*===\s*['\"]number['\"]",
        block,
    )
    assert re.search(
        r"typeof\s+c\.days_count\s*===\s*['\"]number['\"]",
        block,
    )


def test_chunk_metrics_endpoint_exposes_days_offset_and_days_count():
    """Inputs del filter — endpoint debe exponer ambos campos."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "q.days_offset" in src
    assert "q.days_count" in src


def test_history_jsx_documents_root_cause_week_number_sequential():
    """Comentario load-bearing cita la causa raíz: week_number
    es secuencial (MAX+1), no cronológico — para que un futuro
    refactor que vea `_maxValidWeek` no asuma equivalencia."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    fix10_idx = text.find("[P0-HIST-FIX-10")
    assert fix10_idx > -1
    block = text[fix10_idx:fix10_idx + 2000]
    # Debe mencionar "secuencial" o "MAX+1" para señalar el modelo
    # real del backend.
    assert ("secuencial" in block.lower()) or ("max+1" in block.lower()), (
        "El comentario debe documentar que week_number es secuencial, "
        "no cronológico — sin esto, futuros refactors podrían recaer "
        "en el mismo bug."
    )
