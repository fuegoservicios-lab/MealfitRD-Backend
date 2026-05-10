"""[P0-HIST-FIX-8 · 2026-05-09] Cross-link backend del fix UX:
diferenciar error_message ACTIVO vs HISTÓRICO según el status del chunk.

Bug reportado en producción 2026-05-09:
    Chunk con `status='pending'` y `attempts=1` mostraba el chip
    "Error: 'charmap' codec can't encode character..." en el modal.
    El usuario preguntó "por qué aparece ese mensaje de error en el
    chunk en cola?".

    Causa: el LEFT JOIN con `plan_chunk_metrics` recupera el
    `error_message` del intento previo (que falló) aunque el chunk
    haya sido re-encolado para reintento. El error es HISTÓRICO,
    no activo.

Fix (frontend-only):
    Diferenciar render según status del chunk:
      - failed / pending_user_action → "Error: ..." (palette rojo
        tierBadgeBad). Error ACTIVO en estado terminal.
      - pending / processing / stale → "Último error: ..." (palette
        amber tierBadgeWarn). Error HISTÓRICO de intento previo;
        chunk será reintentado.
      - completed → no render (chunk superó el error, mostrarlo
        sería confuso).
    Tooltip enriquecido con prefijo "Error de un intento previo. El
    chunk fue re-encolado y se reintentará automáticamente." para
    estados no-terminales.

Cobertura backend (cross-link del marker):
    1. Anchor del marker en History.jsx.
    2. Endpoint /chunk-metrics expone `q.status` Y `m.error_message`
       (inputs del cómputo).
    3. Frontend declara variables _isActiveError, _isCompleted,
       _toneClass, _tooltipPrefix con la lógica condicional.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"


def test_marker_present_in_history_jsx():
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-8" in text


def test_history_jsx_detects_active_error_states():
    """`_isActiveError = status === failed || status === pending_user_action`
    — los dos estados terminales/blocking donde el error es 'el estado
    actual' del chunk."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"_isActiveError\s*=\s*c\.status\s*===\s*['\"]failed['\"][\s\S]{0,80}?c\.status\s*===\s*['\"]pending_user_action['\"]",
        text,
    )


def test_history_jsx_skips_completed_chunks():
    """Chunks completados pasaron del error → no mostrar ruido."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"_isCompleted\s*=\s*c\.status\s*===\s*['\"]completed['\"]",
        text,
    )
    assert "if (_isCompleted) return null" in text


def test_history_jsx_uses_tone_class_conditional():
    """Tone class condicional: bad para errores activos, warn para
    históricos."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"_toneClass\s*=\s*_isActiveError[\s\S]{0,150}?tierBadgeBad[\s\S]{0,150}?tierBadgeWarn",
        text,
    )


def test_history_jsx_tooltip_prefix_for_historical_errors():
    """El tooltip de errores históricos debe explicar al user que el
    chunk se reintentará automáticamente — sin esto ver "Último error"
    sin contexto sería igual de confuso que el bug original."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "_tooltipPrefix" in text
    assert "se reintentará automáticamente" in text


def test_chunk_metrics_endpoint_exposes_status_and_error_message():
    """Inputs del cómputo — endpoint debe seguir exponiéndolos."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "q.status" in src
    assert "m.error_message" in src
