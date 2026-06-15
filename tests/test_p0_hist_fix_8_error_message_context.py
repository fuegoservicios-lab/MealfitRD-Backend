"""[P0-HIST-FIX-8 · 2026-05-09 · SUPERSEDED por P0-HIST-FIX-9 · 2026-05-09] Cross-link backend del fix UX
del error_message del chunk según su status.

[drift-fix · 2026-06-15] P0-HIST-FIX-8 mostraba el error HISTÓRICO con tono warn + tooltip "se
reintentará". P0-HIST-FIX-9 (History.jsx ~4441) lo SUPERSEDED con una decisión más limpia: mostrar el
error SOLO cuando es ACCIONABLE (`status` failed / pending_user_action → badge rojo `tierBadgeBad` con
el error crudo en el title) y OCULTARLO (`return null`) para pending/processing/stale/completed (el
contexto histórico vive en los chips diagnósticos: "Intentos: N", "Calidad reducida"). Estos tests se
actualizaron al contrato vigente de P0-HIST-FIX-9 (antes asertaban las variables viejas
`_isActiveError`/`_isCompleted`/`_toneClass`/`_tooltipPrefix`, ya inexistentes).

Comportamiento ORIGINAL P0-HIST-FIX-8 (histórico, ya no vigente):
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
    # P0-HIST-FIX-9 superseded a P0-HIST-FIX-8 y es el marker que gobierna el render actual del error.
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "P0-HIST-FIX-9" in text


def test_history_jsx_detects_actionable_error_states():
    """[P0-HIST-FIX-9] El error se muestra SOLO en estados ACCIONABLES:
    `_shouldShowError = status === failed || status === pending_user_action`."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"_shouldShowError\s*=\s*c\.status\s*===\s*['\"]failed['\"][\s\S]{0,80}?c\.status\s*===\s*['\"]pending_user_action['\"]",
        text,
    )


def test_history_jsx_hides_non_actionable_chunks():
    """[P0-HIST-FIX-9] Estados no-accionables (pending/processing/stale/completed) NO renderean el error
    (`if (!_shouldShowError) return null`) — el error histórico sería ruido; su contexto vive en los chips
    diagnósticos. Reemplaza el `_isCompleted return null` de P0-HIST-FIX-8 (que solo ocultaba completed)."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "if (!_shouldShowError) return null" in text


def test_history_jsx_active_error_uses_bad_tone():
    """[P0-HIST-FIX-9] El error accionable se renderiza con la paleta roja `tierBadgeBad` (estado de
    error real). El tono condicional bad/warn de P0-HIST-FIX-8 desapareció: los históricos ya no se
    muestran con warn, simplemente se ocultan."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    # En el badge del error (tras `if (!_shouldShowError) return null`) se usa tierBadgeBad + errorMessageBadge.
    assert re.search(
        r"if \(!_shouldShowError\) return null[\s\S]{0,400}?tierBadgeBad[\s\S]{0,120}?errorMessageBadge",
        text,
    )


def test_history_jsx_active_error_shows_raw_in_title():
    """[P0-HIST-FIX-9] El error accionable lleva el mensaje crudo en el `title` (tooltip) + un resumen
    corto visible. Reemplaza el `_tooltipPrefix`/'se reintentará' de P0-HIST-FIX-8 (que era para los
    históricos, ahora ocultos)."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"if \(!_shouldShowError\) return null[\s\S]{0,400}?title=\{_raw\}",
        text,
    )


def test_chunk_metrics_endpoint_exposes_status_and_error_message():
    """Inputs del cómputo — endpoint debe seguir exponiéndolos."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "q.status" in src
    assert "m.error_message" in src
