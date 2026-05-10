"""[P0-HIST-FIX-9 · 2026-05-09] Cross-link backend del fix UX final
sobre el render de error_message: solo cuando es accionable.

Bug reportado en producción 2026-05-09 (escalación de FIX-8):
    Tras P0-HIST-FIX-8 (que diferenciaba "Error:" rojo vs "Último
    error:" amber por status), el usuario seguía confundido al ver
    el chip en chunks "En cola". Pidió: ocultar completamente el
    error en estados no-accionables.

Fix (frontend-only, simplificación):
    Render del error_message SOLO cuando bloquea al user:
      - status `failed` o `pending_user_action` → "Error: <texto>" rojo.
      - cualquier otro status (pending/processing/stale/completed)
        → no render.

    Variables del FIX-8 anterior (`_isActiveError`, `_isCompleted`,
    `_toneClass`, `_tooltipPrefix`, label condicional "Último error:")
    eliminadas. Lógica simplificada a un solo `_shouldShowError`.

    Quien necesite el error histórico de un chunk en cola sigue
    teniendo: "Intentos: N", "Calidad reducida", admin tools.

Cobertura backend (cross-link del marker):
    1. Anchor del marker en History.jsx.
    2. Endpoint /chunk-metrics expone `q.status` Y `m.error_message`
       (inputs sin cambio).
    3. Frontend declara `_shouldShowError` y NO declara las variables
       deprecated del FIX-8.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"


def test_marker_present_in_history_jsx():
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-9" in text


def test_history_jsx_uses_should_show_error():
    """Variable canónica del FIX-9: `_shouldShowError = status ===
    failed || pending_user_action`."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"_shouldShowError\s*=\s*c\.status\s*===\s*['\"]failed['\"][\s\S]{0,80}?c\.status\s*===\s*['\"]pending_user_action['\"]",
        text,
    )
    assert re.search(
        r"if\s*\(\s*!_shouldShowError\s*\)\s*return\s+null",
        text,
    )


def test_history_jsx_no_longer_uses_deprecated_fix8_vars():
    """Las variables del FIX-8 anterior (label condicional histórico/
    activo) ya no se usan — solo había sentido cuando renderizábamos
    en non-active states. FIX-9 simplifica."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    # Buscar dentro del bloque del IIFE de error_message (entre el
    # marker FIX-9 y el cierre del IIFE).
    fix9_idx = text.find("[P0-HIST-FIX-9")
    assert fix9_idx > -1
    block = text[fix9_idx:fix9_idx + 3000]
    # _isActiveError, _tooltipPrefix, _toneClass, "Último error:" NO
    # deben estar en este bloque (eran del FIX-8).
    assert "_isActiveError" not in block, (
        "Variable `_isActiveError` del FIX-8 sigue en el código — "
        "FIX-9 simplifica a `_shouldShowError`."
    )
    assert "_tooltipPrefix" not in block
    assert "Último error:" not in block


def test_chunk_metrics_endpoint_exposes_status_and_error_message():
    """Inputs sin cambio del FIX-8."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "q.status" in src
    assert "m.error_message" in src
