"""[P0-HIST-FIX-5 · 2026-05-09] Cross-link backend del fix UX al
tab Métricas — humanización de status / chunk_kind / chips.

Bug reportado en producción 2026-05-09 (screenshot del usuario):
    El tab Métricas mostraba copies internos crudos:
      - Status `completed` / `pending` (snake_case)
      - chunk_kind `first_chunk` (no humanizado, fallback al code crudo)
      - "tier: —" cuando no hay tier (ruido)
      - Chip "Degraded" (en inglés, jerga interna)
      - Chip "Sin learning" (jerga interna del backend T2/commit)
      - "Lag: 0s" (jerga; "espera" es más claro)

    Operator/usuario power leía los chips pero no entendía el
    significado sin docs/tooltip — el modal del Historial es para
    POST-MORTEM del plan, no para internos del sistema.

Fix (frontend-only):
    - Helper SSOT `chunkStatus.js` con map status → es-DO.
      `completed` → "Completado", `pending` → "En cola", etc.
    - `chunkKinds.js` extendido con `first_chunk` → "Inicial"
      (alias semántico de `initial_plan` que la DB tiene legacy).
    - Meta line: drop "tier: —" cuando no hay tier; uppercase
      cuando sí lo hay (LLM, SHUFFLE).
    - Chips humanizados:
      - "Degraded" → "Calidad reducida" + tooltip explicativo.
      - "Sin learning" → "Sin aprendizaje guardado" + tooltip.
      - "Lag:" → "Espera:" + tooltip explica saturación.
      - dead_letter_reason → prefijo "No recuperable: <razón>".
    - Tooltips en TODOS los chips diagnósticos (Duración, Espera,
      Intentos, Repetición) explican qué mide cada métrica.

Cobertura backend (cross-link del marker):
    1. Anchor del marker en History.jsx, chunkStatus.js, chunkKinds.js.
    2. Endpoint chunk-metrics expone los campos canónicos (status,
       quality_tier, was_degraded, learning_repeat_pct, etc.) que el
       frontend humaniza.
    3. Helper chunkStatus cubre los 6 status canónicos del enum
       `plan_chunk_queue.status`.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
_CHUNK_STATUS_JS = _REPO_ROOT / "frontend" / "src" / "utils" / "chunkStatus.js"
_CHUNK_KINDS_JS = _REPO_ROOT / "frontend" / "src" / "utils" / "chunkKinds.js"


# ---------------------------------------------------------------------------
# 1. Anchors del marker
# ---------------------------------------------------------------------------
def test_marker_present_in_history_jsx():
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-5" in text


def test_marker_present_in_chunk_status_js():
    assert _CHUNK_STATUS_JS.exists(), (
        "Helper chunkStatus.js debe existir."
    )
    text = _CHUNK_STATUS_JS.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-5" in text


def test_marker_present_in_chunk_kinds_js():
    text = _CHUNK_KINDS_JS.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-5" in text, (
        "chunkKinds.js debe citar el marker donde se agregó "
        "`first_chunk` como alias de `initial_plan`."
    )


# ---------------------------------------------------------------------------
# 2. Helper chunkStatus cubre los 6 statuses canónicos del enum
# ---------------------------------------------------------------------------
def test_chunk_status_helper_covers_canonical_enum():
    """`plan_chunk_queue.status` enum tiene 6 valores (verificable en
    DB/migrations). El helper debe mapearlos todos para no caer al
    fallback raw cuando aparezcan en el modal."""
    text = _CHUNK_STATUS_JS.read_text(encoding="utf-8")
    canonical_statuses = {
        "completed",
        "pending",
        "processing",
        "stale",
        "failed",
        "pending_user_action",
    }
    for status in canonical_statuses:
        assert f"{status}:" in text, (
            f"Helper chunkStatus.js no mapea `{status}`. Aparecerá "
            f"crudo en el meta line cuando un chunk tenga ese status."
        )


def test_chunk_status_helper_exports_get_label_function():
    """Función pública `getChunkStatusLabel` debe estar exportada."""
    text = _CHUNK_STATUS_JS.read_text(encoding="utf-8")
    assert "export const getChunkStatusLabel" in text


# ---------------------------------------------------------------------------
# 3. chunkKinds tiene first_chunk + alias semánticos
# ---------------------------------------------------------------------------
def test_chunk_kinds_includes_first_chunk():
    """`first_chunk` aparece en DB legacy aunque no como literal en
    código backend. El helper lo mapea para que el badge no muestre
    snake_case."""
    text = _CHUNK_KINDS_JS.read_text(encoding="utf-8")
    assert "first_chunk:" in text, (
        "chunkKinds.js debe mapear `first_chunk` (alias legacy de "
        "`initial_plan`)."
    )


# ---------------------------------------------------------------------------
# 4. History.jsx usa los helpers
# ---------------------------------------------------------------------------
def test_history_jsx_imports_chunk_status_helper():
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"import\s*\{\s*getChunkStatusLabel\s*\}\s*from\s*['\"]\.\./utils/chunkStatus['\"]",
        text,
    ), "History.jsx debe importar getChunkStatusLabel del helper SSOT."


def test_history_jsx_uses_humanized_chip_labels():
    """Los chips humanizados deben aparecer en el render, NO los
    crudos del backend."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    # Chip "Calidad reducida" reemplaza "Degraded".
    assert "Calidad reducida" in text, (
        "Chip `was_degraded === true` debe decir 'Calidad reducida' "
        "(no 'Degraded')."
    )
    # Chip "Sin aprendizaje guardado" reemplaza "Sin learning".
    assert "Sin aprendizaje guardado" in text, (
        "Chip de T2 fail debe decir 'Sin aprendizaje guardado' "
        "(no 'Sin learning')."
    )
    # "Espera:" reemplaza "Lag:".
    assert ">Espera:" in text or "Espera: " in text, (
        "Chip de lag debe decir 'Espera:' (no 'Lag:')."
    )


# ---------------------------------------------------------------------------
# 5. Backend payload contract preservado
# ---------------------------------------------------------------------------
def test_chunk_metrics_endpoint_exposes_status_and_tier():
    """El frontend humaniza estos campos — backend debe seguir
    exponiéndolos en el payload."""
    from routers.plans import api_plan_chunk_metrics
    src = inspect.getsource(api_plan_chunk_metrics)
    assert "q.status" in src
    assert "q.quality_tier" in src
    assert "m.was_degraded" in src
    assert "m.learning_repeat_pct" in src
