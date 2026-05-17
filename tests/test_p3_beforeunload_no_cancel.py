"""[P3-BEFOREUNLOAD-NO-CANCEL · 2026-05-16] Anti-regresión: el handler
`beforeunload` que disparaba `cancelGeneration()` en tab close fue REMOVIDO
intencionalmente porque contradecía P1-DEEP-SEARCH-PIPELINE (2026-05-15).

Historia:
  - P2-NEW-15 (2026-05-11): añadió `useEffect` con `beforeunload` listener
    que llamaba `cancelGeneration()` al cerrar la tab. Razón: evitar
    "memleak del SSE reader" y "quema de cuota LLM con un user que ya se fue".
  - P1-DEEP-SEARCH-PIPELINE (2026-05-15): cambió el paradigma — el pipeline
    SIGUE corriendo cuando el SSE muere; el user recupera el plan al volver.
  - P3-PLAN-RECOVERY-LOADING (2026-05-16 PM): cerró el gap de UX donde
    el user volvía y no veía loading screen.
  - **P3-BEFOREUNLOAD-NO-CANCEL (este fix)**: removió el handler P2-NEW-15
    porque cancelaba el plan que se suponía debía persistir. Bug observable
    log 22:32:33 (plan 7deee778): user cerró tab → beforeunload disparó
    `POST /api/plans/cancel` → KV `pending_pipeline:bf6f1383` limpiado →
    PendingPipelineRecovery no encontró nada que recuperar → user vio el
    formulario al volver, no la pantalla de carga.

Concerns originales de P2-NEW-15 ya NO aplican:
  - Memleak del SSE reader: el browser GC lo limpia al cerrar tab (JS heap
    se libera). Trivial.
  - Quota LLM quemada: ya NO es desperdicio — el user vuelve y el plan se
    le entrega. Equivale a cualquier plan exitoso (SSE no interrumpido).

El botón "Cancelar" explícito (onCancel={cancelGeneration} en LoadingScreen)
SIGUE funcionando. Solo se removió el cancel AUTOMÁTICO en tab-close.
Intent del user es distinta:
  - Cerrar tab = "me voy, vuelvo después" → preservar pipeline.
  - Click cancelar = "no quiero este plan, descártalo" → cancel explícito.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_FRONTEND_ROOT = Path(__file__).resolve().parent.parent.parent / "frontend"
_PLAN_FP = _FRONTEND_ROOT / "src" / "pages" / "Plan.jsx"


@pytest.fixture(scope="module")
def plan_src() -> str:
    return _PLAN_FP.read_text(encoding="utf-8")


def test_p3_beforeunload_marker_present(plan_src: str):
    """El marker P3-BEFOREUNLOAD-NO-CANCEL debe estar en Plan.jsx,
    documentando la decisión de revertir P2-NEW-15."""
    assert "P3-BEFOREUNLOAD-NO-CANCEL" in plan_src, (
        "Marker P3-BEFOREUNLOAD-NO-CANCEL ausente en Plan.jsx. Sin él, un "
        "refactor cosmético podría re-añadir el handler beforeunload sin "
        "saber por qué se removió, rompiendo P1-DEEP-SEARCH-PIPELINE."
    )


def test_no_beforeunload_cancel_handler(plan_src: str):
    """ANTI-REGRESIÓN: NO debe haber un handler `addEventListener('beforeunload', ...)`
    que invoque `cancelGeneration()`. Eso revertiría el fix y reintroduciría
    el bug del usuario 2026-05-16."""
    # Buscar pattern: addEventListener('beforeunload', ...) seguido en pocas
    # líneas por cancelGeneration().
    pattern = re.compile(
        r"addEventListener\(\s*['\"]beforeunload['\"][\s\S]{0,500}?cancelGeneration\(",
        re.DOTALL,
    )
    assert not pattern.search(plan_src), (
        "REGRESIÓN P3-BEFOREUNLOAD-NO-CANCEL: detectado un listener "
        "`beforeunload` que invoca `cancelGeneration()`. Esto revierte el "
        "fix — cuando user cierra tab, el cancel borraría el KV "
        "`pending_pipeline` y PendingPipelineRecovery no encontraría nada "
        "que recuperar al volver. Eliminar el handler. El cancel SOLO debe "
        "dispararse desde click explícito del botón 'Cancelar'."
    )


def test_cancel_generation_still_exists(plan_src: str):
    """El `cancelGeneration()` SIGUE existiendo — solo se removió la
    invocación AUTOMÁTICA en beforeunload. El botón 'Cancelar' explícito
    sigue funcional."""
    assert "export const cancelGeneration" in plan_src, (
        "`cancelGeneration` desapareció — esto rompe el botón 'Cancelar' "
        "explícito en LoadingScreen. Solo se debía remover la invocación "
        "automática en beforeunload, NO la función entera."
    )


def test_onCancel_still_wired_to_cancelGeneration(plan_src: str):
    """El prop `onCancel` en LoadingScreen (y demás) sigue cableado a
    `cancelGeneration` — el botón explícito funciona."""
    # Buscar al menos una instancia de `onCancel={cancelGeneration}`.
    assert "onCancel={cancelGeneration}" in plan_src, (
        "El prop `onCancel={cancelGeneration}` desapareció — el botón "
        "'Cancelar' en LoadingScreen ya no funciona. Re-conectar."
    )
