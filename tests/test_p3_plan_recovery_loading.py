"""[P3-PLAN-RECOVERY-LOADING · 2026-05-16] Cierre del gap donde el usuario
cerraba la app/laptop durante la generación de un plan, y al volver veía
el formulario (`/assessment`) en lugar de la pantalla de carga (`/plan`).

Contexto:
  - P1-DEEP-SEARCH-PIPELINE (2026-05-15) mantiene el pipeline backend vivo
    aunque el SSE del cliente se desconecte. El plan se persiste y el KV
    `pending_pipeline:<user_id>` tiene `status='generating'`.
  - PendingPipelineRecovery (App.jsx) polea cada 10s `/api/plans/pending-status`.
  - P3-RECOVERY-NO-REDIRECT-LOOP (2026-05-16) DESACTIVÓ el redirect a /plan
    cuando se detectaba generating, para evitar loop infinito Plan.jsx → SSE
    → 409 → /dashboard → recovery → /plan → ...

Síntoma reportado por usuario 2026-05-16: al volver tras cerrar laptop, el
recovery NO redirigía a /plan (por P3-RECOVERY-NO-REDIRECT-LOOP) → user landed
en /assessment y no veía la pantalla de carga.

Fix de 2 partes:

  (1) Plan.jsx::processPlan agrega pre-flight check de `/pending-status`. Si
      `status='generating'`, MOSTRAR loading screen pero NO disparar SSE.
      Esto cierra el origen del loop: sin SSE, no hay 409, no hay redirect
      a /dashboard.

  (2) PendingPipelineRecovery RE-HABILITA el navigate a /plan cuando
      `status='generating'` AND `location.pathname !== '/plan'`. Plan.jsx
      ahora es idempotente (no 409) → loop cerrado.

Tests parser-based porque los componentes React necesitan DOM + context para
ejecutar. Los anchors importan: el patrón "loop fix" depende de la combinación
exacta de los 2 cambios.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_FRONTEND_ROOT = Path(__file__).resolve().parent.parent.parent / "frontend"
_PLAN = (_FRONTEND_ROOT / "src" / "pages" / "Plan.jsx").read_text(encoding="utf-8")
_RECOVERY = (
    _FRONTEND_ROOT / "src" / "components" / "PendingPipelineRecovery.jsx"
).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fix #1: Plan.jsx detecta pending pre-SSE
# ---------------------------------------------------------------------------


def test_plan_jsx_has_pending_check_before_sse():
    """Plan.jsx::processPlan debe chequear `/api/plans/pending-status` ANTES
    de la fase de SSE. Sin esto, el navigate de recovery a /plan dispararía
    SSE → 409 → loop."""
    assert "P3-PLAN-RECOVERY-LOADING" in _PLAN, (
        "Marker P3-PLAN-RECOVERY-LOADING ausente en Plan.jsx — el check "
        "pre-SSE fue removido."
    )
    # Buscar el call a pending-status DENTRO de processPlan (debe estar
    # ANTES del FASE 1 "analizando").
    idx = _PLAN.find("const processPlan = async () =>")
    assert idx > 0, "processPlan no encontrado."
    end = _PLAN.find("FASE 1: UI de", idx)
    pre_block = _PLAN[idx:end if end > 0 else idx + 5000]

    assert "/api/plans/pending-status" in pre_block, (
        "Plan.jsx no chequea `/api/plans/pending-status` ANTES de FASE 1 "
        "(analizando). Sin este check, redirige a SSE incluso cuando hay "
        "pending pipeline → 409 + loop."
    )


def test_plan_jsx_skips_sse_when_generating():
    """Cuando el pre-flight detecta `status='generating'`, Plan.jsx DEBE
    `return` antes del SSE (no continuar al stream)."""
    idx = _PLAN.find("P3-PLAN-RECOVERY-LOADING")
    assert idx > 0
    block = _PLAN[idx:idx + 3000]
    # El branch generating debe:
    # 1. setStatus('generating')
    # 2. return (early exit)
    assert "pendingData?.status === 'generating'" in block, (
        "El check no compara `pendingData?.status === 'generating'`."
    )
    # Debe haber un `return;` dentro del if(generating)
    # Pattern: setStatus('generating') ... return;
    has_early_return = re.search(
        r"pendingData\?\.status === 'generating'.*?return;",
        block,
        re.DOTALL,
    )
    assert has_early_return, (
        "El branch generating no termina con `return;` — el SSE se "
        "dispararía igualmente, causando 409 y el loop."
    )


# ---------------------------------------------------------------------------
# Fix #2: PendingPipelineRecovery re-habilita navigate a /plan
# ---------------------------------------------------------------------------


def test_recovery_navigates_to_plan_when_generating():
    """PendingPipelineRecovery DEBE navegar a `/plan` cuando detecta
    `status='generating'` Y NO estamos ya en `/plan`. Sin este redirect,
    el user landed en /assessment y no ve loading."""
    assert "P3-PLAN-RECOVERY-LOADING" in _RECOVERY, (
        "Marker P3-PLAN-RECOVERY-LOADING ausente en PendingPipelineRecovery "
        "— el redirect a /plan no se re-habilitó."
    )
    # Patrón: dentro del branch generating, hay navigate('/plan', ...)
    # condicionado a location.pathname !== '/plan'
    idx = _RECOVERY.find("status.status === 'generating'")
    assert idx > 0
    block = _RECOVERY[idx:idx + 3000]
    assert "navigate('/plan'" in block, (
        "Recovery NO navega a /plan cuando generating. El user landed "
        "en /assessment sin ver loading screen."
    )
    assert "location.pathname !== '/plan'" in block, (
        "El navigate no está gated por `location.pathname !== '/plan'`. "
        "Sin este gate, si el user ya está en /plan, el navigate se "
        "dispara redundantemente cada 10s del polling."
    )


def test_recovery_navigate_does_not_set_handled():
    """CRÍTICO: el navigate a /plan NO debe marcar `handledRef.current = true`.
    Si lo hace, el polling se detiene y nunca detectamos `complete` →
    el user se queda en loading screen para siempre."""
    idx = _RECOVERY.find("status.status === 'generating'")
    end = _RECOVERY.find("} else if (status.status === 'failed')", idx)
    block = _RECOVERY[idx:end if end > 0 else idx + 3000]
    # En el branch generating, NO debe haber handledRef.current = true:
    assert "handledRef.current = true" not in block, (
        "El branch generating NO debe setear `handledRef.current = true` — "
        "eso detendría el polling y el user se quedaría en loading screen "
        "permanentemente sin recibir el toast/redirect cuando complete."
    )


# ---------------------------------------------------------------------------
# Loop closure: ambos fixes deben coexistir para cerrar el loop original
# ---------------------------------------------------------------------------


def test_loop_closure_anchors():
    """Sanity end-to-end: ambos componentes tienen los anchors del fix.
    Si UNO se revierte sin el otro, el loop vuelve."""
    plan_has_check = (
        "P3-PLAN-RECOVERY-LOADING" in _PLAN
        and "pendingData?.status === 'generating'" in _PLAN
    )
    recovery_has_navigate = (
        "P3-PLAN-RECOVERY-LOADING" in _RECOVERY
        and "navigate('/plan'" in _RECOVERY
    )
    assert plan_has_check and recovery_has_navigate, (
        "Solo uno de los 2 fixes está presente. Necesitan ambos para cerrar "
        "el loop:\n"
        "  - Plan.jsx debe skip SSE cuando pending → cierra origen del 409.\n"
        "  - Recovery debe redirect a /plan cuando generating → cierra UX gap.\n"
        f"Plan.jsx tiene el check: {plan_has_check}\n"
        f"Recovery tiene el navigate: {recovery_has_navigate}"
    )
