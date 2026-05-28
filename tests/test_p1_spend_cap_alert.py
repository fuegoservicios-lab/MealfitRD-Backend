"""[P1-SPEND-CAP-ALERT · 2026-05-28] Contrato del manejo del spending-cap de Gemini.

Incidente raíz (2026-05-28, user bf6f1383): el pipeline cayó con
`429 RESOURCE_EXHAUSTED: monthly spending cap` pero (a) el único rastro quedó en
logs — cero `system_alerts` para el operador — y (b) el usuario veía "IA
temporalmente saturada, intenta en 1-2 min", mensaje FALSO (reintentar no ayuda
hasta subir el cap en https://ai.studio/spend).

Este test ancla, parser-based (robusto a venv roto), las tres patas del fix:
  1. graph_orchestrator: detecta el spending-cap reusando el detector canónico
     `_is_gemini_spending_cap_error` (NO re-implementa el substring matching),
     marca `plan_result._llm_spend_cap` y emite el alert al operador.
  2. routers/plans.py: AMBOS guards (`/analyze` sync 503 y `/analyze/stream` SSE)
     dan un mensaje honesto cuando `_llm_spend_cap`, conservando el mensaje
     transitorio "1-2 minutos" para el resto de fallbacks.
  3. La row `gemini_spend_cap_exceeded` está documentada en la tabla canónica
     (el test de drift bidireccional P2-AUDIT-4 también lo exige).

Si renombras `_persist_gemini_spend_cap_alert` o el alert_key, este test falla
ANTES que el cambio llegue a producción (tooltip-anchor en el source).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"

_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_DOC = (_BACKEND / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")


def test_helper_and_alert_key_exist():
    assert "def _persist_gemini_spend_cap_alert(" in _GRAPH, (
        "Falta el helper `_persist_gemini_spend_cap_alert` en graph_orchestrator.py."
    )
    assert '"gemini_spend_cap_exceeded"' in _GRAPH, (
        "El alert_key canónico `gemini_spend_cap_exceeded` no aparece como literal "
        "en graph_orchestrator.py (el test de drift P2-AUDIT-4 lo necesita ahí)."
    )


def test_detection_reuses_canonical_detector_and_flags_result():
    # Reusa el detector de shopping_calculator (no re-implementa substring match).
    assert "_is_gemini_spending_cap_error" in _GRAPH, (
        "graph_orchestrator debe reusar `_is_gemini_spending_cap_error` "
        "(detector canónico en shopping_calculator), no re-implementarlo."
    )
    assert "_spend_cap_hit" in _GRAPH
    # El pipeline marca el plan_result y emite el alert cuando hay spend-cap.
    assert '"_llm_spend_cap"' in _GRAPH, (
        "El pipeline debe marcar plan_result['_llm_spend_cap'] para que plans.py "
        "dé el mensaje honesto."
    )
    assert "_persist_gemini_spend_cap_alert(" in _GRAPH, (
        "El catch del pipeline debe invocar `_persist_gemini_spend_cap_alert` "
        "cuando detecta spend-cap."
    )


def test_plans_both_guards_branch_honest_message():
    # Ambos guards leen el flag.
    assert _PLANS.count('result.get("_llm_spend_cap")') >= 2, (
        "Los dos guards (sync 503 + SSE) deben ramificar por `_llm_spend_cap`."
    )
    # Mensaje honesto (sin promesa falsa de '1-2 minutos') en ambos guards.
    honest = "Estamos trabajando para restablecerlo"
    assert _PLANS.count(honest) >= 2, (
        "El mensaje honesto de spend-cap debe estar en ambos guards "
        "(/analyze sync y /analyze/stream)."
    )
    # El mensaje transitorio se preserva para el resto de fallbacks (no-cap).
    assert "intenta de nuevo en 1-2 minutos" in _PLANS, (
        "El mensaje transitorio '1-2 minutos' debe preservarse para fallbacks "
        "que NO son spend-cap."
    )


def test_alert_key_documented_in_resolution_table():
    assert "gemini_spend_cap_exceeded" in _DOC, (
        "La row `gemini_spend_cap_exceeded` debe estar en "
        "backend/docs/system_alerts_resolution_table.md (drift P2-AUDIT-4)."
    )
    assert "ai.studio/spend" in _DOC, (
        "La row debe apuntar al remediation URL del operador (ai.studio/spend)."
    )


def test_detector_distinguishes_spend_cap_from_transient_429():
    """Funcional (guarded): el detector canónico separa el cap persistente del
    rate-limit transitorio. Se salta si el entorno no puede importar el módulo
    (venv roto) — las patas estructurales arriba ya anclan el contrato."""
    try:
        from shopping_calculator import _is_gemini_spending_cap_error
    except Exception as e:  # pragma: no cover - entorno sin deps
        pytest.skip(f"shopping_calculator no importable en este entorno: {e!r}")

    cap = Exception(
        "Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429 "
        "RESOURCE_EXHAUSTED. Your project has exceeded its monthly spending cap. "
        "Please go to AI Studio at https://ai.studio/spend to manage your project."
    )
    transient = Exception("429 RESOURCE_EXHAUSTED: rate limit exceeded, retry later")

    assert _is_gemini_spending_cap_error(cap) is True
    assert _is_gemini_spending_cap_error(transient) is False
