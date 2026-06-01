"""[P2-PIPELINE-ROBUSTNESS · 2026-05-30] Regression guards para 3 bugs del
pipeline de generación deep-search (audit prod-readiness 2026-05-30):

  1. P2-PIPELINE-FALLBACK-GUARD-DONE: `_fallback_postprocess` (done-callback)
     persistía un plan de emergencia `_is_fallback` sin el guard que SÍ tienen
     los 2 paths reales (sync L2447 + SSE L3157). Si el SSE generator moría
     antes de procesar `_done` y el LLM estaba caído, el usuario terminaba con
     un "Fallback: pollo y arroz" persistido + N chunks encolados.

  2. P2-PIPELINE-DISCONNECT-PERSIST: el 2º re-check de disconnect en el SSE
     generator hacía `break` ANTES del postprocess + KV-complete, con el
     sentinel ya en True → el done-callback no-op → plan NUNCA persistido y KV
     `pending_pipeline` en 'generating' para siempre (spinner perpetuo). El
     comentario decía "ya NO cortamos persistencia" pero el `break` vivía.

  3. P2-PLAN-PERSIST-FAILED: `_postprocess_pipeline_result` chunking-path
     ignoraba `save_partial_plan_get_id() -> None` (INSERT meal_plans fallido)
     → el generator marcaba KV `complete` con plan_id_final=None y emitía
     `complete`: phantom success (plan inexistente, sin alerta).

Cobertura cruzada: el contrato "disconnect NO hace break" lo ancla además
`test_p1_deep_search_pipeline.py::test_disconnect_does_not_break_persistence`
(regex endurecido en este mismo audit).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ROUTER = _BACKEND_ROOT / "routers" / "plans.py"
_SERVICES = _BACKEND_ROOT / "services.py"
_ALERT_DOC = _BACKEND_ROOT / "docs" / "system_alerts_resolution_table.md"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Bug 11 — fallback done-callback guard
# --------------------------------------------------------------------------

def test_done_callback_has_is_fallback_guard():
    """El `_fallback_postprocess` debe chequear `_is_fallback` ANTES de
    `_postprocess_pipeline_result`, igual que los 2 paths reales."""
    src = _read(_ROUTER)
    # Aísla el cuerpo de la closure `_fallback_postprocess`.
    m = re.search(r"async def _fallback_postprocess\(\):(.*?)\n {16}\S", src, re.DOTALL)
    body = m.group(1) if m else src  # fallback: scan whole file
    pos_guard = body.find('_result.get("_is_fallback")')
    pos_postprocess = body.find("_postprocess_pipeline_result")
    assert pos_guard != -1, (
        "El `_fallback_postprocess` (done-callback) debe chequear "
        "`_result.get('_is_fallback')` — sin él, un plan de emergencia se "
        "persiste cuando el SSE generator muere pre-postprocess (P2-PIPELINE-FALLBACK-GUARD-DONE)."
    )
    assert pos_postprocess != -1, "No se encontró la llamada a _postprocess_pipeline_result en el callback."
    assert pos_guard < pos_postprocess, (
        "El guard `_is_fallback` debe ejecutarse ANTES de `_postprocess_pipeline_result` "
        "en el done-callback."
    )


def test_fallback_guard_anchor_present():
    assert "P2-PIPELINE-FALLBACK-GUARD-DONE" in _read(_ROUTER)


# --------------------------------------------------------------------------
# Bug 13 — persist-failed propagation
# --------------------------------------------------------------------------

def test_postprocess_sets_persist_failed_on_none_plan_id():
    """El branch chunking debe tener un `else` (plan_id falsy) que marque
    `result['_persist_failed'] = True` + emita el alert."""
    src = _read(_ROUTER)
    assert 'result["_persist_failed"] = True' in src, (
        "El chunking-path de `_postprocess_pipeline_result` debe marcar "
        "`result['_persist_failed'] = True` cuando `save_partial_plan_get_id` "
        "devuelve None (INSERT fallido). Sin esto, el caller emite un phantom "
        "`complete` (P2-PLAN-PERSIST-FAILED)."
    )
    assert "_persist_plan_persist_failed_alert" in src, (
        "El path debe emitir el system_alert `plan_persist_failed` para visibilidad."
    )


def test_three_consumers_check_persist_failed():
    """Los 3 consumidores (sync, SSE, done-callback) deben chequear el flag."""
    src = _read(_ROUTER)
    n = src.count('.get("_persist_failed")')
    assert n >= 3, (
        f"Se esperaban >= 3 chequeos de `.get('_persist_failed')` (sync 503 + SSE "
        f"error + done-callback KV failed); encontré {n}. Si baja, algún consumidor "
        f"dejó de propagar la falla de persistencia y reabre el phantom-complete."
    )


def test_services_has_alert_helper_and_emits():
    src = _read(_SERVICES)
    assert "def _persist_plan_persist_failed_alert" in src, (
        "Falta el helper `_persist_plan_persist_failed_alert` en services.py."
    )
    assert "plan_persist_failed:" in src, (
        "El helper debe emitir el alert_key `plan_persist_failed:<user_id>`."
    )
    assert "INSERT INTO system_alerts" in src, (
        "El helper debe insertar en system_alerts."
    )
    # El background-save except debe invocar el alert.
    assert src.count("_persist_plan_persist_failed_alert(") >= 1


def test_persist_failed_alert_documented():
    """El alert_key debe estar en la tabla canónica (drift detection P2-AUDIT-4)."""
    doc = _read(_ALERT_DOC)
    assert "plan_persist_failed:" in doc, (
        "Falta la row `plan_persist_failed:<user_id>` en "
        "backend/docs/system_alerts_resolution_table.md."
    )


# --------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------

def test_all_anchors_present():
    src = _read(_ROUTER)
    for anchor in (
        "P2-PIPELINE-FALLBACK-GUARD-DONE",
        "P2-PIPELINE-DISCONNECT-PERSIST",
        "P2-PLAN-PERSIST-FAILED",
    ):
        assert anchor in src, f"Falta anchor `{anchor}` en routers/plans.py."
