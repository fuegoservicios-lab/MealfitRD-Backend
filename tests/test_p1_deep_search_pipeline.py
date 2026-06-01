"""[P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Regression guards para el modo
"deep-search style" de generación de plan.

Pre-fix: el pipeline se cancelaba si el cliente cerraba el SSE (P0-3 +
P6-CANCEL-PROPAGATION-FIX-3). Razón histórica: evitar "plan huérfano" en
DB. Resultado: si el usuario cerraba la pestaña a mitad de generación, todo
el progreso se perdía y debía empezar de cero al volver.

Fix: el pipeline corre como task independiente del SSE. Persistencia
+ KV tracking en `app_kv_store::pending_pipeline:<user_id>` permiten al
frontend (vía `<PendingPipelineRecovery />`) detectar el plan al volver
y redirigir al dashboard automáticamente.

Trade-off documentado: el pipeline cuesta Gemini quota aunque el usuario
abandone. Mitigaciones: guardrail "1 pipeline activo por user <15min"
(rechaza 409 nuevos requests) + cron de limpieza de stale rows.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_PLANS = _BACKEND_ROOT / "db_plans.py"
_ROUTER = _BACKEND_ROOT / "routers" / "plans.py"
_FRONT_ROOT = _BACKEND_ROOT.parent / "frontend" / "src"
_RECOVERY = _FRONT_ROOT / "components" / "PendingPipelineRecovery.jsx"
_APP = _FRONT_ROOT / "App.jsx"
_PLAN = _FRONT_ROOT / "pages" / "Plan.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ----- Backend: helpers en db_plans.py -----

def test_db_plans_helpers_defined():
    text = _read(_DB_PLANS)
    for name in (
        "upsert_pending_pipeline",
        "get_pending_pipeline",
        "check_user_has_active_pipeline",
        "clear_pending_pipeline",
    ):
        assert re.search(rf"^def {re.escape(name)}\(", text, re.MULTILINE), (
            f"Falta helper `{name}` en db_plans.py."
        )


def test_db_plans_uses_app_kv_store():
    """Storage layer = `app_kv_store` (no toca `meal_plans` schema).
    Key prefix = `pending_pipeline:<user_id>`."""
    text = _read(_DB_PLANS)
    assert "_PENDING_PIPELINE_KV_PREFIX" in text
    assert "pending_pipeline:" in text
    assert "app_kv_store" in text


# ----- Backend: guardrail + endpoints -----

def test_guardrail_409_on_active_pipeline():
    """Si el user ya tiene un pipeline `status=generating <15min`, el
    endpoint debe retornar 409 con `pipeline_already_running`."""
    text = _read(_ROUTER)
    assert "pipeline_already_running" in text, (
        "Falta el code='pipeline_already_running' en el guardrail."
    )
    assert "check_user_has_active_pipeline" in text, (
        "El endpoint /analyze/stream debe llamar al guardrail."
    )


def test_endpoint_pending_status_defined():
    text = _read(_ROUTER)
    assert "async def api_pending_pipeline_status" in text
    assert '@router.get("/pending-status")' in text
    assert "async def api_pending_pipeline_ack" in text
    assert '@router.post("/pending-status/ack")' in text


# ----- Backend: pipeline NO se cancela en disconnect -----

def test_pipeline_task_cancel_only_on_explicit_or_wallclock():
    """El pipeline NO debe cancelarse por disconnect (deep-search es feature).

    Pre-fix tenía 3 sites de `_pipeline_task.cancel()` (disconnect en
    _should_stop, except CancelledError, finally). El contrato es que NINGÚN
    cancel viva en un path de disconnect.

    [P2-GEN-WALL-TIMEOUT actualización · 2026-05-30] El conteo legítimo subió de
    1 a 2 cuando se añadió `_pipeline_wall_clock_guard` (anti-zombi: cancela si el
    pipeline excede el wall-clock timeout). Los 2 cancels legítimos son:
      1. cancel explícito por `is_session_cancelled(session_id)` (usuario clickeó
         "Cancelar Generación").
      2. wall-clock guard (anti-zombi por timeout duro).
    NINGUNO es un disconnect-cancel. La protección "disconnect NO corta
    persistencia" la ancla `test_disconnect_does_not_break_persistence`.
    """
    text = _read(_ROUTER)
    cancel_calls = re.findall(r"_pipeline_task\.cancel\(\)", text)
    assert len(cancel_calls) == 2, (
        f"P1-DEEP-SEARCH-PIPELINE: esperaba EXACTAMENTE 2 `_pipeline_task.cancel()` "
        f"(explícito por session_id + wall-clock anti-zombi), encontré {len(cancel_calls)}. "
        f"Si hay más, posiblemente el disconnect volvió a cancelar el pipeline — "
        f"deep-search dejaría de funcionar. Verifica que ningún cancel nuevo esté en "
        f"un path de `is_disconnected()`."
    )
    # Invariante real: ningún `_pipeline_task.cancel()` dentro de un bloque
    # `is_disconnected()` (1..8 líneas). Eso reabriría el bug que deep-search cerró.
    bad = re.findall(r"is_disconnected\(\):(?:[^\n]*\n){1,8}?\s*_pipeline_task\.cancel\(\)", text)
    assert len(bad) == 0, (
        f"P1-DEEP-SEARCH-PIPELINE: encontré {len(bad)} `_pipeline_task.cancel()` "
        f"dentro de un check `is_disconnected()` — el disconnect NO debe cancelar el "
        f"pipeline (el plan huérfano es feature)."
    )


def test_disconnect_does_not_break_persistence():
    """Pre-fix tenía 2 `if await request.is_disconnected(): break` que
    cortaban la persistencia. Post-fix deben ser `logger.info` que
    continúa el flujo."""
    text = _read(_ROUTER)
    # Buscar la región post-`_done` (donde estaba la persistencia).
    # No debe haber `break` después de `is_disconnected` en el contexto del SSE
    # generator. (El break en `if isinstance(result, dict) and result.get("_is_fallback"):`
    # es legítimo — ese caso NO se persiste por diseño separado.)
    region = re.search(
        r"if event_data\.get\(\"event\"\) == \"_done\":.*?return\s*$",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if region:
        body = region.group(0)
        # [P2-PIPELINE-DISCONNECT-PERSIST · 2026-05-30] Regex ENDURECIDO. El patrón
        # previo `is_disconnected():\s*\n[^\n]*\n[^\n]*break` solo capturaba la forma
        # de 2 líneas entre `is_disconnected():` y `break`. La forma REAL en prod era
        # un `logger.warning(...)` multi-línea (4 líneas) seguido de `break` — el
        # regex NO la matcheaba → blind spot: el test daba falso verde mientras un
        # `break` vivo dejaba el KV `pending_pipeline` en 'generating' para siempre.
        # Nuevo patrón: `is_disconnected():` seguido de 1..8 líneas y luego `break`
        # (non-greedy). El `break` legítimo del guard `_is_fallback` NO matchea porque
        # NO está precedido por `is_disconnected()`.
        bad = re.findall(r"is_disconnected\(\):(?:[^\n]*\n){1,8}?\s*break\b", body)
        assert len(bad) == 0, (
            f"P1-DEEP-SEARCH-PIPELINE: encontré {len(bad)} `is_disconnected → break` "
            f"que cortan persistencia (el plan no se persiste y el KV queda "
            f"'generating' para siempre). Convertir a `logger.info` y continuar — "
            f"el plan huérfano es FEATURE (recovery vía /pending-status)."
        )


# ----- Backend: KV update tras complete -----

def test_kv_updated_with_plan_id_final_on_complete():
    text = _read(_ROUTER)
    # Tras `_postprocess_pipeline_result`, debe haber un upsert con
    # status="complete" y plan_id_final.
    assert 'status="complete"' in text and "plan_id_final=" in text, (
        "Tras persistir el plan, debe haber `upsert_pending_pipeline(..., "
        "status='complete', plan_id_final=...)` para que el frontend que "
        "vuelve sepa qué plan_id usar."
    )


def test_kv_updated_with_failed_on_exception():
    text = _read(_ROUTER)
    assert 'status="failed"' in text, (
        "Si el pipeline lanza, debe marcar `status='failed'` en KV para "
        "que el frontend muestre toast de error en lugar de loading infinito."
    )


# ----- Frontend: localStorage flag set en Plan.jsx -----

def test_plan_jsx_sets_localstorage_flag():
    text = _read(_PLAN)
    assert "'mealfit_plan_in_progress'" in text, (
        "Plan.jsx debe `localStorage.setItem('mealfit_plan_in_progress', ...)` "
        "ANTES de empezar el SSE stream. Sin esto, el boot hook al volver "
        "no detecta que había un plan generándose."
    )


# ----- Frontend: PendingPipelineRecovery component -----

def test_recovery_component_exists():
    assert _RECOVERY.exists(), (
        "Falta `frontend/src/components/PendingPipelineRecovery.jsx`."
    )


def test_recovery_polls_pending_status_endpoint():
    text = _read(_RECOVERY)
    assert "/api/plans/pending-status" in text
    assert "/api/plans/pending-status/ack" in text


def test_recovery_kill_switch():
    text = _read(_RECOVERY)
    assert "VITE_DEEP_SEARCH_RECOVERY" in text, (
        "El componente debe respetar `VITE_DEEP_SEARCH_RECOVERY=false` como kill switch."
    )


def test_recovery_handles_3_statuses():
    """Component debe manejar 3 status estados: complete (redirect + toast),
    failed (toast error), generating (redirect a /plan si no estamos ahí)."""
    text = _read(_RECOVERY)
    for st in ("'complete'", "'failed'", "'generating'"):
        assert st in text, f"Recovery component debe manejar status={st}."


# ----- Frontend: App.jsx monta el componente -----

def test_app_jsx_mounts_recovery():
    text = _read(_APP)
    assert "PendingPipelineRecovery" in text, (
        "App.jsx debe importar y montar `<PendingPipelineRecovery />`."
    )
    # Debe estar DENTRO del Router (para que useLocation/useNavigate funcionen).
    router_start = text.find("<Router>")
    recovery_mount = text.find("<PendingPipelineRecovery")
    router_end = text.find("</Router>")
    assert router_start > 0 and recovery_mount > router_start and recovery_mount < router_end, (
        "`<PendingPipelineRecovery />` debe estar montado DENTRO de `<Router>`."
    )
