"""[P1-GUEST-INFLIGHT-GUARD · 2026-07-09] Los guests (sin actual_user_id) NO pasaban
por el guard de "pipeline activo" (gateado en `_deep_search_user_id`, que es None para
guests). Un guest que recargaba la pestaña mid-generación disparaba un 2º pipeline
completo = doble gasto DeepSeek. Fix: espejo del guard autenticado, keyed en el
session_id del guest (misma KV `pending_pipeline:<sid>`), ANTES de registrar/arrancar
el pipeline → 409 `pipeline_already_running`.

Parser-based (mismo patrón que test_p1_dedup_recent_plan / test_p1_guest_plan_recovery):
ancla el contrato en el source. El dedup POST-completion (get_latest_meal_plan_with_id)
NO aplica a guests (no persisten en meal_plans); solo el guard in-flight (KV).
"""
import os

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def _plans_src():
    return _read(_BACKEND, "routers", "plans.py")


def _guest_block(src):
    """Segmento del bloque guest: desde el marker del guard hasta el fin del bloque
    (el persist global de update_reason que le sigue)."""
    start = src.find("P1-GUEST-INFLIGHT-GUARD")
    # fallback: si aún no existe el marker, devolver el bloque P1-GUEST-PLAN-RECOVERY del router
    if start < 0:
        start = src.find("_guest_recovery_sid = (")
    end = src.find("[P0 FIX GAP 1] Persistir update_reason", start)
    return src[start: end if end > start else start + 2000]


def test_marker_present():
    assert "P1-GUEST-INFLIGHT-GUARD" in _plans_src(), "falta el tooltip-anchor P1-GUEST-INFLIGHT-GUARD"


def test_guest_block_checks_active_pipeline():
    seg = _guest_block(_plans_src())
    assert "check_user_has_active_pipeline(_guest_recovery_sid" in seg, (
        "el bloque guest debe chequear pipeline activo keyed en el session_id del guest"
    )


def test_guest_guard_returns_409_pipeline_already_running():
    seg = _guest_block(_plans_src())
    assert '"code": "pipeline_already_running"' in seg, (
        "el guard guest debe devolver el mismo 409 pipeline_already_running que el autenticado"
    )


def test_guest_409_propagates_not_swallowed():
    """El 409 debe re-lanzarse (except HTTPException: raise) para que el best-effort
    `except Exception` del registro guest no lo trague."""
    seg = _guest_block(_plans_src())
    assert "except HTTPException:" in seg and "raise" in seg, (
        "el bloque guest debe re-lanzar HTTPException (si no, el except Exception traga el 409)"
    )


def test_guest_guard_runs_before_upsert():
    """El check debe ir ANTES del upsert a 'generating' (si no, deja una KV stale)."""
    seg = _guest_block(_plans_src())
    pos_check = seg.find("check_user_has_active_pipeline(_guest_recovery_sid")
    pos_upsert = seg.find('_upp_guest(_guest_recovery_sid, status="generating")')
    assert pos_check >= 0 and pos_upsert >= 0, "faltan el check o el upsert en el bloque guest"
    assert pos_check < pos_upsert, "el check de pipeline activo debe ir antes del upsert a 'generating'"
