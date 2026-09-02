"""[P1-ARQ25-F1-LIFECYCLE · 2026-09-02] Runs de generación durables (roadmap 2.5 §5.6).

    POST /api/plans/generation-runs              crea o REPRODUCE un run (I9, `idempotency_key`)
    GET  /api/plans/generation-runs/{run_id}     snapshot derivado (status, availability, chunks)
    GET  /api/plans/generation-runs/{run_id}/events   SSE que TAILEA DB + KV de progreso
    POST /api/plans/generation-runs/{run_id}/cancel   cancelación cooperativa

Todo detrás de `MEALFIT_INITIAL_VIA_QUEUE` (default OFF ⇒ 404 en el POST; el resto sigue
respondiendo para runs que ya existan). Guests: 400 → siguen por `/analyze/stream`
(decisión #3 del roadmap §16, opción recomendada).

Cuota: `verify_api_quota` en el POST, como el SSE. El coste LLM lo anota el postprocess
(`log_api_usage`) cuando el worker termina, igual que hoy. Los GET de polling van con
`get_verified_user_id` + RateLimiter (misma regla que la tabla «Historial-quota-exemption»
de CLAUDE.md: leer el estado de tu propio run no puede costar un crédito).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from auth import get_verified_user_id, verify_api_quota
from knobs import _env_int
from rate_limiter import RateLimiter
# El MISMO limitador que `/analyze/stream` (3/min): la cola no abre una segunda puerta.
# `routers.plans` no importa este módulo, así que no hay ciclo.
from routers.plans import _PLAN_GEN_LIMITER

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/plans/generation-runs", tags=["plans-generation"])

#: Polling del snapshot: 60/min por usuario (el cliente sondea cada 2-3 s durante la generación).
_RUN_STATUS_LIMITER = RateLimiter(max_calls=60, period_seconds=60)
_RUN_CANCEL_LIMITER = RateLimiter(max_calls=10, period_seconds=60)


def _require_user(verified_user_id: Optional[str]) -> str:
    if not verified_user_id:
        raise HTTPException(status_code=401, detail={"code": "auth_required"})
    return str(verified_user_id)


@router.post("")
async def api_create_generation_run(
    request: Request,
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(verify_api_quota),
    _rl: None = Depends(_PLAN_GEN_LIMITER),
):
    from generation_lifecycle import (
        RunFingerprintConflict, create_or_replay_run, create_placeholder_plan_and_enqueue_initial,
        initial_via_queue_enabled, load_run_snapshot, request_fingerprint,
    )
    from generation_inputs import build_initial_pipeline_inputs, validate_generation_request

    user_id = data.get("user_id")
    if not user_id or user_id == "guest":
        raise HTTPException(status_code=400, detail={
            "code": "guest_not_supported_use_stream",
            "message": "Los invitados generan por /api/plans/analyze/stream.",
        })
    if verified_user_id != user_id:
        raise HTTPException(status_code=401, detail={"code": "user_mismatch"})
    # Knob global o allowlist por usuario (canary): 404 ⇒ el cliente cae al SSE legacy.
    if not initial_via_queue_enabled(user_id):
        raise HTTPException(status_code=404, detail={"code": "initial_via_queue_disabled"})

    idempotency_key = (request.headers.get("Idempotency-Key") or data.get("idempotency_key") or "").strip()
    if not idempotency_key or len(idempotency_key) > 128:
        raise HTTPException(status_code=422, detail={
            "code": "idempotency_key_required",
            "message": "Envía `idempotency_key` (≤128 chars) para que un reintento no cree dos planes.",
        })
    session_id = data.get("session_id")

    validate_generation_request(data, verified_user_id)
    fingerprint = request_fingerprint(data)

    from correlation import get_correlation_id
    try:
        run, created = await asyncio.to_thread(
            create_or_replay_run,
            user_id=user_id, idempotency_key=idempotency_key, fingerprint=fingerprint,
            requested_days=int(data.get("totalDays", 3)),
            market_country=(str(data.get("country")) if data.get("country") else None),
            locale=(str(data.get("locale")) if data.get("locale") else None),
            input_snapshot={"keys": sorted(k for k in data.keys() if not str(k).startswith("_"))},
            correlation_id=get_correlation_id(),
        )
    except RunFingerprintConflict as e:
        raise HTTPException(status_code=409, detail={
            "code": "idempotency_key_conflict", "run_id": e.run_id,
            "message": "Esa clave ya se usó con otro formulario. Genera una clave nueva.",
        })

    if not created:
        snap = await asyncio.to_thread(load_run_snapshot, str(run["id"]), user_id)
        return {"replayed": True, **(snap or {"run_id": str(run["id"]), "plan_id": None, "status": "PENDING"})}

    # Guard legacy: un pipeline SSE vivo del mismo usuario no puede convivir con la cola.
    try:
        from db_plans import check_user_has_active_pipeline
        _active = await asyncio.to_thread(check_user_has_active_pipeline, user_id, 15)
    except Exception:
        _active = None
    if _active:
        raise HTTPException(status_code=409, detail={
            "code": "pipeline_already_running", "started_at": _active.get("started_at"),
            "message": "Ya tienes un plan generándose. Espera a que termine.",
        })

    inputs = await asyncio.to_thread(build_initial_pipeline_inputs, data, user_id, session_id)
    snapshot = {
        "form_data": inputs["pipeline_data"],
        "raw_data": data,
        "history": inputs["history"],
        "taste_profile": inputs["taste_profile"],
        "memory_context": inputs["memory_ctx"],
        "rejected_meal_names": inputs["rejected_meal_names"],
        "session_id": session_id,
        "totalDays": inputs["total_days_requested"],
        "use_chunking": inputs["use_chunking"],
    }
    try:
        plan_id, chunk_id = await asyncio.to_thread(
            create_placeholder_plan_and_enqueue_initial,
            user_id=user_id, run_id=str(run["id"]), snapshot=snapshot,
            total_days_requested=inputs["total_days_requested"], days_count=inputs["days_count"],
        )
    except Exception as e:
        logger.exception(f"[ARQ25-F1] no se pudo encolar el chunk 0 run={str(run['id'])[:8]}: {e}")
        from generation_lifecycle import mark_run_error
        await asyncio.to_thread(mark_run_error, str(run["id"]), "enqueue_failed", str(e)[:200], completed=True)
        raise HTTPException(status_code=503, detail={"code": "enqueue_failed", "message": "No pudimos programar la generación. Inténtalo de nuevo."})

    try:  # compat con PendingPipelineRecovery (KV legacy)
        from db_plans import upsert_pending_pipeline
        await asyncio.to_thread(upsert_pending_pipeline, user_id, "generating")
    except Exception:
        pass

    return {
        "replayed": False, "run_id": str(run["id"]), "plan_id": plan_id, "chunk_id": chunk_id,
        "status": "PENDING", "availability": "NONE", "requested_days": inputs["total_days_requested"],
    }


@router.get("/{run_id}")
async def api_get_generation_run(
    run_id: str,
    include_plan: bool = False,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_RUN_STATUS_LIMITER),
):
    from generation_lifecycle import load_run_snapshot
    user_id = _require_user(verified_user_id)
    snap = await asyncio.to_thread(load_run_snapshot, run_id, user_id, include_plan=include_plan)
    if snap is None:
        raise HTTPException(status_code=404, detail={"code": "run_not_found"})
    return snap


@router.post("/{run_id}/cancel")
async def api_cancel_generation_run(
    run_id: str,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_RUN_CANCEL_LIMITER),
):
    from generation_lifecycle import load_run_snapshot, request_run_cancel
    user_id = _require_user(verified_user_id)
    snap = await asyncio.to_thread(load_run_snapshot, run_id, user_id)
    if snap is None:
        raise HTTPException(status_code=404, detail={"code": "run_not_found"})
    res = await asyncio.to_thread(request_run_cancel, run_id, user_id)
    try:
        from db_plans import clear_pending_pipeline
        await asyncio.to_thread(clear_pending_pipeline, user_id)
    except Exception:
        pass
    return {**res, "status": "CANCEL_REQUESTED"}


def _sse(event: str, data: Any) -> str:
    return f"data: {json.dumps({'event': event, 'data': data}, ensure_ascii=False, default=str)}\n\n"


@router.get("/{run_id}/events")
async def api_generation_run_events(
    run_id: str,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """SSE que tailea el estado durable. Transporte, no autoridad: si el cliente se cae y
    vuelve, recibe el mismo `complete` desde DB. Emite el `complete` con el plan en cuanto
    el chunk 0 rellenó el placeholder (availability ≠ NONE), no al final de la cola."""
    from generation_lifecycle import (
        AVAIL_NONE, RUN_CANCELLED, RUN_FAILED, load_run_snapshot,
    )
    user_id = _require_user(verified_user_id)
    first = await asyncio.to_thread(load_run_snapshot, run_id, user_id)
    if first is None:
        raise HTTPException(status_code=404, detail={"code": "run_not_found"})

    poll_s = max(0.5, _env_int("MEALFIT_RUN_EVENTS_POLL_MS", 2000, validator=lambda v: 250 <= v <= 30000) / 1000.0)
    max_s = _env_int("MEALFIT_RUN_EVENTS_MAX_S", 1500, validator=lambda v: 60 <= v <= 7200)
    heartbeat_s = 15.0

    async def _gen():
        last_seq = -1
        last_status = None
        started = time.monotonic()
        last_hb = started
        while True:
            snap = await asyncio.to_thread(load_run_snapshot, run_id, user_id, include_plan=True)
            if snap is None:
                yield _sse("error", {"code": "run_not_found", "message": "El run ya no existe."})
                return
            prog = snap.get("progress") or {}
            seq = int(prog.get("seq") or -1)
            if seq > last_seq and isinstance(prog.get("event"), dict):
                last_seq = seq
                ev = prog["event"]
                ev_name = str(ev.get("event") or "progress")
                if ev_name not in ("complete", "error"):
                    yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
            if snap.get("status") != last_status:
                last_status = snap.get("status")
                yield _sse("status", {k: snap.get(k) for k in ("run_id", "plan_id", "status", "availability", "days_generated", "revision")})
            if snap.get("availability") != AVAIL_NONE and snap.get("plan"):
                yield _sse("complete", snap["plan"])
                return
            if snap.get("status") in (RUN_FAILED, RUN_CANCELLED):
                ev = (prog.get("event") or {}) if isinstance(prog.get("event"), dict) else {}
                data = ev.get("data") if ev.get("event") == "error" and isinstance(ev.get("data"), dict) else {}
                yield _sse("error", {
                    "code": data.get("code") or snap.get("error_code") or ("cancelled" if snap.get("status") == RUN_CANCELLED else "generation_failed"),
                    "message": data.get("message") or snap.get("error_message") or "La generación no pudo completarse.",
                })
                return
            now = time.monotonic()
            if now - started > max_s:
                yield _sse("error", {"code": "events_timeout", "message": "Sigue generándose; vuelve a consultar el estado."})
                return
            if now - last_hb >= heartbeat_s:
                last_hb = now
                yield _sse("heartbeat", {"t": int(now - started)})
            await asyncio.sleep(poll_s)

    return StreamingResponse(_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })
