"""[P1-ARQ25-F1-LIFECYCLE · 2026-09-02] Entradas del pipeline del Bloque 1, compartidas.

El endpoint SSE legacy (`routers/plans.py::api_analyze_stream`) construye `pipeline_data`
inline a lo largo de ~300 líneas. El endpoint de la cola (`routers/plans_generation.py`)
necesita EXACTAMENTE las mismas entradas para que el plan generado por la cola sea
indistinguible del generado por SSE. Este módulo las construye llamando a los MISMOS
helpers que usa el SSE (importados de `routers.plans` en tiempo de llamada, así los
tests parser-based que anclan esos helpers siguen mirando el mismo sitio).

Deuda declarada (roadmap 2.5 §10.3, Fase 9): los dos bloques de inyección server-side
(`weight_history`/check-ins y «desde mi Nevera») siguen inline en el SSE y aquí se
reproducen con la misma lógica. Cuando el SSE legacy se retire (Fase 9), esta será la
única copia.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _rp():
    """`routers.plans` en tiempo de llamada (evita import circular en el arranque)."""
    from routers import plans as _plans
    return _plans


def validate_generation_request(data: dict, verified_user_id: Optional[str]) -> None:
    """Las mismas 9 comprobaciones pre-pipeline del SSE, en el mismo orden y con los
    mismos códigos de error. Muta `data` igual que el SSE (cierre de texto libre médico,
    país desde el perfil, saneo de básicos)."""
    rp = _rp()
    rp._close_medical_freetext_scope(data)
    rp._hydrate_country_from_profile_for_submit(data, verified_user_id)
    if "country" in data:
        rp._assert_supported_country_for_request(data.get("country"))
    _ok, _missing = rp._validate_form_data_min(data)
    if not _ok:
        raise HTTPException(status_code=422, detail={
            "code": "missing_required_fields", "missing_fields": _missing,
            "message": (f"Faltan campos críticos para generar tu plan: {', '.join(_missing)}. "
                        f"Completa el formulario antes de continuar."),
        })
    if rp._has_out_of_scope_clinical_declaration(data):
        raise HTTPException(status_code=422, detail={
            "code": "clinical_scope_exceeded",
            "message": ("Todavía no podemos calcular un plan seguro para una condición o "
                        "medicamento fuera de nuestra lista clínica verificada. Preferimos "
                        "decírtelo antes que entregarte un plan que parezca calculado y no lo esté."),
        })
    _mc_ok, _mc_count, _mc_cap = rp._validate_medical_conditions_cap(data)
    if not _mc_ok:
        raise HTTPException(status_code=422, detail={
            "code": "too_many_medical_conditions", "max": _mc_cap,
            "message": ("Para garantizar la calidad clínica del plan, selecciona máximo "
                        f"{_mc_cap} condiciones prioritarias."),
        })
    rp._sanitize_staple_foods_for_generation(data)
    rp._log_dislikes_signal_loss(data)
    _ok_td, _td_err = rp._validate_total_days(data)
    if not _ok_td:
        raise HTTPException(status_code=422, detail={
            "code": "invalid_total_days", "errors": [_td_err],
            "message": (f"`totalDays` inválido: {(_td_err or {}).get('reason', 'rango/tipo')} "
                        f"(recibido: {(_td_err or {}).get('value')!r}, aceptado: "
                        f"enteros en {(_td_err or {}).get('accepted_range')})."),
        })
    _ok, _bio_errors = rp._validate_form_data_ranges(data)
    if not _ok:
        raise HTTPException(status_code=422, detail={
            "code": "invalid_biometric_range", "errors": _bio_errors,
            "message": ("Algunos datos biométricos están fuera del rango aceptado: "
                        + ", ".join(e["field"] for e in _bio_errors)),
        })
    try:
        from nutrition_calculator import validate_budget_sufficient as _vbs
        _budget_ok, _budget_detail = _vbs(data)
    except Exception:
        _budget_ok, _budget_detail = True, None
    if not _budget_ok:
        raise HTTPException(status_code=422, detail={"code": "budget_insufficient", **(_budget_detail or {})})


def inject_adaptive_renewal_signals(pipeline_data: dict, actual_user_id: Optional[str]) -> None:
    """Réplica de [P1-ADAPTIVE-RENEWAL] + [P1-CHECKIN-SIGNALS-GATE] del SSE."""
    if not (actual_user_id and os.environ.get("MEALFIT_ADAPTIVE_RENEWAL_INJECT", "true").strip().lower() in ("1", "true", "yes", "on")):
        return
    try:
        from db_core import execute_sql_query
        hp = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (actual_user_id,), fetch_one=True) or {}
        hp = hp.get("health_profile") or {}
        if isinstance(hp, str):
            hp = json.loads(hp)
        wh = hp.get("weight_history") or []
        if isinstance(wh, list) and len(wh) >= 2:
            pipeline_data["weight_history"] = wh
            ck = hp.get("_renewal_checkins") or []
            if isinstance(ck, list) and ck:
                last = ck[-1] if isinstance(ck[-1], dict) else {}
                sig = {k: last.get(k) for k in ("hunger", "energy", "adherence_pct")}
                if any(v is not None for v in sig.values()):
                    pipeline_data["_renewal_signals"] = sig
    except Exception as e:
        logger.debug(f"[ARQ25-F1/ADAPTIVE-RENEWAL] inyección no-op: {type(e).__name__}: {e}")


def inject_pantry_first_ingredients(pipeline_data: dict, actual_user_id: Optional[str], data: dict) -> None:
    """Réplica de [P1-PANTRY-FIRST-PLAN] del SSE."""
    if not (actual_user_id and str(data.get("planSource") or "").strip().lower() == "pantry"
            and os.environ.get("MEALFIT_PANTRY_FIRST_MODE", "true").strip().lower() in ("1", "true", "yes", "on")):
        return
    try:
        from db_core import execute_sql_query
        inv = execute_sql_query(
            "SELECT ingredient_name, quantity::float8 AS quantity, unit FROM user_inventory "
            "WHERE user_id = %s AND quantity > 0", (actual_user_id,), fetch_all=True,
        ) or []
        items = []
        for it in inv:
            nm = str(it.get("ingredient_name") or "").strip()
            if not nm:
                continue
            q = it.get("quantity") or 0
            u = str(it.get("unit") or "").strip()
            qs = "%g" % float(q)
            items.append(f"{qs} {nm}" if u.lower().startswith("unidad") else f"{qs} {u} de {nm}")
        if items:
            pipeline_data["current_pantry_ingredients"] = items
    except Exception as e:
        logger.debug(f"[ARQ25-F1/PANTRY-FIRST] inyección no-op: {type(e).__name__}: {e}")


def local_start_date_iso(tz_offset_mins: int) -> str:
    """Medianoche local del usuario expresada en UTC, igual que el SSE ([P1-1])."""
    now_utc = datetime.now(timezone.utc)
    local_time = now_utc - timedelta(minutes=tz_offset_mins)
    local_midnight = local_time.replace(hour=0, minute=0, second=0, microsecond=0)
    return (local_midnight + timedelta(minutes=tz_offset_mins)).isoformat()


def build_initial_pipeline_inputs(data: dict, actual_user_id: str, session_id: Optional[str]) -> dict[str, Any]:
    """Todo lo que `run_plan_pipeline` + postprocess necesitan, serializable a `pipeline_snapshot`."""
    rp = _rp()
    from constants import PLAN_CHUNK_SIZE, split_with_absorb

    history: list = []
    likes: list = []
    memory: dict = {}
    if session_id:
        rp.get_or_create_session(session_id)
        memory = rp.build_memory_context(session_id)
        history = memory.get("recent_messages") or []
    if actual_user_id:
        likes = rp.get_user_likes(actual_user_id)
    active_rejections = rp.get_active_rejections(user_id=actual_user_id, session_id=session_id)
    rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []
    taste_profile = rp.analyze_preferences_agent(likes, history, active_rejections=rejected_meal_names)

    total_days_requested = int(data.get("totalDays", 3))
    user_has_profile = bool(actual_user_id and rp._user_has_profile(actual_user_id))
    use_chunking = bool(user_has_profile and total_days_requested > PLAN_CHUNK_SIZE)

    pipeline_data = dict(data)
    rp._strip_untrusted_internal_keys(pipeline_data, allow_set=None, log_prefix="ROUTER /generation-runs")
    rp._merge_other_text_fields(pipeline_data)
    inject_adaptive_renewal_signals(pipeline_data, actual_user_id)
    inject_pantry_first_ingredients(pipeline_data, actual_user_id, data)

    tz_offset_mins = rp._resolve_request_tz_offset(data.get("tzOffset"), actual_user_id)
    start_date_iso = local_start_date_iso(tz_offset_mins)
    pipeline_data["_plan_start_date"] = start_date_iso
    pipeline_data["tz_offset_minutes"] = tz_offset_mins
    if use_chunking or total_days_requested > PLAN_CHUNK_SIZE:
        pipeline_data["_days_to_generate"] = PLAN_CHUNK_SIZE
    rp._enforce_days_to_generate_cap(pipeline_data, log_prefix="ROUTER /generation-runs")
    if actual_user_id:
        from cron_tasks import inject_learning_signals_from_profile
        inject_learning_signals_from_profile(actual_user_id, pipeline_data)

    if use_chunking:
        days_count = int(split_with_absorb(total_days_requested, PLAN_CHUNK_SIZE)[0])
    else:
        days_count = int(pipeline_data.get("_days_to_generate") or total_days_requested)

    memory_ctx = memory.get("full_context_str", "") if session_id else ""
    return {
        "pipeline_data": pipeline_data,
        "history": history,
        "taste_profile": taste_profile,
        "memory_ctx": memory_ctx,
        "rejected_meal_names": rejected_meal_names,
        "total_days_requested": total_days_requested,
        "use_chunking": use_chunking,
        "tz_offset_mins": tz_offset_mins,
        "start_date_iso": start_date_iso,
        "days_count": max(1, days_count),
    }
