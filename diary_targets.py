"""[P1-DIARY-DAY-TARGETS] Resolución y registro histórico de metas calóricas por día.

El diario (`consumed-range` y `consumed/{user_id}`) devolvía únicamente los totales
consumidos sin la meta vigente en cada fecha. Al cambiar de plan o de objetivo
(p. ej. bajar de ~3350 kcal a 2050 kcal), la interfaz aplicaba la meta actual
retroactivamente a todos los días pasados y marcaba días cumplidos como excedidos.

Este módulo resuelve la meta efectiva de cada fecha combinando:
1. Instantáneas diarias en `app_kv_store` (`diary_targets_hist:<user_id>`) —
   capturan cambios de perfil/calculadora y planes sin requerir migración DDL.
2. Historial de planes en `meal_plans` ordenados cronológicamente — permite
   reconstruir retrospectivamente la meta vigente en fechas anteriores.
"""

from __future__ import annotations

import json
import logging
import re
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

KV_PREFIX = "diary_targets_hist:"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_MAX_HISTORY_DAYS = 400


def _parse_positive_int(raw: Any) -> Optional[int]:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = int(round(float(raw)))
        return val if val > 0 else None
    text = str(raw).replace(",", "").strip()
    if not text:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        val = int(round(float(m.group(0))))
        return val if val > 0 else None
    except (TypeError, ValueError):
        return None


def _normalize_macros(raw: Any) -> Optional[Dict[str, int]]:
    if not isinstance(raw, dict):
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                return None
        else:
            return None
    if not isinstance(raw, dict):
        return None

    out: Dict[str, int] = {}
    p = _parse_positive_int(raw.get("protein"))
    c = _parse_positive_int(raw.get("carbs"))
    f = _parse_positive_int(raw.get("fats") if raw.get("fats") is not None else raw.get("healthy_fats"))
    if p is not None:
        out["protein"] = p
    if c is not None:
        out["carbs"] = c
    if f is not None:
        out["fats"] = f
    return out or None


def _normalize_goal(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def _clean_snapshot(
    calories: Any,
    macros: Any = None,
    main_goal: Any = None,
) -> Optional[Dict[str, Any]]:
    cal = _parse_positive_int(calories)
    if cal is None:
        return None
    snap: Dict[str, Any] = {"calories": cal}
    norm_macros = _normalize_macros(macros)
    if norm_macros:
        snap["macros"] = norm_macros
    norm_goal = _normalize_goal(main_goal)
    if norm_goal:
        snap["main_goal"] = norm_goal
    return snap


def resolve_targets_timeline(
    days: Iterable[str],
    plan_rows: Optional[Iterable[Dict[str, Any]]] = None,
    kv_hist: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Calcula la meta vigente para cada fecha `YYYY-MM-DD` sin acceso a DB.

    Cada fecha recibe el último evento de meta (`plan_rows` o `kv_hist`) con
    `fecha_evento <= fecha_consultada`. Si ambos existen en la misma fecha,
    `kv_hist` tiene precedencia (p. ej. cambio de objetivo en perfil sin
    regenerar plan o snapshot explícito). Fechas anteriores al primer evento
    histórico no se inventan (se omiten del diccionario resultante).
    """
    timeline_by_date: Dict[str, Dict[str, Any]] = {}

    for row in plan_rows or ():
        if not isinstance(row, dict):
            continue
        dia = str(row.get("dia") or "")[:10]
        if not _DATE_RE.match(dia):
            continue
        snap = _clean_snapshot(
            row.get("calories"),
            row.get("macros"),
            row.get("main_goal"),
        )
        if snap:
            timeline_by_date[dia] = snap

    raw_days = kv_hist.get("days") if isinstance(kv_hist, dict) and isinstance(kv_hist.get("days"), dict) else kv_hist
    if isinstance(raw_days, dict):
        for k in sorted(raw_days.keys()):
            dia = str(k)[:10]
            if not _DATE_RE.match(dia):
                continue
            val = raw_days[k]
            if not isinstance(val, dict):
                continue
            snap = _clean_snapshot(
                val.get("calories"),
                val.get("macros"),
                val.get("main_goal"),
            )
            if snap:
                timeline_by_date[dia] = snap

    if not timeline_by_date:
        return {}

    event_dates = sorted(timeline_by_date.keys())
    out: Dict[str, Dict[str, Any]] = {}
    for raw_day in days or ():
        day_str = str(raw_day or "")[:10]
        if not _DATE_RE.match(day_str):
            continue
        idx = bisect_right(event_dates, day_str) - 1
        if idx >= 0:
            chosen = timeline_by_date[event_dates[idx]]
            copied: Dict[str, Any] = {"calories": chosen["calories"]}
            if "macros" in chosen:
                copied["macros"] = dict(chosen["macros"])
            if "main_goal" in chosen:
                copied["main_goal"] = chosen["main_goal"]
            out[day_str] = copied
    return out


def _clamp_tz_offset(tz_offset_mins: Optional[int]) -> Optional[int]:
    if tz_offset_mins is None:
        return None
    try:
        return max(-840, min(840, int(tz_offset_mins)))
    except (TypeError, ValueError):
        return None


def resolve_user_effective_off_min(user_id: str, tz_offset_mins: Optional[int] = None) -> int:
    """Devuelve `tz_offset_mins + rollover_hours*60` usando la misma regla que el diario."""
    off = _clamp_tz_offset(tz_offset_mins)
    if off is None and user_id and user_id != "guest":
        try:
            from db_profiles import resolve_user_tz_offset_minutes
            off = _clamp_tz_offset(resolve_user_tz_offset_minutes(user_id))
        except Exception:
            off = None
    if off is None:
        off = 240  # República Dominicana (UTC-4) como último recurso
    rollover_h = 4
    try:
        from db_facts import _diary_rollover_hour
        rollover_h = int(_diary_rollover_hour())
    except Exception:
        rollover_h = 4
    return off + (rollover_h * 60)


def biological_today_str(user_id: str, tz_offset_mins: Optional[int] = None) -> str:
    eff_off = resolve_user_effective_off_min(user_id, tz_offset_mins)
    return (datetime.now(timezone.utc) - timedelta(minutes=eff_off)).date().isoformat()


def _parse_kv_value(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    if not isinstance(raw, dict):
        return {}
    days = raw.get("days") if isinstance(raw.get("days"), dict) else raw
    return {
        str(k)[:10]: v
        for k, v in days.items()
        if _DATE_RE.match(str(k)[:10]) and isinstance(v, dict)
    }


def resolve_targets_by_day(
    user_id: str,
    dates: Iterable[str],
    effective_off_min: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Resuelve las metas calóricas y de macros para una lista de fechas de `user_id`."""
    if not user_id or user_id == "guest":
        return {}
    valid_dates = sorted({str(d)[:10] for d in (dates or ()) if d and _DATE_RE.match(str(d)[:10])})
    if not valid_dates:
        return {}

    try:
        from db_core import connection_pool, execute_sql_query
        if not connection_pool:
            return {}

        kv_row = execute_sql_query(
            "SELECT value FROM app_kv_store WHERE key = %s LIMIT 1",
            (f"{KV_PREFIX}{user_id}",),
            fetch_one=True,
        )
        kv_hist = _parse_kv_value((kv_row or {}).get("value"))

        plan_rows = execute_sql_query(
            """
            SELECT (created_at - make_interval(mins => %s))::date AS dia,
                   plan_data->>'calories' AS calories,
                   plan_data->'macros' AS macros,
                   COALESCE(
                       NULLIF(plan_data->'assessment'->>'mainGoal', ''),
                       NULLIF(plan_data->'userParams'->>'mainGoal', ''),
                       NULLIF(plan_data->>'mainGoal', ''),
                       NULLIF(plan_data->>'goal', '')
                   ) AS main_goal
            FROM meal_plans
            WHERE user_id = %s
              AND COALESCE(plan_data->>'generation_status', '') <> 'failed'
              AND plan_data IS NOT NULL
              AND plan_data ? 'calories'
            ORDER BY created_at ASC
            """,
            (int(effective_off_min or 0), user_id),
            fetch_all=True,
        ) or []

        return resolve_targets_timeline(valid_dates, plan_rows=plan_rows, kv_hist=kv_hist)
    except Exception as e:
        logger.debug(f"[P1-DIARY-DAY-TARGETS] resolve_targets_by_day falló (fail-open): {e}")
        return {}


def record_daily_target_snapshot(
    user_id: str,
    *,
    calories: Any,
    macros: Any = None,
    main_goal: Any = None,
    tz_offset_mins: Optional[int] = None,
    date_str: Optional[str] = None,
    overwrite_today: bool = True,
) -> bool:
    """Guarda de forma idempotente (best-effort) la meta diaria de `user_id`."""
    if not user_id or user_id == "guest":
        return False
    snap = _clean_snapshot(calories, macros, main_goal)
    if not snap:
        return False

    try:
        from psycopg.types.json import Jsonb
        from db_core import connection_pool, execute_sql_query, execute_sql_write
        if not connection_pool:
            return False

        day_str = str(date_str)[:10] if date_str and _DATE_RE.match(str(date_str)[:10]) else biological_today_str(user_id, tz_offset_mins)
        kv_key = f"{KV_PREFIX}{user_id}"

        existing_row = execute_sql_query(
            "SELECT value FROM app_kv_store WHERE key = %s LIMIT 1",
            (kv_key,),
            fetch_one=True,
        )
        days_map = _parse_kv_value((existing_row or {}).get("value"))

        if day_str in days_map:
            if not overwrite_today:
                return True
            if days_map[day_str] == snap:
                return True

        days_map[day_str] = snap
        if len(days_map) > _MAX_HISTORY_DAYS:
            keep_keys = sorted(days_map.keys())[-_MAX_HISTORY_DAYS:]
            days_map = {k: days_map[k] for k in keep_keys}

        execute_sql_write(
            "INSERT INTO app_kv_store (key, value, updated_at) "
            "VALUES (%s, %s, NOW()) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
            (kv_key, Jsonb({"days": days_map})),
        )
        return True
    except Exception as e:
        logger.debug(f"[P1-DIARY-DAY-TARGETS] record_daily_target_snapshot falló (best-effort): {e}")
        return False


def extract_targets_from_plan_data(plan_data: Any, fallback_goal: Any = None) -> Optional[Dict[str, Any]]:
    if not isinstance(plan_data, dict):
        return None
    assessment = plan_data.get("assessment") if isinstance(plan_data.get("assessment"), dict) else {}
    user_params = plan_data.get("userParams") if isinstance(plan_data.get("userParams"), dict) else {}
    goal = (
        assessment.get("mainGoal")
        or user_params.get("mainGoal")
        or plan_data.get("mainGoal")
        or plan_data.get("goal")
        or fallback_goal
    )
    return _clean_snapshot(plan_data.get("calories"), plan_data.get("macros"), goal)


def snapshot_current_targets_for_user(
    user_id: str,
    *,
    overwrite_today: bool = False,
    tz_offset_mins: Optional[int] = None,
) -> bool:
    """Captura best-effort la meta vigente hoy para `user_id` (plan activo o calculadora)."""
    if not user_id or user_id == "guest":
        return False
    try:
        from db_core import connection_pool, execute_sql_query
        if not connection_pool:
            return False

        day_str = biological_today_str(user_id, tz_offset_mins)
        kv_key = f"{KV_PREFIX}{user_id}"

        if not overwrite_today:
            existing_row = execute_sql_query(
                "SELECT value FROM app_kv_store WHERE key = %s LIMIT 1",
                (kv_key,),
                fetch_one=True,
            )
            days_map = _parse_kv_value((existing_row or {}).get("value"))
            if day_str in days_map:
                return True

        snap: Optional[Dict[str, Any]] = None
        plan_row = execute_sql_query(
            """
            SELECT plan_data->>'calories' AS calories,
                   plan_data->'macros' AS macros,
                   COALESCE(
                       NULLIF(plan_data->'assessment'->>'mainGoal', ''),
                       NULLIF(plan_data->'userParams'->>'mainGoal', ''),
                       NULLIF(plan_data->>'mainGoal', ''),
                       NULLIF(plan_data->>'goal', '')
                   ) AS main_goal
            FROM meal_plans
            WHERE user_id = %s
              AND COALESCE(plan_data->>'generation_status', '') <> 'failed'
              AND plan_data IS NOT NULL
              AND plan_data ? 'calories'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
            fetch_one=True,
        )
        if plan_row:
            snap = _clean_snapshot(
                plan_row.get("calories"),
                plan_row.get("macros"),
                plan_row.get("main_goal"),
            )

        if not snap:
            from db_profiles import get_user_profile
            profile = get_user_profile(user_id)
            hp = (profile or {}).get("health_profile") or {}
            req = ("gender", "age", "height", "weight", "weightUnit", "activityLevel", "mainGoal")   # [REVISION-DIARIO-14NOV] sin «medicalConditions»: lista vacía = sin condiciones, y dejaba fuera a 7 de 28 perfiles («weightUnit» se queda: sin ella la calculadora asume libras)
            if all(hp.get(k) for k in req):
                from nutrition_calculator import get_nutrition_targets
                t = get_nutrition_targets(hp)
                m = t.get("macros") or {}
                snap = _clean_snapshot(
                    t.get("target_calories"),
                    {
                        "protein": m.get("protein_g") or m.get("protein_str"),
                        "carbs": m.get("carbs_g") or m.get("carbs_str"),
                        "fats": m.get("fats_g") or m.get("fats_str"),
                    },
                    hp.get("mainGoal"),
                )

        if not snap:
            return False

        return record_daily_target_snapshot(
            user_id,
            calories=snap["calories"],
            macros=snap.get("macros"),
            main_goal=snap.get("main_goal"),
            date_str=day_str,
            overwrite_today=overwrite_today,
        )
    except Exception as e:
        logger.debug(f"[P1-DIARY-DAY-TARGETS] snapshot_current_targets_for_user falló (best-effort): {e}")
        return False


def record_calculator_snapshot_if_tracking_only(user_id: str, health_profile: Any) -> bool:
    """Si el usuario no tiene un plan activo guardado, actualiza la meta de hoy tras editar su perfil."""
    if not user_id or user_id == "guest" or not isinstance(health_profile, dict):
        return False
    req = ("gender", "age", "height", "weight", "weightUnit", "activityLevel", "mainGoal")   # [REVISION-DIARIO-14NOV] sin «medicalConditions»: lista vacía = sin condiciones, y dejaba fuera a 7 de 28 perfiles («weightUnit» se queda: sin ella la calculadora asume libras)
    if not all(health_profile.get(k) for k in req):
        return False
    try:
        from db_core import connection_pool, execute_sql_query
        if not connection_pool:
            return False
        has_plan = execute_sql_query(
            "SELECT 1 FROM meal_plans WHERE user_id = %s AND COALESCE(plan_data->>'generation_status', '') <> 'failed' "
            "AND plan_data IS NOT NULL AND plan_data ? 'calories' LIMIT 1",
            (user_id,),
            fetch_one=True,
        )
        if has_plan:
            return False
        from nutrition_calculator import get_nutrition_targets
        t = get_nutrition_targets(health_profile)
        m = t.get("macros") or {}
        tz_raw = health_profile.get("tz_offset_minutes")
        if tz_raw is None:
            tz_raw = health_profile.get("tzOffset")
        return record_daily_target_snapshot(
            user_id,
            calories=t.get("target_calories"),
            macros={
                "protein": m.get("protein_g") or m.get("protein_str"),
                "carbs": m.get("carbs_g") or m.get("carbs_str"),
                "fats": m.get("fats_g") or m.get("fats_str"),
            },
            main_goal=health_profile.get("mainGoal"),
            tz_offset_mins=tz_raw,
            overwrite_today=True,
        )
    except Exception as e:
        logger.debug(f"[P1-DIARY-DAY-TARGETS] record_calculator_snapshot_if_tracking_only falló: {e}")
        return False
