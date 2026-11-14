# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1001 · G-06 · 2026-11-14] Visibilidad de modelos efectivos y alerta de desvío al arranque.

Cierra el gap G-06 del plan de arquitectura del coach:
- Expone en `/health/version` qué modelos efectivos resuelven hoy coach, respaldo del coach y visión.
- Al arrancar el proceso (`lifespan`), compara la terna efectiva contra la guardada en
  `app_kv_store["effective_models_snapshot"]`. Si cambia entre dos arranques (p. ej. si se pierde
  `MEALFIT_CHAT_AGENT_MODEL` en `.env` y el coach cae al default), registra WARNING en logs y emite una
  alerta en `system_alerts` (`alert_key='model_effective_changed'`, `alert_type='model_drift'`).
- Auto-registra su knob via `_env_bool` siguiendo la convención P3-NEW-D.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from knobs import _env_bool

logger = logging.getLogger("graph_orchestrator")

_KV_EFFECTIVE_MODELS_KEY = "effective_models_snapshot"
_ALERT_KEY_MODEL_DRIFT = "model_effective_changed"
_MONITORED_KEYS = ("coach_model", "coach_fallback_model", "vision_model")

_state_lock = threading.Lock()
_LAST_STARTUP_DRIFT: Optional[dict[str, Any]] = None


def model_drift_alert_enabled() -> bool:
    """Knob `MEALFIT_MODEL_DRIFT_ALERT_ENABLED` (default True)."""
    return _env_bool("MEALFIT_MODEL_DRIFT_ALERT_ENABLED", True)


def get_effective_models_snapshot() -> dict[str, Optional[str]]:
    """Resuelve los modelos efectivos actuales de coach, respaldo y visión (best-effort)."""
    coach_model: Optional[str] = None
    coach_fallback_model: Optional[str] = None
    vision_model: Optional[str] = None
    vision_provider: Optional[str] = None

    try:
        from agent import _chat_agent_fallback_model, _chat_agent_model_name

        coach_model = _chat_agent_model_name(None) or None
        coach_fallback_model = _chat_agent_fallback_model(coach_model or "") or None
    except Exception as exc:
        logger.debug("[G-06] snapshot coach_model falló (best-effort): %s", type(exc).__name__)

    try:
        from vision_agent import _vision_model_name, _vision_provider

        vision_model = _vision_model_name() or None
        vision_provider = _vision_provider() or None
    except Exception as exc:
        logger.debug("[G-06] snapshot vision_model falló (best-effort): %s", type(exc).__name__)

    return {
        "coach_model": coach_model,
        "coach_fallback_model": coach_fallback_model,
        "vision_model": vision_model,
        "vision_provider": vision_provider,
    }


def compute_model_drift(
    previous: Optional[dict[str, Any]],
    current: dict[str, Optional[str]],
) -> dict[str, dict[str, Optional[str]]]:
    """Devuelve `{campo: {"previous": ..., "current": ...}}` para cambios entre arranques."""
    if not isinstance(previous, dict) or not previous:
        return {}
    if not any(k in previous for k in _MONITORED_KEYS):
        return {}
    changed: dict[str, dict[str, Optional[str]]] = {}
    for key in _MONITORED_KEYS:
        prev_val = previous.get(key) or None
        curr_val = current.get(key) or None
        if prev_val != curr_val:
            changed[key] = {"previous": prev_val, "current": curr_val}
    return changed


def _parse_kv_snapshot(raw: Any) -> Optional[dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def detect_and_record_model_drift_on_startup() -> dict[str, Any]:
    """Compara la configuración de modelos contra el arranque previo y alerta si hubo cambios.

    Best-effort: jamás bloquea ni tumba el arranque del servidor si PostgreSQL no está disponible.
    """
    global _LAST_STARTUP_DRIFT

    current = get_effective_models_snapshot()
    result: dict[str, Any] = {
        "checked": False,
        "drift_detected": False,
        "changed": {},
        "current": current,
    }

    try:
        from db_core import connection_pool, execute_sql_query, execute_sql_write
    except Exception:
        return result

    if connection_pool is None:
        return result

    previous: Optional[dict[str, Any]] = None
    try:
        row = execute_sql_query(
            "SELECT value FROM app_kv_store WHERE key = %s LIMIT 1",
            (_KV_EFFECTIVE_MODELS_KEY,),
            fetch_one=True,
        )
        if row:
            previous = _parse_kv_snapshot(row.get("value"))
        result["checked"] = True
    except Exception as exc:
        logger.debug("[G-06] lectura previa en app_kv_store falló: %s", type(exc).__name__)
        return result

    changed = compute_model_drift(previous, current)
    now_utc = datetime.now(timezone.utc)

    if changed:
        result["drift_detected"] = True
        result["changed"] = changed
        diff_parts = [
            f"{field}: {vals['previous'] or '(ninguno)'} -> {vals['current'] or '(ninguno)'}"
            for field, vals in changed.items()
        ]
        summary = "; ".join(diff_parts)
        logger.warning(
            "🚨 [G-06 MODEL_DRIFT] Modelo efectivo cambió entre arranques: %s",
            summary,
        )
        if model_drift_alert_enabled():
            try:
                execute_sql_write(
                    """
                    INSERT INTO system_alerts
                        (alert_key, alert_type, severity, title, message, metadata, triggered_at, resolved_at)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, NULL)
                    ON CONFLICT (alert_key) DO UPDATE
                    SET alert_type = EXCLUDED.alert_type,
                        severity = EXCLUDED.severity,
                        title = EXCLUDED.title,
                        message = EXCLUDED.message,
                        metadata = EXCLUDED.metadata,
                        triggered_at = EXCLUDED.triggered_at,
                        resolved_at = NULL
                    """,
                    (
                        _ALERT_KEY_MODEL_DRIFT,
                        "model_drift",
                        "warning",
                        "Modelo efectivo de IA cambió entre arranques",
                        summary[:500],
                        json.dumps(
                            {
                                "changed": changed,
                                "previous": previous,
                                "current": current,
                                "detected_by": "startup_model_drift_check",
                            },
                            ensure_ascii=False,
                        ),
                        now_utc,
                    ),
                )
            except Exception as alert_exc:
                logger.warning(
                    "[G-06 MODEL_DRIFT] no se pudo persistir system_alerts: %s",
                    type(alert_exc).__name__,
                )
    else:
        logger.info(
            "✅ [G-06] Modelos efectivos sin cambios entre arranques "
            "(coach=%s, fallback=%s, vision=%s).",
            current.get("coach_model"),
            current.get("coach_fallback_model"),
            current.get("vision_model"),
        )

    payload = {
        **current,
        "recorded_at": now_utc.isoformat(),
    }
    try:
        execute_sql_write(
            """
            INSERT INTO app_kv_store (key, value, updated_at)
            VALUES (%s, %s::jsonb, %s)
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value,
                updated_at = EXCLUDED.updated_at
            """,
            (
                _KV_EFFECTIVE_MODELS_KEY,
                json.dumps(payload, ensure_ascii=False),
                now_utc,
            ),
        )
    except Exception as kv_exc:
        logger.debug("[G-06] persistencia en app_kv_store falló: %s", type(kv_exc).__name__)

    with _state_lock:
        _LAST_STARTUP_DRIFT = dict(result)
    return result


def get_health_version_model_fields() -> dict[str, Any]:
    """Devuelve las claves de G-06 para `/health/version` sin romper jamás el 200 OK."""
    current = get_effective_models_snapshot()
    with _state_lock:
        last_drift = dict(_LAST_STARTUP_DRIFT) if _LAST_STARTUP_DRIFT is not None else None

    drift_detected = bool(last_drift and last_drift.get("drift_detected"))
    # Si algún env knob cambió en caliente o antes del check de startup, reflejar también
    # divergencia respecto al snapshot guardado en startup si existiera.
    if not drift_detected and last_drift and isinstance(last_drift.get("current"), dict):
        if compute_model_drift(last_drift.get("current"), current):
            drift_detected = True

    return {
        "coach_model": current.get("coach_model"),
        "coach_fallback_model": current.get("coach_fallback_model"),
        "vision_model": current.get("vision_model"),
        "model_drift_detected": drift_detected,
    }
