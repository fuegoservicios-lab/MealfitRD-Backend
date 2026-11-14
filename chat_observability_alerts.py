# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1007 · G-18 · 2026-11-14] Alertas automáticas de degradación del coach (tasa de respaldo y feedback).

Ejecución sin costo de IA (SQL puro) invocable dentro de `_llm_daily_spend_alert_job` (sin crear un job ID nuevo):
1. `chat_reliability_degraded`: vigila tasa de uso de modelo respaldo (`fallback_used=true`) o turnos no-ok
   en `pipeline_metrics` (`node='chat_turn_phases'`, últimos 24 h).
2. `chat_negative_feedback_high`: vigila proporción de votos `'down'` sobre votos `'up'+'down'` en
   `public.agent_messages` (últimos 7 días).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from knobs import _env_bool, _env_float, _env_int

logger = logging.getLogger(__name__)


def _upsert_alert(
    execute_sql_write: Any,
    *,
    alert_key: str,
    alert_type: str,
    severity: str,
    title: str,
    message: str,
    metadata: dict[str, Any],
) -> None:
    execute_sql_write(
        """
        INSERT INTO system_alerts (alert_key, alert_type, severity, title, message, metadata)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (alert_key) DO UPDATE
        SET triggered_at = NOW(), severity = EXCLUDED.severity,
            title = EXCLUDED.title, message = EXCLUDED.message,
            metadata = EXCLUDED.metadata, resolved_at = NULL
        """,
        (alert_key, alert_type, severity, title, message, json.dumps(metadata, ensure_ascii=False)),
    )


def _resolve_alert(execute_sql_write: Any, alert_key: str) -> None:
    execute_sql_write(
        "UPDATE system_alerts SET resolved_at = NOW() WHERE alert_key = %s AND resolved_at IS NULL",
        (alert_key,),
    )


def _row_val(row: Any, key: str, idx: int, default: int = 0) -> int:
    if isinstance(row, dict):
        return int(row.get(key) or default)
    if isinstance(row, (list, tuple)) and len(row) > idx:
        return int(row[idx] or default)
    return default


def evaluar_salud_coach_y_alertar(
    *,
    execute_sql_query: Any = None,
    execute_sql_write: Any = None,
) -> dict[str, Any]:
    """Evalúa tasa de fallback/errores (24h) y feedback negativo (7d) en SQL sin costo de IA."""
    if not _env_bool("MEALFIT_CHAT_HEALTH_ALERT_ENABLED", True):
        return {"enabled": False}

    if execute_sql_query is None or execute_sql_write is None:
        from db_core import execute_sql_query as _ssot_q, execute_sql_write as _ssot_w

        execute_sql_query = execute_sql_query or _ssot_q
        execute_sql_write = execute_sql_write or _ssot_w

    fb_thresh = _env_float("MEALFIT_CHAT_FALLBACK_ALERT_RATE", 0.15, validator=lambda v: 0.02 <= v <= 1.0)
    err_thresh = _env_float("MEALFIT_CHAT_ERROR_ALERT_RATE", 0.10, validator=lambda v: 0.01 <= v <= 1.0)
    min_turns = max(3, min(_env_int("MEALFIT_CHAT_HEALTH_ALERT_MIN_TURNS", 10), 1000))

    neg_thresh = _env_float("MEALFIT_CHAT_NEG_FEEDBACK_ALERT_RATE", 0.30, validator=lambda v: 0.05 <= v <= 1.0)
    min_votes = max(2, min(_env_int("MEALFIT_CHAT_NEG_FEEDBACK_MIN_VOTES", 5), 500))

    summary: dict[str, Any] = {"enabled": True}

    try:
        rel_row = execute_sql_query(
            """
            SELECT
                COUNT(*)::int AS turns,
                COUNT(*) FILTER (WHERE COALESCE(metadata->>'outcome', 'ok') <> 'ok')::int AS error_turns,
                COUNT(*) FILTER (
                    WHERE EXISTS (
                        SELECT 1 FROM jsonb_array_elements(
                            CASE WHEN jsonb_typeof(metadata->'llm_calls') = 'array'
                                 THEN metadata->'llm_calls' ELSE '[]'::jsonb END
                        ) AS c
                        WHERE (c->>'fallback_used')::boolean IS TRUE
                    )
                )::int AS fallback_turns
            FROM public.pipeline_metrics
            WHERE node = 'chat_turn_phases'
              AND created_at >= NOW() - INTERVAL '24 hours'
            """,
            (),
            fetch_one=True,
        ) or {}
        turns = _row_val(rel_row, "turns", 0)
        err_turns = _row_val(rel_row, "error_turns", 1)
        fb_turns = _row_val(rel_row, "fallback_turns", 2)
        fb_rate = round(fb_turns / turns, 4) if turns else 0.0
        err_rate = round(err_turns / turns, 4) if turns else 0.0
        summary["reliability"] = {
            "turns": turns,
            "fallback_turns": fb_turns,
            "error_turns": err_turns,
            "fallback_rate": fb_rate,
            "error_rate": err_rate,
        }
        if turns >= min_turns and (fb_rate >= fb_thresh or err_rate >= err_thresh):
            sev = "critical" if (fb_rate >= max(fb_thresh * 2, 0.35) or err_rate >= max(err_thresh * 2, 0.25)) else "warning"
            _upsert_alert(
                execute_sql_write,
                alert_key="chat_reliability_degraded",
                alert_type="reliability_degradation",
                severity=sev,
                title=f"Coach con tasa alta de respaldo/error ({fb_turns} fallback, {err_turns} error / {turns} turnos)",
                message=(
                    f"En las últimas 24h: respaldo={fb_rate*100:.1f}% (umbral {fb_thresh*100:.0f}%), "
                    f"errores={err_rate*100:.1f}% (umbral {err_thresh*100:.0f}%) sobre {turns} turnos."
                ),
                metadata=summary["reliability"],
            )
            summary["reliability"]["alerted"] = True
        else:
            _resolve_alert(execute_sql_write, "chat_reliability_degraded")
            summary["reliability"]["alerted"] = False
    except Exception as exc:
        logger.debug("[G-18] reliability check omitido (best-effort): %r", exc)

    try:
        fb_row = execute_sql_query(
            """
            SELECT
                COUNT(*) FILTER (WHERE feedback IN ('up', 'down'))::int AS votes,
                COUNT(*) FILTER (WHERE feedback = 'down')::int AS down_votes
            FROM public.agent_messages
            WHERE feedback IN ('up', 'down')
              AND created_at >= NOW() - INTERVAL '7 days'
            """,
            (),
            fetch_one=True,
        ) or {}
        votes = _row_val(fb_row, "votes", 0)
        downs = _row_val(fb_row, "down_votes", 1)
        down_rate = round(downs / votes, 4) if votes else 0.0
        summary["feedback"] = {"votes": votes, "down_votes": downs, "down_rate": down_rate}
        if votes >= min_votes and down_rate >= neg_thresh:
            sev = "critical" if down_rate >= max(neg_thresh * 1.5, 0.50) else "warning"
            _upsert_alert(
                execute_sql_write,
                alert_key="chat_negative_feedback_high",
                alert_type="quality_degradation",
                severity=sev,
                title=f"Feedback negativo alto en el coach ({downs}/{votes} últimos 7d)",
                message=f"{downs} votos negativos de {votes} totales ({down_rate*100:.1f}% >= {neg_thresh*100:.0f}%).",
                metadata=summary["feedback"],
            )
            summary["feedback"]["alerted"] = True
        else:
            _resolve_alert(execute_sql_write, "chat_negative_feedback_high")
            summary["feedback"]["alerted"] = False
    except Exception as exc:
        logger.debug("[G-18] feedback check omitido (best-effort): %r", exc)

    return summary
