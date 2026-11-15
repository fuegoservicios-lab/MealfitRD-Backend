# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1014 · A-04] Un «fail-open» nunca es silencioso.

El 14-nov se encontró que el tope diario de gasto por usuario (G-19) y la alerta de pico de coste (G-18) consultaban
columnas que `llm_usage_events` no tiene. El error se atrapaba con `except Exception: logger.debug(...)`, el turno seguía
(fail-open, correcto) y NADIE se enteró durante todo el día: dos protecciones que nunca funcionaron.

`fail_open(sitio, exc)` conserva el comportamiento (el turno sigue) pero:
  · cuenta cada fallo por sitio (visible en `/admin/chat-insights` y, como total, en `/health/version`);
  · escribe un WARNING la primera vez por sitio y hora (no inunda el log);
  · en sitios `critico=True` abre además una fila en `system_alerts` (`alert_key = fail_open:<sitio>`), como mucho una por hora.

Nunca lanza: una protección que falla no puede romper lo que protege.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SITIOS: dict[str, dict[str, Any]] = {}
_MAX_SITIOS = 64
_VENTANA_S = 3600.0


def _ahora() -> float:
    return time.monotonic()


def _abrir_alerta(sitio: str, n: int, error: str) -> None:
    try:
        import json

        from db_core import execute_sql_write

        execute_sql_write(
            """
            INSERT INTO system_alerts (alert_key, alert_type, severity, title, message, metadata)
            VALUES (%s, 'fail_open', 'warning', %s, %s, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(), message = EXCLUDED.message, metadata = EXCLUDED.metadata, resolved_at = NULL
            """,
            (
                f"fail_open:{sitio}",
                f"Protección del coach fallando en silencio: {sitio}",
                (f"`{sitio}` falló {n} vez(ces) y el turno siguió sin ella (fail-open). Último error: {error[:300]}. "
                 "Mientras tanto esa protección NO está actuando."),
                json.dumps({"sitio": sitio, "fallos": n, "ultimo_error": error[:500]}),
            ),
        )
    except Exception as exc:  # noqa: BLE001 — sin recursión: si ni la alerta se puede abrir, queda el WARNING
        logger.warning("[A-04] no se pudo abrir la alerta fail_open:%s: %s", sitio, type(exc).__name__)


def fail_open(sitio: str, exc: Any, *, critico: bool = False) -> None:
    """Registra que `sitio` falló y el flujo siguió. `exc` es la excepción (o un texto)."""
    try:
        error = f"{type(exc).__name__}: {str(exc)[:240]}" if isinstance(exc, BaseException) else str(exc)[:260]
        ahora = _ahora()
        avisar_log = abrir_alerta = False
        with _LOCK:
            if sitio not in _SITIOS and len(_SITIOS) >= _MAX_SITIOS:
                return
            s = _SITIOS.setdefault(sitio, {"n": 0, "critico": bool(critico), "log_ts": None, "alerta_ts": None,
                                           "ultimo_error": "", "ultimo_ts": ahora})
            s["n"] += 1
            s["critico"] = s["critico"] or bool(critico)
            s["ultimo_error"] = error
            s["ultimo_ts"] = ahora
            if s["log_ts"] is None or ahora - s["log_ts"] >= _VENTANA_S:
                s["log_ts"] = ahora
                avisar_log = True
            if critico and (s["alerta_ts"] is None or ahora - s["alerta_ts"] >= _VENTANA_S):
                s["alerta_ts"] = ahora
                abrir_alerta = True
            n = s["n"]
        if avisar_log:
            logger.warning("⚠️ [A-04 FAIL-OPEN] %s falló (%d en este proceso; el turno siguió sin esa protección): %s",
                           sitio, n, error)
        if abrir_alerta:
            _abrir_alerta(sitio, n, error)
    except Exception:  # noqa: BLE001
        pass


def snapshot() -> dict[str, dict[str, Any]]:
    """Copia de los contadores: `{sitio: {n, critico, ultimo_error, hace_s}}`."""
    ahora = _ahora()
    with _LOCK:
        return {k: {"n": v["n"], "critico": v["critico"], "ultimo_error": v["ultimo_error"],
                    "hace_s": int(ahora - v["ultimo_ts"])} for k, v in _SITIOS.items()}


def total(solo_criticos: bool = False) -> int:
    with _LOCK:
        return sum(v["n"] for v in _SITIOS.values() if v["critico"] or not solo_criticos)


def reset_para_tests() -> None:
    with _LOCK:
        _SITIOS.clear()
