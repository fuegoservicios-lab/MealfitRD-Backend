# backend/telemetria_escaner.py
"""[P1-PLAN-LOTE-575 · 2026-09-27] Dos señales del escáner para el panel de administración (spec §2), sin texto libre.

  · `vision_scan_resultado`: cada análisis de foto — cuánto tardó y si sirvió (ok / error / no_comida / sin_totales).
  · `scan_outcome`: al REGISTRAR un plato escaneado, cuánto corrigió el usuario lo que dijo la IA (solo conteos).

Best-effort: un fallo aquí jamás rompe el escaneo ni el registro, y un `scan_meta` con basura se acota en vez de
rechazar la comida (lo manda el cliente: no se confía en su forma).
"""
from __future__ import annotations

import json
import logging
import math
from typing import Optional

from db import execute_sql_write

logger = logging.getLogger(__name__)

_TOPES = {"componentes": 40, "cambiados": 40, "cantidades_editadas": 40, "desmarcados": 40, "dudas": 5,
          "dudas_cambiadas": 5, "kcal_ia": 10000, "kcal_final": 10000}
_BANDERAS = ("redescrito", "nombre_editado", "macros_tecleadas")


def _entero(v, tope: int) -> int:
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int):                       # sin pasar por float: float(10**400) lanza OverflowError
        return max(0, min(tope, v))
    try:
        n = float(v or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(n):
        return 0
    return max(0, min(tope, int(n)))


def resultado_del_analisis(res) -> str:
    if not isinstance(res, dict) or res.get("analysis_failed"):
        return "error"
    if res.get("is_food") is False:
        return "no_comida"
    try:
        return "ok" if float(res.get("calories") or 0) > 0 else "sin_totales"
    except (TypeError, ValueError):
        return "sin_totales"


def resumen_de_correcciones(crudo) -> Optional[dict]:
    if not isinstance(crudo, dict):
        return None
    r = {k: _entero(crudo.get(k), tope) for k, tope in _TOPES.items()}
    try:
        porcion = float(crudo.get("porcion", 1) or 1)
    except (TypeError, ValueError, OverflowError):
        porcion = 1.0
    r["porcion"] = max(0.0, min(4.0, porcion)) if math.isfinite(porcion) else 1.0
    for k in _BANDERAS:
        r[k] = crudo.get(k) is True
    r["corregido"] = bool(r["cambiados"] or r["cantidades_editadas"] or r["desmarcados"] or r["dudas_cambiadas"]
                          or r["redescrito"] or r["nombre_editado"] or r["macros_tecleadas"]
                          or abs(r["porcion"] - 1.0) > 1e-6)
    r["desvio_kcal"] = round(abs(r["kcal_final"] - r["kcal_ia"]) / r["kcal_ia"], 3) if r["kcal_ia"] else None
    return r


def _insertar(user_id: Optional[str], node: str, duracion_ms: float, metadata: dict) -> None:
    try:
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, tokens_estimated, "
            "confidence, metadata) VALUES (%s, %s, %s, %s, 0, 0, 0, %s::jsonb)",
            (user_id, None, node, int(duracion_ms or 0), json.dumps(metadata, ensure_ascii=False)),
        )
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-575] señal {node} no-op: {e!r}")


def registrar_vision_scan(
    user_id: Optional[str],
    *,
    duracion_ms: float,
    resultado,
    purpose: str,
    fases_ms: Optional[dict] = None,
    modelo: Optional[str] = None,
    resize_info: Optional[dict] = None,
) -> None:
    kind = resultado.get("photo_kind") if isinstance(resultado, dict) else ""
    meta: dict = {
        "resultado": resultado_del_analisis(resultado),
        "photo_kind": str(kind or "")[:16],
        "purpose": str(purpose or "")[:16],
    }
    if modelo:
        meta["modelo"] = str(modelo)[:64]
    if isinstance(fases_ms, dict) and fases_ms:
        limpio = {
            str(k)[:24]: round(float(v), 2)
            for k, v in list(fases_ms.items())[:12]
            if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) >= 0
        }
        if limpio:
            meta["fases_ms"] = limpio
    if isinstance(resize_info, dict) and resize_info:
        meta["resize"] = {
            "original_bytes": _entero(resize_info.get("original_bytes"), 50_000_000),
            "sent_bytes": _entero(resize_info.get("sent_bytes"), 50_000_000),
            "resized": bool(resize_info.get("resized")),
            "sent_wh": str(resize_info.get("sent_wh") or "")[:24],
        }
    _insertar(user_id, "vision_scan_resultado", duracion_ms, meta)


def registrar_scan_outcome(user_id: str, resumen: dict) -> None:
    _insertar(user_id, "scan_outcome", 0, resumen)
