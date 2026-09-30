# backend/diagnostico_voz.py
"""[P1-PLAN-LOTE-909 · 2026-09-30] Diagnóstico anónimo del dictado y el modo voz.

El dueño: «en android todavía tienen problemas». El servidor no veía nada: el dictado ocurre dentro del teléfono y,
si no oye, no manda nada. Lo único visible era un Xiaomi que tocó «Detener» dos veces. Ahora la app avisa cuando el
reconocedor falla (con el código ORIGINAL de Android, que el adaptador traducía y perdía) y, una vez por arranque, en
qué estado está la voz en ese binario (¿trae los plugins? ¿qué motor elige?): eso cubre el caso en que el micrófono ni
aparece, que no produce ningún fallo.

Qué se guarda: fila en `pipeline_metrics` (node `voz_diagnostico`) SIN la cuenta —ni `user_id` ni un hash: un hash del
id es un seudónimo (revisión legal de e7)—: el modelo del teléfono y la versión de Android salen del user agent. Nunca
audio ni texto. Se purga con el resto de `pipeline_metrics` (`MEALFIT_PIPELINE_METRICS_RETENTION_DAYS`, 30 días): la
Política de Privacidad §2 lo declara con ese plazo.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

NODO = "voz_diagnostico"
DONDE = frozenset({"dictado", "modo_voz", "estado", "sintesis"})
_CODIGO_RE = re.compile(r"^[a-z0-9_\-:.]{1,48}$", re.I)
_ANDROID_RE = re.compile(r"Android (\d+(?:\.\d+)?); ([^;)]+?)(?: Build/|\)|;)")


def _corto(v, n: int) -> Optional[str]:
    if v is None or isinstance(v, (dict, list)):
        return None
    s = str(v).strip()
    return s[:n] if s else None


def _codigo(v) -> Optional[str]:
    s = _corto(v, 48)
    return s if s and _CODIGO_RE.match(s) else None


def normalizar(datos: dict, user_agent: str = "") -> Optional[dict]:
    """Solo los campos conocidos, cortos y sin texto libre. `None` si no es un diagnóstico válido."""
    if not isinstance(datos, dict):
        return None
    donde = _codigo(datos.get("donde"))
    if donde not in DONDE:
        return None
    meta = {
        "donde": donde,
        "codigo": _codigo(datos.get("codigo")),
        "crudo": _corto(datos.get("crudo"), 80),       # el código del motor sin traducir (UNKNOWN_12, AUDIO…)
        "motor": _codigo(datos.get("motor")),
        "idioma": _codigo(datos.get("idioma")),
        "plataforma": _codigo(datos.get("plataforma")),
        "plugin_reconocimiento": datos.get("plugin_reconocimiento") if isinstance(datos.get("plugin_reconocimiento"), bool) else None,
        "plugin_sintesis": datos.get("plugin_sintesis") if isinstance(datos.get("plugin_sintesis"), bool) else None,
        "dictado_disponible": datos.get("dictado_disponible") if isinstance(datos.get("dictado_disponible"), bool) else None,
    }
    m = _ANDROID_RE.search(user_agent or "")
    if m:
        meta["android"] = m.group(1)
        meta["modelo"] = m.group(2).strip()[:40]
    elif "iPhone" in (user_agent or ""):
        meta["modelo"] = "iPhone"
    return {k: v for k, v in meta.items() if v is not None}


def registrar(meta: dict) -> None:
    """Una fila por aviso. No lanza: el diagnóstico nunca rompe a quien lo manda."""
    try:
        from db_core import execute_sql_write
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, tokens_estimated, confidence, metadata) "
            "VALUES (NULL, NULL, %s, 0, 0, 0, 0, %s::jsonb)",
            (NODO, json.dumps(meta, ensure_ascii=False)),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-909] no se pudo guardar el diagnóstico de voz: {type(e).__name__}: {e}")
    logger.info(f"🎙️ [P1-PLAN-LOTE-909] diagnóstico de voz: {meta}")
