# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1023 · A-32] El margen del límite de OpenAI, a la vista y con aviso al 70 %.

No se conocía el límite de tokens por minuto de la cuenta y cada turno del coach gasta ~30-60 mil. Al llegar al tope,
el proveedor contesta 429 y el coach cae al respaldo (Hydrogen, ~50 veces más caro por llamada). OpenAI manda el margen
en cada respuesta (`x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, y lo mismo para `-requests`).

`registrar(headers, modelo)` guarda la última foto del margen y, si el uso de la ventana pasa del 70 %
(`MEALFIT_OPENAI_RATELIMIT_ALERT_PCT`), escribe un WARNING (uno cada 10 min) y abre una fila en `system_alerts`
(`alert_key = openai_ratelimit`, una por hora). `snapshot()` alimenta `/health/version`. Nunca lanza.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_ultimo: dict[str, Any] = {}
_avisos = {"log": None, "alerta": None}
_LOG_S = 600.0
_ALERTA_S = 3600.0


def umbral_pct() -> int:
    from knobs import _env_int
    return _env_int("MEALFIT_OPENAI_RATELIMIT_ALERT_PCT", 70, validator=lambda v: 1 <= v <= 100)


def vigilancia_activa() -> bool:
    from knobs import _env_bool
    return _env_bool("MEALFIT_OPENAI_RATELIMIT_WATCH", True)


def _entero(v) -> Optional[int]:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return None


def _uso(limite: Optional[int], restante: Optional[int]) -> Optional[float]:
    if not limite or limite <= 0 or restante is None:
        return None
    return max(0.0, min(1.0, 1.0 - restante / limite))


def _abrir_alerta(foto: dict) -> None:
    try:
        import json

        from db_core import execute_sql_write

        execute_sql_write(
            """
            INSERT INTO system_alerts (alert_key, alert_type, severity, title, message, metadata)
            VALUES ('openai_ratelimit', 'openai_ratelimit', 'warning', %s, %s, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(), message = EXCLUDED.message, metadata = EXCLUDED.metadata, resolved_at = NULL
            """,
            (
                "El coach se acerca al límite de OpenAI",
                (f"Uso de la ventana: tokens {foto.get('uso_tokens_pct')} %, peticiones {foto.get('uso_peticiones_pct')} % "
                 f"(modelo {foto.get('modelo')}). Al llegar al 100 % OpenAI responde 429 y el coach cae al respaldo, mucho más caro."),
                json.dumps(foto),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[A-32] no se pudo abrir la alerta openai_ratelimit: %s", type(exc).__name__)


def registrar(headers: Any, modelo: str = "") -> Optional[dict]:
    """Lee las cabeceras de una respuesta de OpenAI. Devuelve la foto (o None si no traían el margen)."""
    try:
        h = {str(k).lower(): v for k, v in dict(headers or {}).items()}
        lim_t, res_t = _entero(h.get("x-ratelimit-limit-tokens")), _entero(h.get("x-ratelimit-remaining-tokens"))
        lim_r, res_r = _entero(h.get("x-ratelimit-limit-requests")), _entero(h.get("x-ratelimit-remaining-requests"))
        uso_t, uso_r = _uso(lim_t, res_t), _uso(lim_r, res_r)
        if uso_t is None and uso_r is None:
            return None
        foto = {
            "modelo": str(modelo or "")[:60],
            "limite_tokens": lim_t, "restante_tokens": res_t,
            "limite_peticiones": lim_r, "restante_peticiones": res_r,
            "uso_tokens_pct": None if uso_t is None else round(uso_t * 100, 1),
            "uso_peticiones_pct": None if uso_r is None else round(uso_r * 100, 1),
        }
        peor = max(x for x in (uso_t, uso_r) if x is not None)
        ahora = time.monotonic()
        avisar_log = abrir = False
        with _LOCK:
            _ultimo.clear()
            _ultimo.update(foto, ts=ahora)
            if peor * 100 >= umbral_pct():
                if _avisos["log"] is None or ahora - _avisos["log"] >= _LOG_S:
                    _avisos["log"] = ahora
                    avisar_log = True
                if _avisos["alerta"] is None or ahora - _avisos["alerta"] >= _ALERTA_S:
                    _avisos["alerta"] = ahora
                    abrir = True
        if avisar_log:
            logger.warning("⚠️ [A-32 RATELIMIT] %s: tokens %s %%, peticiones %s %% de la ventana de OpenAI",
                           foto["modelo"] or "?", foto["uso_tokens_pct"], foto["uso_peticiones_pct"])
        if abrir:
            _abrir_alerta(foto)
        return foto
    except Exception:  # noqa: BLE001
        return None


def snapshot() -> dict:
    with _LOCK:
        if not _ultimo:
            return {}
        out = {k: v for k, v in _ultimo.items() if k != "ts"}
        out["hace_s"] = int(time.monotonic() - _ultimo["ts"])
        return out


def reset_para_tests() -> None:
    with _LOCK:
        _ultimo.clear()
        _avisos.update({"log": None, "alerta": None})
