# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1022 · A-21] Aviso cuando una llamada del coach pasa del presupuesto de tokens de entrada.

Medido en producción (15-nov): una llamada normal del coach entra con ~30-34 mil tokens (manual ~17 mil, herramientas
~7 mil, contexto e historial el resto) y con caché v2 casi todo es caché. Pasar de 40 mil significa que algo creció sin
que nadie lo decidiera (un bloque de contexto desbocado, un historial sin recortar, un resultado de herramienta enorme):
cuesta más, tarda más y rompe el prefijo cacheado. Antes nadie se enteraba.

`vigilar(input_tokens, ...)` no cambia nada del turno: si la llamada pasa del umbral
(`MEALFIT_CHAT_TURN_TOKEN_ALERT`, 40 000 por defecto; 0 lo apaga) escribe un WARNING y abre (o refresca) una fila en
`system_alerts` (`alert_key = coach_tokens_turno`), como mucho una vez por hora por proceso. Nunca lanza.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_VENTANA_S = 3600.0
_estado = {"ultimo_aviso": None, "excesos": 0, "maximo": 0}


def umbral() -> int:
    from knobs import _env_int
    return _env_int("MEALFIT_CHAT_TURN_TOKEN_ALERT", 40000, validator=lambda v: 0 <= v <= 2_000_000)


def _abrir_alerta(tokens: int, umbral_: int, excesos: int, maximo: int, modelo: str) -> None:
    try:
        import json

        from db_core import execute_sql_write

        execute_sql_write(
            """
            INSERT INTO system_alerts (alert_key, alert_type, severity, title, message, metadata)
            VALUES ('coach_tokens_turno', 'coach_tokens', 'warning', %s, %s, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(), message = EXCLUDED.message, metadata = EXCLUDED.metadata, resolved_at = NULL
            """,
            (
                "Una llamada del coach pasó del presupuesto de tokens",
                (f"Una llamada de {modelo} entró con {tokens} tokens (umbral {umbral_}). {excesos} exceso(s) en este proceso, "
                 f"máximo {maximo}. Algo del prompt creció sin que se decidiera: revisa el desglose de capas."),
                json.dumps({"tokens": tokens, "umbral": umbral_, "excesos": excesos, "maximo": maximo, "modelo": modelo}),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[A-21] no se pudo abrir la alerta coach_tokens_turno: %s", type(exc).__name__)


def vigilar(input_tokens: Optional[int], *, modelo: str = "") -> bool:
    """True si esta llamada pasó del umbral (para tests y métricas). Nunca lanza."""
    try:
        tokens = int(input_tokens or 0)
        u = umbral()
        if u <= 0 or tokens <= u:
            return False
        ahora = time.monotonic()
        with _LOCK:
            _estado["excesos"] += 1
            _estado["maximo"] = max(_estado["maximo"], tokens)
            avisar = _estado["ultimo_aviso"] is None or ahora - _estado["ultimo_aviso"] >= _VENTANA_S
            if avisar:
                _estado["ultimo_aviso"] = ahora
            excesos, maximo = _estado["excesos"], _estado["maximo"]
        if avisar:
            logger.warning("⚠️ [A-21 TOKENS] llamada de %s con %d tokens de entrada (umbral %d; %d exceso(s) en este proceso)",
                           modelo or "?", tokens, u, excesos)
            _abrir_alerta(tokens, u, excesos, maximo, modelo or "?")
        return True
    except Exception:  # noqa: BLE001
        return False


def snapshot() -> dict:
    with _LOCK:
        return {"excesos": _estado["excesos"], "maximo": _estado["maximo"]}


def reset_para_tests() -> None:
    with _LOCK:
        _estado.update({"ultimo_aviso": None, "excesos": 0, "maximo": 0})
