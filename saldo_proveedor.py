# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-654 · 2026-09-27] Un proveedor de IA sin saldo deja una alerta, no solo un 429 en el log.

El 17-sep Z.ai respondió `429 code 1113 "Insufficient balance"` y el backend lo trató como un rate-limit que se pasa
solo: el tráfico acabó en DeepSeek y ninguna alerta lo contó. Este módulo reconoce el agotamiento de saldo de cada
proveedor (por la frase del error, que es lo único estable entre SDKs) y emite `system_alert`
`llm_provider_balance_exhausted:<proveedor>`. Lo llama la subclase `ChatOpenAI` de `llm_provider` en su camino de
error —el punto por el que pasan Z.ai, DeepSeek y OpenAI— y nunca cambia el error: el breaker y la red cruzada siguen
decidiendo como antes. Resolución manual: el operador recarga y cierra la alerta.

tooltip-anchor: P1-PLAN-LOTE-654
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# La frase de cada proveedor cuando se queda sin saldo. Un 429 de rate-limit NO la trae.
_FRASES_SIN_SALDO = re.compile(
    r"insufficient[ _]balance|code['\"]?\s*[:=]\s*['\"]?1113\b|余额不足|insufficient_quota|"
    r"exceeded your current quota|"
    r"spending cap|ai\.studio/spend|no resource package",
    re.I,
)
_PROVEEDOR_POR_MODELO = (("glm", "zai"), ("deepseek", "deepseek"), ("gpt", "openai"), ("o3", "openai"),
                         ("o4", "openai"), ("gemini", "gemini"))
_PROVEEDOR_POR_FRASE = ((re.compile(r"insufficient_quota|exceeded your current quota", re.I), "openai"),
                        (re.compile(r"spending cap|ai\.studio/spend", re.I), "gemini"),
                        (re.compile(r"1113|余额不足|no resource package", re.I), "zai"))
_VENTANA_S = 600
_ultimo_aviso: dict = {}


def proveedor_sin_saldo(exc, modelo: str = "") -> Optional[str]:
    """El proveedor que se quedó sin saldo según `exc` (y el modelo que se llamaba), o None si el error es otra cosa."""
    try:
        texto = f"{type(exc).__name__}: {exc}"
    except Exception:  # noqa: BLE001
        return None
    if not _FRASES_SIN_SALDO.search(texto):
        return None
    m = str(modelo or "").lower()
    for prefijo, prov in _PROVEEDOR_POR_MODELO:
        if m.startswith(prefijo):
            return prov
    for rx, prov in _PROVEEDOR_POR_FRASE:
        if rx.search(texto):
            return prov
    return "desconocido"


def _escribir_alerta(proveedor: str, detalle: str) -> None:
    from db_core import execute_sql_write
    execute_sql_write(
        """
        INSERT INTO system_alerts (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
        VALUES (%s, 'llm_provider_balance', 'critical', %s, %s, %s::jsonb, '[]'::jsonb)
        ON CONFLICT (alert_key) DO UPDATE
        SET triggered_at = NOW(), message = EXCLUDED.message, metadata = EXCLUDED.metadata, resolved_at = NULL
        """,
        (
            f"llm_provider_balance_exhausted:{proveedor}",
            f"Proveedor de IA sin saldo: {proveedor}",
            (f"El proveedor {proveedor} rechaza las llamadas por falta de saldo. El tráfico cae al respaldo mientras "
             f"el breaker lo permita; recarga la cuenta y cierra esta alerta."),
            json.dumps({"provider": proveedor, "detalle": str(detalle)[:300]}, ensure_ascii=False),
        ),
    )


def avisar_si_saldo_agotado(exc, modelo: str = "") -> Optional[str]:
    """Si `exc` es falta de saldo, emite la alerta (una vez por ventana y proveedor) y devuelve el proveedor. Nunca
    lanza: quien llama va a re-lanzar su propio error."""
    prov = proveedor_sin_saldo(exc, modelo)
    if not prov:
        return None
    ahora = time.time()
    if ahora - _ultimo_aviso.get(prov, 0.0) >= _VENTANA_S:
        _ultimo_aviso[prov] = ahora
        logger.error(f"🛑 [P1-PLAN-LOTE-654] proveedor de IA sin saldo: {prov} (modelo {modelo}): {str(exc)[:200]}")
        try:
            _escribir_alerta(prov, str(exc))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[P1-PLAN-LOTE-654] no se pudo escribir la alerta de saldo de {prov}: {e!r}")
    return prov
