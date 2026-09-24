# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-208 · 2026-09-24] «Modo automático» (`logging_preference = 'auto_proxy'`) cumple lo que promete.

Configuración → «Modo automático»: «Si confías en el plan y prefieres no loguear cada comida, actívalo. No pausaremos tu
plan aunque dejes de registrar comidas» (y el aviso al activarlo: «ya no pausaremos tu plan por falta de logs»). El gate
de aprendizaje (`_check_chunk_learning_ready`) sólo lo miraba DENTRO de la rama del proxy de inventario —con ≥ N descuentos
de la Nevera por consumo—, donde se salta las pausas de «proxy agotado» y «registro crónico ausente». Sin registros y
sin esos descuentos, que es el caso de casi todos, el bloque siguiente se pausaba igual (`learning_zero_logs`, 6 h de
TTL) con el push «Loguea tus comidas para continuar», y el cron de nudges le escribía cada día «Si no logueas pronto, el
siguiente bloque de tu plan se pausará». Medido sin IA contra producción: el plan de 30 días del dueño (modo automático
activado, 0 registros de comida, 18 cambios manuales en la Nevera y 0 descuentos por consumo) iba a pausarse al cambiar
de bloque el 26-sep a las 00:30 en los dos escenarios (abriendo la app el 25 o no).

Aquí, con el modo automático:
  · el gate deja pasar el bloque con señal DÉBIL (el worker fuerza variedad, como hace con el proxy de inventario);
  · un bloque que ya estaba pausado por falta de registros se reanuda en el siguiente tick del recovery;
  · el nudge diario de «loguea o se pausará» no le llega.
Los usuarios en modo manual no cambian. Knob `MEALFIT_AUTO_PROXY_HONORED` (True).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

AUTO = "auto_proxy"


def honrado() -> bool:
    """tooltip-anchor: MEALFIT_AUTO_PROXY_HONORED"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_AUTO_PROXY_HONORED", True)
    except Exception:
        return True


def activo(user_id, consultar: Optional[Callable] = None, preferencia=None) -> bool:
    """¿El usuario eligió el modo automático? `preferencia` ya leída (el gate la trae del mismo SELECT del perfil) o, si
    es None, se consulta. Fail-closed: ante error, False (conducta de siempre)."""
    if not honrado():
        return False
    if preferencia is not None:
        return str(preferencia) == AUTO
    try:
        if consultar is None:
            from db_core import execute_sql_query as consultar
        fila = consultar("SELECT logging_preference FROM user_profiles WHERE id = %s", (str(user_id),), fetch_one=True)
        return (fila or {}).get("logging_preference") == AUTO
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-208] logging_preference no disponible para {user_id}: {e}")
        return False


def filtro_nudge() -> str:
    """Condición SQL (sin parámetros) para que el nudge de «loguea o se pausará» no le escriba a quien eligió no loguear."""
    return "AND COALESCE(p.logging_preference, 'manual') <> 'auto_proxy'" if honrado() else ""
