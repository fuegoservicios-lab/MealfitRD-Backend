# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-135 · 2026-09-20] La invitación «¿Quieres que la IA te arme el plan?»: una vez por SEMANA y por USUARIO.

El dueño, con la captura del contador: «quiero que aparezca una sola vez por usuario y que cuando le den a "Ahora no" no
aparezca más, ya que eso puedes encenderlo en Configuración… lo que sí puedes hacer es que aparezca 1 vez a la semana, para
así motivar al usuario a encenderlo».

Por qué la veía «tan seguido»: el descarte vivía SOLO en el `localStorage` del dispositivo (`mealfit_turnon_card_dismissed`).
Cada binario nuevo de TestFlight, cada navegador y la PWA traen un almacén distinto, así que «Ahora no» valía para ESE
almacén y la tarjeta volvía en el siguiente. Ahora la cuenta la lleva el servidor, por usuario.

La regla (pura, `estado_de_la_invitacion`):
  · nunca vista → se muestra, y esa primera vista abre la semana;
  · vista → sigue a la vista `HORAS_A_LA_VISTA` (24) —quien la vio por la mañana y vuelve por la tarde no la pierde a
    mitad de decidirse— y luego se esconde sola hasta cumplir `DIAS_ENTRE_INVITACIONES` (7) desde esa vista;
  · «Ahora no» → se esconde en el acto, hasta 7 días después del descarte.
Estado en `app_kv_store` (`plan_invite:<user_id>`), con barrido por TTL y borrado en la purga de la cuenta: sin DDL.
tooltip-anchor: estado_de_la_invitacion (test_p1_plan_lote_135.py)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from knobs import _env_int

logger = logging.getLogger(__name__)

PREFIJO = "plan_invite:"
HORAS_A_LA_VISTA = 24


def _dias_entre_invitaciones() -> int:
    return _env_int("MEALFIT_PLAN_INVITE_EVERY_DAYS", 7, validator=lambda v: 1 <= v <= 90)


def _parse(valor) -> Optional[datetime]:
    if not valor or not isinstance(valor, str):
        return None
    try:
        dt = datetime.fromisoformat(valor)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def estado_de_la_invitacion(doc: Optional[dict], ahora: datetime) -> dict:
    """`{visible, next_at}`. Pura: todo entra por parámetro."""
    doc = doc or {}
    vista, descartada = _parse(doc.get("shown_at")), _parse(doc.get("dismissed_at"))
    semana = timedelta(days=_dias_entre_invitaciones())
    if vista is None and descartada is None:
        return {"visible": True, "next_at": None}
    if descartada is not None and (vista is None or descartada >= vista):
        vuelve = descartada + semana
        return {"visible": ahora >= vuelve, "next_at": vuelve.isoformat()}
    vuelve = vista + semana
    visible = ahora < vista + timedelta(hours=HORAS_A_LA_VISTA) or ahora >= vuelve
    return {"visible": visible, "next_at": vuelve.isoformat()}


def _leer(user_id: str) -> dict:
    from db_core import execute_sql_query
    fila = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (PREFIJO + str(user_id),), fetch_one=True)
    valor = (fila or {}).get("value") if isinstance(fila, dict) else None
    if isinstance(valor, str):
        valor = json.loads(valor)
    return dict(valor) if isinstance(valor, dict) else {}


def _guardar(user_id: str, doc: dict) -> None:
    from db_core import execute_sql_write
    execute_sql_write(
        "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
        (PREFIJO + str(user_id), json.dumps(doc)),
    )


def leer_invitacion(user_id: str, ahora: Optional[datetime] = None) -> dict:
    return estado_de_la_invitacion(_leer(user_id), ahora or datetime.now(timezone.utc))


def anotar(user_id: str, accion: str, ahora: Optional[datetime] = None) -> dict:
    """`accion`: 'seen' (la tarjeta se pintó) o 'dismiss' («Ahora no»). Idempotente dentro de la misma ventana."""
    ahora = ahora or datetime.now(timezone.utc)
    doc = _leer(user_id)
    if accion == "dismiss":
        doc["dismissed_at"] = ahora.isoformat()
    elif accion == "seen":
        vista = _parse(doc.get("shown_at"))
        previo = estado_de_la_invitacion(doc, ahora)
        # solo una vista que ABRE semana mueve el reloj: re-pintarla dentro de sus 24 h no la alarga
        abre_semana = vista is None or ahora >= vista + timedelta(hours=HORAS_A_LA_VISTA)
        if previo["visible"] and abre_semana:
            doc["shown_at"] = ahora.isoformat()
    else:
        raise ValueError("accion")
    _guardar(user_id, doc)
    return estado_de_la_invitacion(doc, ahora)
