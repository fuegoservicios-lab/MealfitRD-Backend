# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-847 · 2026-09-29] Al borrar la cuenta, borrar también su persona y sus eventos en PostHog.

Por qué: la app identifica al usuario en PostHog con su id (`identifyPostHog`, frontend), y el borrado de cuenta no
contactaba con PostHog: la persona y sus eventos seguían vinculados al id (auditoría App Store 2026-09-29, fila 3.3;
RGPD art. 17 sobre lo que guarda un encargado).

Cómo (API de personas de PostHog, vigente):
  1. `GET {host}/api/projects/{project_id}/persons/?distinct_id={user_id}` → la persona (su `id` es un UUID).
  2. `DELETE {host}/api/projects/{project_id}/persons/{uuid}/?delete_events=true&delete_recordings=true`.
  PostHog procesa el borrado de eventos de forma asíncrona (en su lado); aquí basta con que acepte la petición.
  Solo se borra una persona cuyo `distinct_ids` CONTIENE el id: si el filtro se ignorara y la lista trajera a otros,
  no se toca a nadie más.

Configuración (VPS), por `os.environ` y no por `knobs` (el registro de knobs guarda el valor crudo y lo expone):
  - `POSTHOG_PERSONAL_API_KEY`: clave PERSONAL con permiso de escritura sobre personas (no la `phc_` del frontend);
  - `POSTHOG_PROJECT_ID`: el id numérico del proyecto;
  - `POSTHOG_HOST`: por defecto `https://us.posthog.com` (la región del frontend: `us.i.posthog.com`).
Sin clave o sin proyecto: no se llama a nadie, se anota UNA advertencia y el borrado sigue.

Best-effort y con plazo TOTAL: nunca lanza ni bloquea el borrado de la cuenta. Nunca se escribe en el log la clave;
del usuario, solo el prefijo del id. Interruptor: `MEALFIT_POSTHOG_ACCOUNT_DELETE=false`.
tooltip-anchor: P1-PLAN-LOTE-847-POSTHOG-BORRADO
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from knobs import _env_bool

logger = logging.getLogger(__name__)

POSTHOG_DEFAULT_HOST = "https://us.posthog.com"
# Plazo TOTAL (búsqueda + borrado) dentro de `/api/account/delete`: pasado, el borrado responde sin esperar.
DELETE_DEADLINE_S = 10.0
_HTTP_TIMEOUT_S = 6.0
_HTTP_CONNECT_TIMEOUT_S = 3.0

_PROYECTO_RE = re.compile(r"^\d{1,20}$")
_PERSONA_RE = re.compile(r"^[0-9A-Za-z-]{1,64}$")

_avisado_sin_configurar = False


def borrado_activado() -> bool:
    """Interruptor de emergencia (sin redeploy)."""
    return _env_bool("MEALFIT_POSTHOG_ACCOUNT_DELETE", True)


def _conf() -> Optional[dict]:
    """La configuración, o None si falta algo. La clave NO sale de aquí más que hacia la cabecera."""
    clave = (os.environ.get("POSTHOG_PERSONAL_API_KEY") or "").strip()
    proyecto = (os.environ.get("POSTHOG_PROJECT_ID") or "").strip()
    host = (os.environ.get("POSTHOG_HOST") or "").strip().rstrip("/") or POSTHOG_DEFAULT_HOST
    if not clave or not proyecto:
        return None
    if not _PROYECTO_RE.match(proyecto) or not host.startswith("https://"):
        logger.warning("[P1-PLAN-LOTE-847] POSTHOG_PROJECT_ID o POSTHOG_HOST con formato inválido: no se borra en PostHog.")
        return None
    return {"clave": clave, "proyecto": proyecto, "host": host}


def _peticion(metodo: str, url: str, clave: str, params: Optional[dict] = None):
    """Punto ÚNICO de red (los tests lo sustituyen). Devuelve (status, json|None)."""
    import httpx

    with httpx.Client(timeout=httpx.Timeout(_HTTP_TIMEOUT_S, connect=_HTTP_CONNECT_TIMEOUT_S)) as cliente:
        r = cliente.request(metodo, url, params=params, headers={"Authorization": f"Bearer {clave}"})
    try:
        cuerpo = r.json()
    except Exception:
        cuerpo = None
    return r.status_code, cuerpo


def borrar_de_posthog(user_id: str) -> dict:
    """Borra la persona `distinct_id = user_id` y sus eventos. Devuelve `{borrado, motivo[, personas]}`. Nunca lanza."""
    global _avisado_sin_configurar
    try:
        uid = str(user_id or "").strip()
        if not uid or uid == "guest":
            return {"borrado": False, "motivo": "sin_usuario"}
        if not borrado_activado():
            return {"borrado": False, "motivo": "apagado"}
        conf = _conf()
        if conf is None:
            if not _avisado_sin_configurar:
                _avisado_sin_configurar = True
                logger.warning(
                    "[P1-PLAN-LOTE-847] Falta POSTHOG_PERSONAL_API_KEY o POSTHOG_PROJECT_ID: al borrar una cuenta NO se "
                    "borra su persona en PostHog (el borrado de la cuenta sigue)."
                )
            return {"borrado": False, "motivo": "sin_configurar"}

        base = f"{conf['host']}/api/projects/{conf['proyecto']}/persons/"
        status, cuerpo = _peticion("GET", base, conf["clave"], params={"distinct_id": uid})
        if status != 200 or not isinstance(cuerpo, dict):
            logger.warning(f"[P1-PLAN-LOTE-847] PostHog: buscar la persona {uid[:8]} devolvió HTTP {status}.")
            return {"borrado": False, "motivo": f"buscar_http_{status}"}

        personas = [
            p for p in (cuerpo.get("results") or [])
            if isinstance(p, dict) and uid in (p.get("distinct_ids") or []) and _PERSONA_RE.match(str(p.get("id") or ""))
        ]
        if not personas:
            # Nunca dio permiso de analítica (o PostHog ya no la tiene): nada que borrar.
            return {"borrado": True, "motivo": "sin_persona", "personas": 0}

        for p in personas:
            status, _ = _peticion(
                "DELETE", f"{base}{p['id']}/", conf["clave"],
                params={"delete_events": "true", "delete_recordings": "true"},
            )
            # 404: ya no estaba (otro borrado en curso). Cuenta como hecho.
            if status not in (200, 202, 204, 404):
                logger.warning(f"[P1-PLAN-LOTE-847] PostHog: borrar la persona de {uid[:8]} devolvió HTTP {status}.")
                return {"borrado": False, "motivo": f"borrar_http_{status}"}

        logger.info(f"✅ [P1-PLAN-LOTE-847] PostHog: persona de {uid[:8]} y sus eventos, borrados ({len(personas)}).")
        return {"borrado": True, "motivo": "ok", "personas": len(personas)}
    except Exception as e:
        # Solo el tipo: el mensaje de httpx puede llevar la URL, y el de otras capas, cabeceras.
        logger.warning(f"[P1-PLAN-LOTE-847] borrar_de_posthog lanzó {type(e).__name__} (el borrado sigue).")
        return {"borrado": False, "motivo": "error"}


async def borrar_de_posthog_con_plazo(user_id: str, plazo_s: Optional[float] = None) -> dict:
    """`borrar_de_posthog` con plazo TOTAL: si PostHog tarda, el borrado de la cuenta no espera. Nunca lanza."""
    import asyncio

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(borrar_de_posthog, user_id),
            timeout=DELETE_DEADLINE_S if plazo_s is None else plazo_s,
        )
    except asyncio.TimeoutError:
        logger.warning("[P1-PLAN-LOTE-847] el borrado en PostHog pasó el plazo (el borrado de la cuenta sigue).")
        return {"borrado": False, "motivo": "plazo"}
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-847] borrar_de_posthog_con_plazo lanzó {type(e).__name__} (el borrado sigue).")
        return {"borrado": False, "motivo": "error"}
