# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-228 · 2026-09-25] «Tu plan está listo» con la app cerrada.

El dueño: «cuando se esté generando un plan en el móvil, quiero que se pueda salir de la app y que cuando termine
llegue una notificación, como la de hidratación o la de comer».

Salir ya se podía: la generación corre en el servidor (cola `plan_generation_runs` o la tarea del SSE, que no se
cancela al desconectarse) y el plan se guarda igual. Lo que faltaba era ENTERARSE: el único «Tu plan está listo» era
un toast dentro de la app. Dos canales, como los recordatorios de comida (`utils/avisosDeComida.js`):

  · Web / PWA — este módulo: al pasar el estado de la generación a `complete` (o `failed`), una Web Push. El service
    worker NO la muestra si el usuario tiene la app delante (`solo_si_no_mira`): ahí ya ve la pantalla de carga
    terminar y el toast.
  · App nativa — el teléfono no recibe push del servidor (no hay FCM); lo avisa él mismo con una notificación LOCAL
    (`frontend/src/utils/avisoPlanListo.js`).

El disparo vive en `db_plans.upsert_pending_pipeline`, el cuello de botella por el que pasan las DOS vías de
generación, y solo en la TRANSICIÓN (antes no estaba en ese estado): los dos caminos que marcan `complete` para el
mismo run (el natural y el fallback del done-callback) no avisan dos veces. Knob `MEALFIT_PLAN_READY_PUSH` (True).
tooltip-anchor: P1-PLAN-LOTE-228-AVISO-PLAN-LISTO
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

TITULO_LISTO = "Tu plan está listo 🎉"
CUERPO_LISTO = "Toca para verlo."
TITULO_FALLO = "No pudimos terminar tu plan"
CUERPO_FALLO = "Toca para intentarlo de nuevo."
ETIQUETA = "plan-listo"
RUTA_LISTO = "/dashboard"
RUTA_FALLO = "/plan"

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _activo() -> bool:
    from knobs import _env_bool
    return _env_bool("MEALFIT_PLAN_READY_PUSH", True)


def aviso_para(status: str, estado_previo) -> tuple | None:
    """(título, cuerpo, ruta) del aviso que toca al pasar de `estado_previo` a `status`, o None. Pura.

    - `complete` avisa salvo que YA estuviera en `complete` (el segundo marcado del mismo run).
    - `failed` avisa solo si venía de `generating`: un fallo sin generación en curso no es noticia para el usuario.
    """
    if status == "complete" and estado_previo != "complete":
        return TITULO_LISTO, CUERPO_LISTO, RUTA_LISTO
    if status == "failed" and estado_previo == "generating":
        return TITULO_FALLO, CUERPO_FALLO, RUTA_FALLO
    return None


def avisar_fin_de_generacion(user_id: str, status: str, estado_previo) -> bool:
    """Despacha la Web Push del fin de la generación en segundo plano. Best-effort: nunca lanza.
    Solo cuentas (UUID): un invitado no tiene suscripciones y su id es de sesión."""
    try:
        if not _activo() or not isinstance(user_id, str) or not _UUID.match(user_id):
            return False
        aviso = aviso_para(status, estado_previo)
        if not aviso:
            return False
        titulo, cuerpo, ruta = aviso
        from utils_push import send_push_notification

        def _enviar():
            send_push_notification(user_id, titulo, cuerpo, url=ruta, tag=ETIQUETA, solo_si_no_mira=True)

        # Siempre en segundo plano: el marcado `complete` va en el camino de la generación y `webpush` tiene su tope
        # de 10 s por suscripción. Pool lleno ⇒ se pierde el aviso (la app lo da igual al abrirse), nunca se bloquea.
        from bg_executor import submit_bg_task
        submit_bg_task(_enviar, task_name="aviso_plan_listo")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-228] aviso de fin de generación no enviado: {e!r}")
        return False
