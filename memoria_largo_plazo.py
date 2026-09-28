# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-717 · 2026-09-28] «Memoria a Largo Plazo» (Configuración → Capacidades): la LECTURA del flag — SSOT.

Lo que promete el interruptor (migración add_long_term_memory_enabled_2026_05_13.sql y su texto en Configuración):
pausada, «la IA no aprende ni consulta lo aprendido. Tus datos guardados se conservan». Hasta hoy se cumplía la mitad:
la EXTRACCIÓN se pausaba en los dos caminos del chat, pero los dos caminos del coach (`agent.py`, normal y streaming)
seguían buscando en `user_facts` e inyectando «MEMORIA VECTORIAL (RAG)» en cada turno; la cola de pendientes
(`process_pending_queue_sync`) y los textos libres de los paneles de Configuración aprendían sin mirarlo; y todas las
lecturas fallaban ABIERTAS: un perfil ilegible contaba como «activada».

Dos lecturas, una por uso:
  · `leer_memoria(user_id)` — la VERDAD o `MemoriaIlegible`. La usa Configuración: con la base caída el GET responde
    503 en vez de inventar «activada» (un interruptor que no refleja nada).
  · `memoria_activa(user_id)` — la decisión de privacidad para CONSULTAR y APRENDER, fail-CLOSED: invitado, sin id o
    base caída ⇒ False. Equivocarse hacia «pausada» cuesta un turno sin recuerdos; hacia «activada», usar lo que el
    usuario pidió no usar.
Sin fila de perfil ⇒ True: la columna es `NOT NULL DEFAULT TRUE` y nadie pausa una fila que no existe (el PATCH del
interruptor exige la fila), así que «sin fila» es el valor por defecto, no un fallo de lectura.

Lo que NO gobierna: las alergias y las condiciones médicas viven en `health_profile` (formulario y perfil), no en
`user_facts`, y siguen llegando al coach por `form_data` (`merge_form_data_with_profile`, `_enrich_clinical_from_profile`):
pausar la memoria no quita ninguna protección clínica. Tampoco los resúmenes de la conversación EN CURSO
(`build_memory_context` por `session_id`): son el hilo del chat, no «lo aprendido».

La lectura va a la base en cada llamada (una fila por clave primaria, sin caché): cambiar el interruptor surte efecto
en el siguiente turno, y una caché fail-open es justo lo que esto viene a quitar.
tooltip-anchor: leer_memoria, memoria_activa, MemoriaIlegible (tests/test_p1_plan_lote_717_memory_sync.py)
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MemoriaIlegible(RuntimeError):
    """La base no respondió: no se sabe si la memoria a largo plazo está activada."""


def _sin_cuenta(user_id: Optional[str]) -> bool:
    return not user_id or str(user_id) == "guest"


def leer_memoria(user_id: str) -> bool:
    """`user_profiles.long_term_memory_enabled` tal cual. Lanza `MemoriaIlegible` si la base falla (nunca inventa).

    Import perezoso de la fachada `db`: este módulo lo importan el chat y el extractor, y los arneses de test que
    sustituyen `db` por un stub no pueden romper su importación."""
    try:
        from db import execute_sql_query
        fila = execute_sql_query(
            "SELECT long_term_memory_enabled FROM user_profiles WHERE id = %s",
            (user_id,), fetch_one=True,
        )
    except Exception as e:
        raise MemoriaIlegible(f"{type(e).__name__}: {e}") from e
    if not fila:
        return True
    valor = fila.get("long_term_memory_enabled") if isinstance(fila, dict) else None
    return True if valor is None else bool(valor)


def memoria_activa(user_id: Optional[str], *, donde: str = "") -> bool:
    """¿Se puede CONSULTAR y APRENDER de este usuario ahora? Fail-closed (ver cabecera)."""
    if _sin_cuenta(user_id):
        return False
    try:
        return leer_memoria(str(user_id))
    except MemoriaIlegible as e:
        logger.warning(
            f"[P1-PLAN-LOTE-717] memoria a largo plazo de {user_id} ilegible{f' ({donde})' if donde else ''}: "
            f"se trata como PAUSADA (ni se consulta ni se aprende): {e}"
        )
        return False
