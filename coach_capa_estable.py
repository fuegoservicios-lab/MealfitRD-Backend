# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1015 · A-01] Capa estable por usuario dentro del ÚNICO bloque de sistema.

Lo que midieron las sondas del 14-nov contra `gpt-6-luna` (Chat Completions, con herramientas):

    tras un cebador [S]:   [S][mensaje de USUARIO]                     → 9.530 de 9.554 tokens en caché
                           [S][mensaje SYSTEM|DEVELOPER][usuario]       → 0   (también con cebador, también con herramientas)
    tras un cebador por usuario [S][capa del usuario] (todo system):
                           [S][capa][mensaje de usuario]                → 9.562 de 9.583
                           [S][capa][system con la hora][usuario]       → 0

La caché solo acierta cuando el bloque de sistema es IDÉNTICO y lo que sigue es un mensaje de usuario: un segundo mensaje de
sistema (o developer) cambia el bloque y rompe el acierto. Por eso lo volátil NO puede ir en un segundo mensaje de sistema, y la
regla del dueño (el bloque clínico y de alergias siempre en rol de sistema) obliga a que ese bloque viva DENTRO del bloque de
sistema: estable por usuario, no por turno.

`reordenar()` toma el `system_prompt` ya armado (manual estático + marcador + cola volátil) y mueve a la zona anterior al
marcador los bloques que solo cambian cuando el usuario edita su perfil: su identidad, su bloque clínico, su meta diaria y la
directiva de idioma. Todo lo demás (hora, diario, faltante, nevera, agua, plan del día, memoria, RAG, suplementos…) se queda
detrás del marcador, de donde `_llm_messages_from_state` lo lleva al mensaje del turno.
"""
from __future__ import annotations

import re
from typing import Optional

# Nombres de bloque de `prompt_desglose._VOLATILE_BLOCK_SPECS` que son estables por usuario.
BLOQUES_ESTABLES = frozenset({"identidad", "clinico", "meta_diaria", "idioma_inicial"})


def _cortes(cola: str) -> list[tuple[int, str]]:
    from prompt_desglose import _find_all_block_hits

    return _find_all_block_hits(cola, 0)


def reordenar(system_prompt: str, marcador: str) -> str:
    """`[manual][línea user_id + bloques estables][marcador][bloques volátiles en su orden original]`.

    Sin marcador, o si no hay nada que mover, devuelve el texto tal cual. Conserva cada carácter: solo cambia el orden de
    los tramos, así que `len` no cambia."""
    if not system_prompt or not marcador or marcador not in system_prompt:
        return system_prompt
    cabeza, cola = system_prompt.split(marcador, 1)
    hits = _cortes(cola)
    if not hits:
        return system_prompt

    primero = hits[0][0]
    linea_uid = cola[:primero]                      # «El user_id actual es: …» (propio del usuario ⇒ capa estable)
    estable: list[str] = [linea_uid]
    volatil: list[str] = []
    for i, (pos, nombre) in enumerate(hits):
        fin = hits[i + 1][0] if i + 1 < len(hits) else len(cola)
        tramo = cola[pos:fin]
        (estable if nombre in BLOQUES_ESTABLES else volatil).append(tramo)
    if not any(t for t in estable[1:]):
        return system_prompt                         # nada estable que mover: el texto no cambia
    return cabeza + "".join(estable) + marcador + "".join(volatil)


def partes(system_prompt: str, marcador: str) -> tuple[str, str]:
    """`(bloque de sistema, cola volátil)` tal como las separa `_llm_messages_from_state`."""
    if not system_prompt or marcador not in system_prompt:
        return system_prompt or "", ""
    a, b = system_prompt.split(marcador, 1)
    return a, b.strip()


def huella(bloque_de_sistema: str) -> str:
    import hashlib

    return hashlib.sha256((bloque_de_sistema or "").encode("utf-8", "replace")).hexdigest()[:12]


def nombres_de_herramientas_permitidas(seleccion, todas) -> Optional[list[str]]:
    """Nombres a pasar como `allowed_tools`, o `None` si la selección es el catálogo entero (no hace falta restringir)."""
    sel = [getattr(t, "name", "") for t in seleccion]
    tod = [getattr(t, "name", "") for t in todas]
    if not sel or set(sel) >= set(tod):
        return None
    return [n for n in tod if n in set(sel)]       # en el orden del catálogo completo


def tool_choice_permitidas(nombres: list[str]) -> dict:
    """Forma verificada contra la API real (sonda T5/T6/Q1-Q4 del 14-nov): no forma parte de lo cacheado."""
    return {"type": "allowed_tools",
            "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": n}} for n in nombres]}}
