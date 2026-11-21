# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1141 · 2026-11-21] El escáner sabe qué fotos mandó el usuario hace un rato.

Caso vivo del dueño (21-nov, 01:01): mandó una foto de 3 lascas de queso con «Me voy a comer 3 papitas de las que te
envié con 3 rodajas de queso» — las papitas eran las Lay's de la foto de las 00:42. El escáner solo ve SU foto y el
texto: leyó «papitas» como tostadas de pan y preguntó «¿Las tostadas llevan algo untado?». Tuvo que contestar «Tú sabes,
la que recién envié» y aun así quedó como «papitas tostadas», 345 kcal.

Cada análisis con comida se apunta aquí (en memoria del proceso: el servicio corre con un solo worker; reiniciar solo
olvida unas horas de contexto). Cuando el texto del usuario se refiere a una foto anterior («de las que te envié», «la
que recién mandé», «de esas»), el prompt del escáner recibe lo que se vio en las fotos de las últimas 3 horas.
"""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

_TTL_S = 3 * 3600
_MAX_POR_USUARIO = 6
_lock = threading.Lock()
_memoria: dict = {}
_ACTUAL: ContextVar[Optional[str]] = ContextVar("fotos_recientes_contexto", default=None)

_RE_REFERENCIA = re.compile(
    r"\b(?:(?:que|las|los|la|el|lo)\s+(?:que\s+)?(?:te\s+)?(?:recien\s+)?(?:envie|mande|pase|ensene|subi)"
    r"|recien\s+(?:te\s+)?(?:envie|mande|pase|subi)"
    r"|foto\s+(?:anterior|de\s+antes)|de\s+(?:esas|esos|esa|ese)\b|las\s+de\s+antes|los\s+de\s+antes)",
)


def _plano(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def anotar(user_id, que_se_vio: str) -> None:
    if not user_id or not str(que_se_vio or "").strip():
        return
    ahora = time.time()
    with _lock:
        cola = _memoria.setdefault(str(user_id), deque(maxlen=_MAX_POR_USUARIO))
        cola.append((ahora, " ".join(str(que_se_vio).split())[:140]))


def refiere_a_otra_foto(texto: str) -> bool:
    return bool(_RE_REFERENCIA.search(_plano(texto)))


def contexto(user_id, texto_usuario: str) -> str:
    """Lo visto en sus fotos recientes, si su texto se refiere a una; si no, ""."""
    if not user_id or not refiere_a_otra_foto(texto_usuario):
        return ""
    ahora = time.time()
    with _lock:
        vistas = [t for (ts, t) in (_memoria.get(str(user_id)) or []) if ahora - ts <= _TTL_S]
    if not vistas:
        return ""
    return " | ".join(f"«{v}»" for v in vistas[-4:])


@contextmanager
def scope(texto: str):
    token = _ACTUAL.set(texto or None)
    try:
        yield
    finally:
        _ACTUAL.reset(token)


def actual() -> Optional[str]:
    return _ACTUAL.get()
