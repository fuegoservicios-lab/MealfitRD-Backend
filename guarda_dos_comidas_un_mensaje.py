"""[P1-PLAN-LOTE-1024 · J5] «Este fue el desayuno y me comí 3 tacos»: dos comidas en un mensaje, no una.

Caso vivo del dueño (22-sep, batería J5, rojo en todas las corridas): manda la foto de su desayuno y escribe «Este fue el
desayuno y me comí 3 tacos rellenos…». El coach registraba los tacos TAMBIÉN como desayuno; la fusión de la misma franja
(lote 982) los metía dentro del desayuno, el coach lo deshacía y contestaba «los tacos quedaron anotados por separado».
No lo estaban: los tacos se perdían y el usuario leía lo contrario.

Regla determinista, estrecha: si el mensaje NOMBRA la franja de la foto con un deíctico («este fue el desayuno», «esto
es mi almuerzo») Y cuenta otra comida («y me comí…», «y también cené…»), y la comida que se va a registrar repite esa
misma franja cuando ya se creó una fila de esa franja en este turno, la segunda comida va a la siguiente franja principal
que falte hoy (desayuno → almuerzo → cena). Si ya están las tres, queda como «merienda». Fail-open: ante cualquier duda,
None (no se toca nada).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

FRANJAS = ("desayuno", "almuerzo", "cena")


def _plano(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t or "") if not unicodedata.combining(c)).lower()


_RE_DEIXIS = re.compile(
    r"\b(?:este|esta|esto|eso|ese|aqui\s+esta)\s+(?:fue|es|era|seria)\s+(?:el|la|mi)\s+(desayuno|almuerzo|cena|merienda)\b"
)
_RE_OTRA_COMIDA = re.compile(
    r"\b(?:y|ademas|tambien|luego|despues)\b[^.?!]{0,40}?\b(?:me\s+com[ii]|com[ii]|me\s+tom[ee]|me\s+beb[ii]|cene|almorce|"
    r"desayune|merende|me\s+zampe)\b"
)


def franja_de_la_foto(texto: str) -> Optional[str]:
    """La franja que el usuario le puso a la foto con un deíctico, o None."""
    m = _RE_DEIXIS.search(_plano(texto))
    return m.group(1) if m else None


def nueva_franja(texto: str, franja_pedida: Optional[str], franjas_de_este_turno: Iterable[str],
                 franjas_de_hoy: Iterable[str]) -> Optional[str]:
    """Franja corregida para la segunda comida, o None si no hay que tocar nada."""
    try:
        t = _plano(texto)
        foto = franja_de_la_foto(t)
        pedida = (franja_pedida or "").strip().lower()
        if not foto or pedida != foto or foto not in FRANJAS:
            return None
        if foto not in {str(f or "").lower() for f in franjas_de_este_turno}:
            return None                                   # la de la foto aún no se registró: ESTA es la de la foto
        if not _RE_OTRA_COMIDA.search(t[t.find(foto):]):
            return None                                   # no cuenta otra comida después de nombrar la foto
        hoy = {str(f or "").lower() for f in franjas_de_hoy}
        for f in FRANJAS[FRANJAS.index(foto) + 1:]:
            if f not in hoy:
                return f
        return "merienda"
    except Exception:
        return None
