"""[P1-PLAN-LOTE-1086 · 2026-11-18] Un turno del coach no repite su respuesta.

Cuenta de prueba, 18-nov 10:35: el modelo escribió la respuesta ENTERA («Anoté tu desayuno: … aprox. 850 kcal …») junto
con la llamada a `log_consumed_meal`, y tras la herramienta la volvió a escribir casi igual. `_build_final_content_from
_messages` junta los textos de todas las pasadas (para no perder la narración que ya se vio en vivo) y solo quitaba los
repetidos LETRA POR LETRA: salió dos veces, con un « oqo?» suelto entre medias, y así quedó guardado.

Aquí: un bloque ANTERIOR que dice casi lo mismo que uno posterior (mismas palabras con contenido, largo parecido) se
quita; manda el posterior, que es el que ya vio el resultado de la herramienta. La narración corta («Lo anoto…»)
comparte pocas palabras con la confirmación y se queda. El último bloque nunca se toca. Puro, sin red.

Y el caso de verdad era otro: en lo guardado las dos copias van separadas por UN salto de línea (no por el doble con
el que se juntan las pasadas): el modelo se repitió dentro de la MISMA respuesta. `quitar_repeticion_entera` corta un
texto que es «X + basura + X'» (X y X' casi iguales y de largo parecido) y deja la segunda copia. Repaso sobre los
1169 turnos del coach de 60 días: solo cambia el del 18-nov.
"""
from __future__ import annotations

import re
import unicodedata

UMBRAL = 0.6          # parecido (Jaccard de palabras con contenido) desde el que dos bloques «dicen lo mismo»
_MIN_PALABRAS = 8      # por debajo, un bloque es narración corta: no se compara
_VACIAS = frozenset(
    "a al con de del el en es la las lo los me mi para por que se su tu te un una y o ya le les "
    "the and of to in is it you your".split()
)


def _palabras(texto: str) -> set:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return {w for w in re.findall(r"[a-z0-9]+", t) if w not in _VACIAS}


def parecido(a: str, b: str) -> float:
    pa, pb = _palabras(a), _palabras(b)
    if not pa or not pb:
        return 0.0
    return len(pa & pb) / len(pa | pb)


def quitar_casi_repetidos(partes: list) -> list:
    """Las partes sin las ANTERIORES que repiten (casi) una posterior. Nunca lanza; nunca deja la lista vacía."""
    try:
        if not isinstance(partes, list) or len(partes) < 2:
            return partes
        quedan = []
        for i, p in enumerate(partes):
            if i < len(partes) - 1 and len(_palabras(p)) >= _MIN_PALABRAS:
                largo = len(str(p).strip())
                if any(
                    parecido(p, q) >= UMBRAL and 0.5 <= largo / max(1, len(str(q).strip())) <= 2.0
                    for q in partes[i + 1:]
                ):
                    continue
            quedan.append(p)
        return quedan or partes
    except Exception:  # noqa: BLE001
        return partes


UMBRAL_ENTERA = 0.8   # para partir un texto en dos mitades «iguales» se pide más parecido que entre bloques


def quitar_repeticion_entera(texto: str) -> str:
    """Si el texto es la misma respuesta dos veces (partida por un salto de línea), solo la segunda. Si no, igual."""
    try:
        t = str(texto or "")
        if len(_palabras(t)) < 2 * _MIN_PALABRAS:
            return texto
        for k in [m.start() for m in re.finditer(r"\n", t)]:
            x, y = t[:k].strip(), t[k:].strip()
            if len(_palabras(x)) < _MIN_PALABRAS or len(_palabras(y)) < _MIN_PALABRAS:
                continue
            if 0.7 <= len(x) / max(1, len(y)) <= 1.4 and parecido(x, y) >= UMBRAL_ENTERA:
                return y
        return texto
    except Exception:  # noqa: BLE001
        return texto
