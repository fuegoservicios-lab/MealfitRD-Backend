# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-862 · 2026-09-29] «El Toque de Fuego: sin cocción; licúa todo…» pierde el relleno, no la instrucción.

P1-NOCOOK-TDF-STRIP (jul) quita el «El Toque de Fuego» de RELLENO de un plato frío («No requiere cocción», «No aplica»)
para no inyectarle un tiempo de fuego falso. Pero buscaba la frase en CUALQUIER parte del paso y borraba el paso entero:
si el modelo escribió «El Toque de Fuego: sin cocción; coloca el guineo, la manzana y la leche en la licuadora y licúa»,
se iba también la instrucción de licuar (batería real MX de 6d, 29-sep, D2: el batido se quedó sin licuar). Aquí sólo sale
la cláusula de relleno; si lo que queda es una instrucción, el paso se queda con ella. tooltip-anchor: P1-PLAN-LOTE-862
"""
from __future__ import annotations

import re
import unicodedata

_RELLENO_RE = re.compile(r"no\s+aplica\b|no\s+(?:requiere|necesita|lleva|hay)\s+(?:coccion|cocinar|fuego)|sin\s+coccion")
_ACCION_RE = re.compile(r"\b(?:lic[uú]\w*|bat[ei]\w*|mezcl\w*|coloc\w*|pon\b|pón\w*|vierte\w*|vi[eé]rt\w*|agreg\w*|a[nñ]ad\w*|"
                        r"incorpor\w*|integr(?!al)\w*|combin\w*|tritur\w*|proces\w*|remoj\w*|hidrat\w*|refriger\w*|"
                        r"reposa\w*|enfr[ií]a\w*|remueve\w*|revuelve\w*|arma\w*|reparte\w*|rellena\w*|unt\w*|dispon\w*)")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def lista(paso: str) -> list:
    """Lo que ocupa el lugar del paso: [el paso sin relleno] o [] (el paso sobra)."""
    r = resto(paso)
    return [r] if r else []


def resto(paso: str) -> "str | None":
    """El paso sin sus cláusulas de relleno; None si no queda ninguna instrucción (entonces el paso sobra)."""
    try:
        cab = re.match(r"^\s*el\s+toque\s+de\s+fuego\s*(?:\([^)]*\))?\s*:\s*", paso, re.IGNORECASE)
        pre = cab.group(0) if cab else ""
        cuerpo = paso[len(pre):]
        quedan = []
        for cl in re.split(r"(?<=[.;:,])\s+", cuerpo):
            if _RELLENO_RE.search(_sa(cl)) and not _ACCION_RE.search(_sa(_RELLENO_RE.sub(" ", _sa(cl)))):
                continue
            quedan.append(cl)
        texto = " ".join(quedan).strip().lstrip(",;:. ").strip()
        if not texto or not _ACCION_RE.search(_sa(texto)):
            return None
        texto = (texto[0].lower() if pre else texto[0].upper()) + texto[1:]   # «El Toque de Fuego: coloca…»
        if texto[-1] not in ".!?":
            texto = texto.rstrip(",;:") + "."
        return pre + texto
    except Exception:                                                          # noqa: BLE001
        return None
