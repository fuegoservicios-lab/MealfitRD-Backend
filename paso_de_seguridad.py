# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-863 · 2026-09-29] Las instrucciones de seguridad (806 huevo, 807 queso) van DENTRO de la receta.

806 y 807 insertaban su instrucción como un paso «💪 …» antes del Montaje. «💪» es la marca de los pasos del CERRADOR de
proteína, y varios pases la tratan como tal: la integración del cerrador la funde en el «El Toque de Fuego», otros la
mueven o la reescriben. Y como el paso nacía junto a una nota («⚠️ … caliéntalo hasta que humee»), cuando algo reescribía
la comida después (la autocrítica del modelo reescribe platos y conserva las notas) el paso se perdía y nadie lo reponía:
el etiquetador del escudo veía la nota y salía. Batería real de embarazo con 806-809 vivos (29-sep, 08:18): «Nabo
crujiente en airfryer con lentejas guisadas y queso fresco» llegó con la nota y SIN el paso de dorar el queso.

Aquí la instrucción se escribe como la dejaría la integración: al final del «El Toque de Fuego» si lo hay, o como su
propio «El Toque de Fuego: …» antes del Montaje si el plato no cocina nada. Y los llamadores la reponen cada vez que
corren (es idempotente: si un paso ya cuece el huevo o calienta el queso, no se añade). tooltip-anchor: P1-PLAN-LOTE-863
"""
from __future__ import annotations

import re

_NOTA_RE = re.compile(r"^\s*(?:⚠|🤰|⚕|🌱|🛡|💡|💪|nota\b)", re.IGNORECASE)


def poner(pasos: list, frase: str) -> list:
    """`pasos` con `frase` (una oración con mayúscula y punto) al final del «El Toque de Fuego»; si no hay, como su propio
    «El Toque de Fuego: …» antes del Montaje (o al final). Devuelve una lista nueva."""
    rec = list(pasos or [])
    i_tdf = next((k for k, p in enumerate(rec) if isinstance(p, str) and not _NOTA_RE.search(p)
                  and re.match(r"^\s*el\s+toque\s+de\s+fuego\b", p, re.IGNORECASE)), None)
    if i_tdf is not None:
        base = rec[i_tdf].rstrip()
        rec[i_tdf] = base + ("" if base.endswith((".", "!", "?")) else ".") + " " + frase
        return rec
    paso = "El Toque de Fuego: " + frase[:1].lower() + frase[1:]
    i_mont = next((k for k, p in enumerate(rec) if isinstance(p, str) and p.strip().lower().startswith("montaje")),
                  len(rec))
    return rec[:i_mont] + [paso] + rec[i_mont:]
