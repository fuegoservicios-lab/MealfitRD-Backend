# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-633 · 2026-09-28] El víver que el complemento incorpora sin cocerlo.

Validación del 592 (adulto mayor con hipertensión, día 2): «Tortitas saladas de trigo con revuelto de huevo y palmito,
ensalada fresca y batata al vapor» —ningún paso toca la batata y el Toque termina «Incorpora también batata durante la
preparación»: cruda, contra el nombre que la promete al vapor—. `_ensure_ingredients_used_in_recipe`
(P2-RECIPE-REVERSE-COHERENCE) ya manda al hervor la yuca, el ñame, la yautía y las legumbres (P1-RECIPE-POLISH-5); la
batata, la papa, el plátano, la auyama, el mapuey y el guineo verde, que tampoco se comen crudos, caían al complemento
genérico. Aquí reciben antes su «💡 Cocción previa» tras el Mise en place (texto SSOT del 408,
`pasos_cantidades.nota_hervor_viver`; al vapor si el NOMBRE lo promete) y el complemento sigue incorporándolos, ya
cocidos. La línea que ya dice cómo viene («batata asada», «plátano hervido», «harina de plátano») no se toca. Replay de
5.042 comidas: 1 caso del generador vivo (los otros 7 son «batata asada» del camino degradado: excluidos).
tooltip-anchor: P1-PLAN-LOTE-633
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_VIVER_633 = re.compile(r"\b(batata|papa|platano|auyama|mapuey|guineo|guineito)s?\b")
_YA_COCIDO_633 = re.compile(r"\b(?:asad|cocid|hervid|sancochad|frit|horne|majad|tostad|precocid)\w*|\bal\s+horno\b"
                            r"|\bpure\b|\bharina\b|\bchips?\b|\bhojuelas?\b|\bmangu\b|\btostones?\b")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _nota(linea: str, nombre_plato: str):
    """La «💡 Cocción previa» del víver de `linea`, o `None` si no es de los que no se comen crudos o ya viene cocido."""
    t = _sa(linea)
    m = _VIVER_633.search(t)
    if not m or _YA_COCIDO_633.search(t):
        return None
    k = m.group(1)
    if k.startswith("guine") and "verde" not in t:
        return None                                    # el guineo maduro se come crudo
    import pasos_cantidades as pq
    nombre = pq._NOMBRE_VIVER_408[k]
    if k == "platano":
        nombre += " verde" if "verde" in t else (" maduro" if "maduro" in t else "")
    elif k.startswith("guine"):
        nombre += " verde"
    nota = pq.nota_hervor_viver(k, nombre)
    if re.search(r"\b" + k[:5] + r"\w*(?:\s+\w+){0,2}?\s+al\s+vapor\b", _sa(nombre_plato)):
        nota = nota.replace(": hierve ", ": cocina ", 1).replace(" en agua ", " al vapor ", 1)
    return nota


def cocer(meal, steps: list, faltan: list) -> list:
    """`steps` con la cocción previa de cada víver de `faltan` (las líneas que el complemento va a incorporar);
    `steps` intacto ante cualquier error o si la receta no es una lista."""
    try:
        if not isinstance(meal, dict) or not isinstance(meal.get("recipe"), list) or not isinstance(steps, list):
            return steps
        import coccion_viver as cv
        out = list(steps)
        for linea in faltan or []:
            nota = _nota(str(linea), str(meal.get("name") or ""))
            if nota and nota not in out:
                out = cv.tras_la_mise(out, nota)
                logger.info(f"🍠 [P1-PLAN-LOTE-633] «{str(meal.get('name'))[:50]}»: {linea} con su cocción previa "
                            f"antes del complemento")
        return out
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-633] no-op: {type(e).__name__}: {e}")
        return steps


__all__ = ["cocer"]
