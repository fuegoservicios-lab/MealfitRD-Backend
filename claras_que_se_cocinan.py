# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-735 · 2026-09-28] Las claras de la lista se cocinan también cuando el huevo va a la sartén.

Batería real sobre el 659 (adulto mayor con HTA): «2 huevos» + «1 clara de huevo» en la lista y «casca 2 huevos en la
misma sartén y cocínalos 3-4 minutos hasta que la clara y la yema cuajen»; «3 huevos» + «3 claras de huevo» y «casca los
huevos dentro, tapa y cocina 6-8 minutos hasta que la clara y la yema cuajen». Ningún paso cocina las claras de la lista:
el plato pierde la proteína que las trajo (el tope de yemas, lote 235). El 405 las da por cocinadas porque su detector ve
«cocina … la clara» y «la clara … cuaj», y esa clara es la del huevo ENTERO; y aunque no se engañara, sólo sabe añadirlas
a un huevo DURO.

Aquí, en la cola del contrato, con huevos enteros y «N claras de huevo» en la lista, y ningún paso que cocine las claras
una vez descontadas las frases de punto del huevo entero («hasta que la clara…», «la clara y la yema»):
- si un paso BATE los huevos, las claras se baten con ellos: «Bate los 2 huevos» → «Bate los 2 huevos y las 3 claras de
  huevo»; si los ECHA a la sartén (revoltillo: «agrega los huevos y revuelve»), van con ellos;
- si un paso los CASCA en la sartén o en la salsa, tras esa frase: «Vierte también las 3 claras de huevo junto a los
  huevos y cocínalas hasta que estén firmes y opacas.»
Otra forma (huevo duro: lo hace el 405; plancha, pochado…): no se toca. Las notas tampoco. tooltip-anchor:
P1-PLAN-LOTE-735
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_NOTAS = ("⚠", "💡", "🌱", "⚕", "🤰", "🛒", "🧊")
_LINEA_CLARAS_RE = re.compile(r"^\s*(?P<q>\d+)\s+claras?\s+de\s+huevos?\b", re.IGNORECASE)
_PUNTO_HUEVO_ENTERO_RE = re.compile(r"\bhasta\s+que\s+(?:la|las)\s+claras?\b[^.;]{0,40}|"
                                    r"\b(?:la|las)\s+claras?\s+y\s+(?:la|las)\s+yemas?\b")
_SIN_CLARAS_DETRAS = r"(?!\s+(?:y|con)\s+(?:(?:la|las)\s+)?(?:\d+\s+)?claras?\b)"
_BATE_RE = re.compile(r"\b(?P<v>[Bb]at[ea]\w*)\s+(?:(?:los|las)\s+)?(?:\d+\s+)?huevos?\b" + _SIN_CLARAS_DETRAS)
#: «agrega los huevos y revuelve», «vierte los huevos batidos»: el revoltillo — las claras van con ellos. Nunca los huevos
#: que llegan YA cocidos («añade los huevos ya bien cocidos y cortados»): la clara cruda no llega así
_AGREGA_RE = re.compile(r"\b(?:[Aa]greg(?:a|ue)|[Aa]ñad(?:e|a)|[Vv]iert(?:e|a)|[Ii]ncorpor(?:a|e)|[Rr]evuelv(?:e|a))\s+(?:los|las)\s+"
                        r"(?:\d+\s+)?huevos(?:\s+batidos)?\b"
                        + _SIN_CLARAS_DETRAS
                        + r"(?![^.;]{0,25}\b(?:ya\s+(?:bien\s+)?cocid|duros?\b|cocidos\b|hervidos\b|cortados\b|en\s+rodajas))")
_CASCA_RE = re.compile(r"\bcasca\w*\b[^.;]{0,40}\bhuevos?\b", re.IGNORECASE)
#: «Cocina huevos a la plancha…», «fríe 2 huevos…» (sobre el texto sin acentos): las claras se cuajan aparte
_PLANCHA_RE = re.compile(r"\b(?:cocin|frie|fri|prepar|haz|hac|cuaj)\w*\s+(?:(?:los|las|el|un|unos)\s+)?(?:\d+\s+)?huevos?\b"
                         r"[^.;]{0,50}\b(?:a\s+la\s+plancha|fritos?|estrellados?|en\s+(?:una|la)\s+sarten)")
_UN_HUEVO_RE = re.compile(r"\bcasca\w*\s+(?:el|un|1)\s+huevo\b(?!s)", re.IGNORECASE)
#: ya USADAS: se cascan o se baten junto a los huevos («casca 3 huevos y 2 claras de huevo y bátelos», «bata 3 huevos y
#: 5 claras de huevo»)
_USADAS_RE = re.compile(r"\bcasca\w*[^.;]{0,40}\bclaras?\b|\bclaras?\b[^.;]{0,25}\bbat\w*|\bbat[ae]\w*[^.;]{0,40}\bclaras?\b")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def integrar(meal) -> int:
    """Nº de pasos corregidos (0 o 1); 0 ante cualquier error."""
    try:
        import pasos_cantidades as pc

        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lineas = [str(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)]
        claras = [m for m in (_LINEA_CLARAS_RE.match(x) for x in lineas) if m]
        if len(claras) != 1:
            return 0
        lista = " | ".join(_sa(x) for x in lineas)
        if not re.search(r"\bhuevos?\b", re.sub(r"(?:claras?|yemas?) de huevos?", " ", lista)):
            return 0                                     # sólo claras: lo hace el 405 con el cerrador
        pasos = [(i, p) for i, p in enumerate(rec) if isinstance(p, str) and not p.lstrip().startswith(_NOTAS)]
        texto = " ".join(_sa(p) for _, p in pasos)
        if "dentro de su huevo entero" in texto or "un huevo por cada clara" in texto:
            return 0
        sin_punto = _PUNTO_HUEVO_ENTERO_RE.sub(" ", texto)
        if pc._CLARA_COCINADA_405_RE.search(sin_punto) or _USADAS_RE.search(sin_punto):
            return 0
        n = int(claras[0].group("q"))
        obj = "la clara de huevo" if n == 1 else f"las {n} claras de huevo"
        for rx in (_BATE_RE, _AGREGA_RE):                  # 1) se baten / se echan con los huevos
            hecho = False
            for i, p in pasos:
                mm = rx.search(p)
                if mm:
                    rec[i] = p[:mm.end()] + f" y {obj}" + p[mm.end():]
                    hecho = True
                    break
            if hecho:
                break
        else:
            hecho = False
            for rx, forma in ((_CASCA_RE, "casca"), (_PLANCHA_RE, "plancha")):  # 2) junto a los huevos / 3) aparte
                for i, p in pasos:
                    if _sa(p).startswith(("montaje", "mise en place")):  # en la mise en place no se cocina nada
                        continue
                    frases = re.split(r"(?<=[.;])\s+", p)
                    k = next((j for j, f in enumerate(frases) if rx.search(_sa(f))), None)
                    if k is None:
                        continue
                    if forma == "casca":
                        cocinar = ("cocínala hasta que esté firme y opaca" if n == 1
                                   else "cocínalas hasta que estén firmes y opacas")
                        junto = "junto al huevo" if _UN_HUEVO_RE.search(frases[k]) else "junto a los huevos"
                        extra = f"Vierte también {obj} {junto} y {cocinar}."
                    else:                                  # el huevo a la plancha no se mezcla: las claras, revueltas
                        extra = (f"Cuaja también {obj} en la sartén, revuelta, hasta que esté firme y opaca." if n == 1
                                 else f"Cuaja también {obj} en la sartén, revueltas, hasta que estén firmes y opacas.")
                    f = frases[k].rstrip()
                    frases[k] = (f if f.endswith((".", ";")) else f + ".") + " " + extra
                    rec[i] = " ".join(frases)
                    hecho = True
                    break
                if hecho:
                    break
            if not hecho:
                return 0
        meal["recipe"] = rec
        meal.pop("_display", None)
        logger.info(f"🥚 [P1-PLAN-LOTE-735] «{str(meal.get('name'))[:50]}»: {obj} de la lista, cocinada(s) con los huevos")
        return 1
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-735] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["integrar"]
