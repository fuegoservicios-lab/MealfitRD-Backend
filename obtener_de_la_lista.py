# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-733 · 2026-09-28] «…suficiente lechosa para obtener ½ taza» dice la taza de la lista.

Batería real sobre el 659 (adulto mayor con HTA, día 1, merienda): «pela y corta suficiente lechosa para obtener ½ taza»
con «1¼ tazas de lechosa» en la lista. Los sincronizadores leen «N taza(s) de X»; esta forma pone el alimento DELANTE
(«suficiente X para obtener N taza», «desgrana la granada hasta obtener N taza») y ninguno la veía. Corpus del VPS (5 147
comidas únicas): 6 menciones, 5 desalineadas («… hasta obtener ½ taza» con «¼ taza de granada desgranada»).

Aquí, en la cola del contrato: si el alimento tiene UNA sola línea en la lista y esa línea va en la MISMA unidad casera
(taza, cda, cdta), la cifra del paso pasa a ser la de la lista, con la unidad en su número. Nunca: una línea en gramos
(«60 g de guineo» contra «½ taza»: son dos medidas, no una errata), dos líneas del alimento, una marca de reparto
(mitad, resto, cada…) ni las notas. tooltip-anchor: P1-PLAN-LOTE-733
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_CUENTA = r"(?P<q>\d+\s*[½¼¾⅓⅔]|\d+(?:[.,]\d+)?|[½¼¾⅓⅔])"
_UNIDAD = r"(?P<u>tazas?|cdas?|cdtas?)"
_PASO_RE = re.compile(
    r"\b(?:suficientes?|la|el|las|los)\s+(?P<food>[a-záéíóúñü]+(?:\s+[a-záéíóúñü]+){0,2}?)\s+"
    r"(?:para\s+obtener|hasta\s+(?:obtener|tener|completar|reunir))\s+" + _CUENTA + r"\s*" + _UNIDAD + r"(?![\w])",
    re.IGNORECASE)
_LINEA_RE = re.compile(r"^\s*" + _CUENTA + r"\s*" + _UNIDAD + r"\.?\s+(?:de\s+)?(?P<food>.+)$", re.IGNORECASE)
_NOTAS = ("⚠", "💡", "🌱", "⚕", "🤰", "🛒", "🧊")
_REPARTO_ANTES_RE = re.compile(r"\b(?:mitad|resto|restantes?|otras?|otros?|parte|cada)\b[^.;:]{0,40}$", re.IGNORECASE)
_FRAC = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}
_FORMAS = {"taza": ("taza", "tazas"), "cda": ("cda", "cdas"), "cdta": ("cdta", "cdtas")}


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _toks(txt) -> list:
    return [t for t in re.split(r"[^a-z0-9ñ]+", _sa(txt)) if len(t) >= 4]


def _valor(q) -> float:
    q = str(q).replace(" ", "")
    if q in _FRAC:
        return _FRAC[q]
    if q[-1] in _FRAC:
        return float(q[:-1]) + _FRAC[q[-1]]
    return float(q.replace(",", "."))


def _familia(u) -> str:
    u = _sa(u).rstrip("s")
    return u if u in _FORMAS else ""


def sincronizar(meal) -> int:
    """Nº de menciones re-alineadas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lineas = [str(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)]
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or p.lstrip().startswith(_NOTAS):
                continue
            cambios = []
            for mm in _PASO_RE.finditer(p):
                ft = _toks(mm.group("food"))
                if not ft or _REPARTO_ANTES_RE.search(p[:mm.start()]):
                    continue
                # «corta el melón y la lechosa hasta completar 1 taza y ½ taza respectivamente»: dos alimentos, dos
                # cifras — la primera NO es de la lechosa (replay de la cola)
                resto_frase = re.split(r"(?<=[.;])\s+", p[mm.end():], maxsplit=1)[0]
                if re.match(r"\s*(?:y|e|,)\s*(?:\d|[½¼¾⅓⅔])", resto_frase) or "respectivamente" in resto_frase.lower():
                    continue
                suyas = [ln for ln in lineas if ft[0] in _toks(ln)]
                if len(suyas) != 1:
                    continue
                ml = _LINEA_RE.match(suyas[0])
                if not ml or _familia(ml.group("u")) != _familia(mm.group("u")) or not _familia(mm.group("u")):
                    continue
                if (_toks(ml.group("food")) or [""])[0] != ft[0]:
                    continue
                q_lista = ml.group("q").replace(" ", "")
                if abs(_valor(q_lista) - _valor(mm.group("q"))) < 1e-6:
                    continue
                sing, plur = _FORMAS[_familia(mm.group("u"))]
                cambios.append((mm.start("q"), mm.end("u"), f"{q_lista} {plur if _valor(q_lista) > 1 else sing}"))
            if cambios:
                s = p
                for ini, fin, texto in sorted(cambios, reverse=True):
                    s = s[:ini] + texto + s[fin:]
                rec[i] = s
                n += len(cambios)
        if n:
            meal.pop("_display", None)
            logger.info(f"🥣 [P1-PLAN-LOTE-733] «{str(meal.get('name'))[:50]}»: {n} «para obtener N taza» con la cifra "
                        f"de la lista")
        return n
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-733] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["sincronizar"]
